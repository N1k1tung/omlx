"""Batched DFlash drafter: ring context, batched forward oracle, scheduler hooks."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx_vlm.speculative.drafters.dflash2.config import DFlash2Config
from mlx_vlm.speculative.drafters.dflash2.dflash2 import DFlash2DraftModel

from omlx.scheduler import Scheduler
from omlx.speculative import dflash_drafter as dd

VOCAB = 64
HIDDEN = 32
TARGET_LAYERS = 6
TARGET_LAYER_IDS = [1, 3, 5]
WINDOW = 12
BLOCK = 4


def _tiny_config(**overrides):
    params = {
        "model_type": "qwen3",
        "hidden_size": HIDDEN,
        "intermediate_size": 48,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "vocab_size": VOCAB,
        "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096,
        "num_target_layers": TARGET_LAYERS,
        "sliding_window": WINDOW,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "tie_word_embeddings": False,
        "dflash_config": {
            "block_size": BLOCK,
            "conv_group_size": 8,
            "conv_kernel_size": 2,
            "mask_token_id": VOCAB - 1,
            "selector_rank": 8,
            "selector_top_k": 4,
            "target_layer_ids": TARGET_LAYER_IDS,
        },
    }
    params.update(overrides)
    return DFlash2Config.from_dict(params)


def _tiny_target():
    embed = nn.Embedding(VOCAB, HIDDEN)
    lm_head = nn.Linear(HIDDEN, VOCAB, bias=False)
    language = SimpleNamespace(
        config={
            "text_config": {
                "hidden_size": HIDDEN,
                "num_hidden_layers": TARGET_LAYERS,
                "vocab_size": VOCAB,
            }
        },
        model=SimpleNamespace(layers=[object()] * TARGET_LAYERS, embed_tokens=embed),
        lm_head=lm_head,
        rollback_speculative_cache=lambda *args, **kwargs: None,
    )
    mx.eval(embed.parameters(), lm_head.parameters())
    return SimpleNamespace(language_model=language)


def _tiny_drafter(seed=0, block_size=BLOCK, verify_mode=None):
    mx.random.seed(seed)
    config = _tiny_config()
    config.block_size = block_size
    model = DFlash2DraftModel(config)
    # Random weights in float32 so both forwards share tight numerics.
    params = {
        key: mx.random.normal(value.shape) * 0.2
        for key, value in nn.utils.tree_flatten(model.parameters())
    }
    model.load_weights(list(params.items()), strict=False)
    model.bind(_tiny_target())
    mx.eval(model.parameters())
    return dd.DFlashDrafter(
        model, block_size=block_size, source_path="tiny", verify_mode=verify_mode
    )


def _captured(n, seed):
    mx.random.seed(seed)
    return [mx.random.normal((1, n, HIDDEN)) for _ in TARGET_LAYER_IDS]


def _run_cycles(drafter, plan, cycle_offset=0):
    """plan: per cycle, {uid: (segment_len, anchor)}; returns proposals per cycle."""
    outputs = []
    for cycle, rows in enumerate(plan, start=cycle_offset):
        jobs = []
        states = []
        for uid, (n, anchor) in rows.items():
            state = SimpleNamespace(
                uid=uid, drafts=None, draft_lps=None, draft_accept_lps=None
            )
            committed = mx.array([anchor], dtype=mx.uint32)
            jobs.append(
                (
                    None,
                    state,
                    _captured(n, seed=1000 * cycle + uid * 7 + n),
                    committed,
                    None,
                )
            )
            states.append((uid, state))
        drafter.draft(jobs)
        outputs.append({uid: state.drafts.tolist() for uid, state in states})
    return outputs


def test_batched_forward_matches_rows_drafted_alone():
    """Padding, masks, vector RoPE offsets and ring writes must not leak across rows."""
    plan = [
        {0: (5, 3), 1: (1, 9)},
        {0: (2, 11), 1: (3, 4), 2: (7, 5)},
        {0: (4, 8), 1: (1, 1), 2: (2, 6)},
        {0: (6, 2), 2: (5, 7)},
        {0: (3, 12), 2: (9, 13)},
    ]
    expected = [{} for _ in plan]
    for uid in (0, 1, 2):
        alone = _tiny_drafter()
        solo_plan = [{uid: rows[uid]} if uid in rows else {} for rows in plan]
        for cycle, rows in enumerate(solo_plan):
            if rows:
                expected[cycle].update(
                    _run_cycles(alone, [rows], cycle_offset=cycle)[0]
                )

    batched = _tiny_drafter()
    got = _run_cycles(batched, plan)
    assert got == expected
    # The ring saw every context token, including the ones that fell out of
    # the window on the long segments.
    assert batched.context_length(0) == sum(rows[0][0] for rows in plan if 0 in rows)
    assert batched.context_length(2) == sum(rows[2][0] for rows in plan if 2 in rows)


def test_adaptive_blocks_are_per_request_and_resize_the_forward(monkeypatch):
    drafter = _tiny_drafter(block_size=8)
    widths = []
    embed = drafter.model._embed_input_tokens

    def observed_embed(inputs):
        widths.append(tuple(inputs.shape))
        return embed(inputs)

    monkeypatch.setattr(drafter.model, "_embed_input_tokens", observed_embed)
    _run_cycles(drafter, [{0: (5, 3), 1: (5, 9)}])
    assert drafter._rows[0].policy is not drafter._rows[1].policy
    for _ in range(4):
        drafter.record_cycle(0, 8, 0, 2_000_000)
    assert drafter.block_limit(0) == 4 and drafter.block_limit(1) == 8
    got = _run_cycles(drafter, [{0: (2, 11), 1: (3, 4)}], cycle_offset=1)[0]
    assert len(got[0]) == 3 and len(got[1]) == 7
    assert widths == [(2, 8), (1, 4), (1, 8)]
    # Splitting the cohort by width retains each row's ring and projections.
    for uid in (0, 1):
        alone = _tiny_drafter(block_size=8)
        _run_cycles(alone, [{uid: (5, 3 if uid == 0 else 9)}])
        if uid == 0:
            for _ in range(4):
                alone.record_cycle(uid, 8, 0, 2_000_000)
        expected = _run_cycles(
            alone, [{uid: (2, 11) if uid == 0 else (3, 4)}], cycle_offset=1
        )[0][uid]
        assert got[uid] == expected
        assert drafter.context_length(uid) == alone.context_length(uid)
    # The native policy probes back up once reduced blocks accept well.
    for _ in range(16):
        drafter.record_cycle(0, 4, 3, 2_000_000)
    assert drafter.block_limit(0) == 8
    for _ in range(4):
        drafter.record_cycle(0, 8, 7, 2_000_000)
    assert drafter._rows[0].policy.mode == "full"
    drafter._row(2).fed = 32768
    assert drafter.block_limit(2) == 4
    drafter.release([0])
    assert drafter.block_limit(0) == 8  # A new request starts with a new policy.
    fixed = _tiny_drafter(block_size=8, verify_mode="dflash")
    assert fixed.block_limit(0, prompt_len=32768) == 8


@pytest.mark.parametrize("expand", [False, True])
def test_adaptive_draft_waits_for_policy_update(expand, monkeypatch):
    drafter, plain = _tiny_drafter(block_size=8), _tiny_drafter(block_size=8)
    for d in (drafter, plain):
        _run_cycles(d, [{0: (3, 5)}])
        if expand:
            for _ in range(4):
                d.record_cycle(0, 8, 0, 2_000_000)
    state = SimpleNamespace(uid=0)
    captured = _captured(4 if expand else 8, seed=90)
    widths = []
    embed = drafter.model._embed_input_tokens

    def observed_embed(inputs):
        widths.append(int(inputs.shape[1]))
        return embed(inputs)

    monkeypatch.setattr(drafter.model, "_embed_input_tokens", observed_embed)
    assert not drafter.predraft(None, state, captured, mx.array([1]), mx.array([7]))
    assert widths == [] and drafter._predraft is None
    for d in (drafter, plain):
        for _ in range(16 if expand else 4):
            d.record_cycle(0, 4 if expand else 8, 3 if expand else 0, 2_000_000)
    assert drafter.context_length(0) == plain.context_length(0) == 3
    for d in (drafter, plain):
        d.draft([(None, state, [c[:, :1] for c in captured], mx.array([7]), None)])
        if d is drafter:
            got = state.drafts.tolist()
    assert got == state.drafts.tolist()
    assert len(got) == (7 if expand else 3)
    assert widths == [8 if expand else 4]
    assert drafter.context_length(0) == plain.context_length(0) == 4


@pytest.mark.parametrize("block_size, verify_mode", [(BLOCK, None), (8, "dflash")])
def test_predraft_adopt_matches_drafting_committed_rows(block_size, verify_mode):
    """A predraft over every verify row equals drafting the committed rows, and
    a discarded one leaves the ring as it was, across ring wrap-around."""

    def cycle(drafter, uid, captured, count, anchor, mode):
        state = SimpleNamespace(
            uid=uid, drafts=None, draft_lps=None, draft_accept_lps=None
        )
        if mode == "plain":
            committed = mx.array([anchor], dtype=mx.uint32)
            drafter.draft(
                [(None, state, [c[:, :count] for c in captured], committed, None)]
            )
        else:
            assert drafter.predraft(
                None, state, captured, mx.array([count - 1]) + 1, mx.array([anchor])
            )
            if mode == "adopt":
                assert drafter.adopt_predraft(state, count) is None
            else:
                drafter.discard_predraft()
                committed = mx.array([anchor], dtype=mx.uint32)
                drafter.draft(
                    [(None, state, [c[:, :count] for c in captured], committed, None)]
                )
        return state.drafts.tolist()

    counts = [2, 4, 1, 4, 3, 2, 4, 1]
    for mode in ("adopt", "discard"):
        plain, other = (
            _tiny_drafter(block_size=block_size, verify_mode=verify_mode)
            for _ in range(2)
        )
        for d in (plain, other):
            d.seed(0, _captured(WINDOW - 3, seed=77))
            d.draft(
                [
                    (
                        None,
                        SimpleNamespace(
                            uid=0, drafts=None, draft_lps=None, draft_accept_lps=None
                        ),
                        [],
                        mx.array([2], dtype=mx.uint32),
                        None,
                    )
                ]
            )
        for step, count in enumerate(counts):
            captured = _captured(BLOCK, seed=500 + step)
            expected = cycle(plain, 0, captured, count, step + 3, "plain")
            assert cycle(other, 0, captured, count, step + 3, mode) == expected
        assert other.context_length(0) == plain.context_length(0)


def test_pending_captures_stay_within_the_window():
    """Decode steps with drafting off keep only the attended rows, at their positions."""
    kept, tail = _tiny_drafter(), _tiny_drafter()
    rows = _captured(30, seed=9)
    for j in range(30):
        kept.observe([0], [layer[:, j : j + 1] for layer in rows])
    assert sum(p.shape[1] for p in kept._rows[0].pending) == kept.ring_slots
    tail._row(0).fed = 30 - tail.ring_slots
    tail.seed(0, [layer[:, 30 - tail.ring_slots :] for layer in rows])
    drafts = []
    for drafter in (kept, tail):
        state = SimpleNamespace(
            uid=0, drafts=None, draft_lps=None, draft_accept_lps=None
        )
        drafter.draft([(None, state, [], mx.array([5], dtype=mx.uint32), None)])
        drafts.append(state.drafts.tolist())
    assert drafts[0] == drafts[1]
    assert kept.context_length(0) == tail.context_length(0) == 30


def test_release_detaches_rows_and_new_cohort_reuses_ring():
    drafter = _tiny_drafter()
    _run_cycles(drafter, [{0: (3, 1), 1: (2, 2)}])
    assert drafter._cohort is not None and drafter._cohort.uids == (0, 1)
    drafter.release([1])
    assert drafter._cohort is None
    assert drafter._rows[0].keys is not None
    _run_cycles(drafter, [{0: (1, 3), 5: (2, 4)}])
    assert drafter._cohort.uids == (0, 5)
    assert drafter.context_length(0) == 4 and drafter.context_length(5) == 2


def test_prefill_seed_binds_to_uid_and_window_slicing():
    drafter = _tiny_drafter()
    drafter.seed_request("req", _captured(3, seed=1))
    drafter.bind_uid("req", 7)
    assert drafter._request_seeds["req"] is drafter._rows[7]
    assert len(drafter._rows[7].pending) == 1
    drafter.release_request("req")

    scheduler = SimpleNamespace(model=SimpleNamespace(_omlx_drafter=drafter))
    request = SimpleNamespace(prompt_token_ids=list(range(30)), request_id="r")
    kwargs = {}
    # Chunk [0, 10) ends before the last WINDOW=12 tokens: nothing to capture.
    assert (
        Scheduler._dflash_prefill_capture(
            scheduler, request, scheduler.model, 0, 10, kwargs
        )
        is None
    )
    assert "capture_layer_ids" not in kwargs
    # Chunk [10, 25) overlaps the window starting at 30 - 12 = 18.
    keep = Scheduler._dflash_prefill_capture(
        scheduler, request, scheduler.model, 10, 15, kwargs
    )
    assert keep == 8
    assert kwargs["capture_layer_ids"] == TARGET_LAYER_IDS
    # A wrapped prefill model (ANE, specprefill) cannot capture.
    assert (
        Scheduler._dflash_prefill_capture(scheduler, request, object(), 10, 15, {})
        is None
    )

    output = SimpleNamespace(hidden_states=_captured(15, seed=2))
    Scheduler._dflash_seed_prefill(scheduler, request, output, keep)
    assert drafter._request_seeds["r"].pending[0].shape == (
        1,
        7,
        HIDDEN * len(TARGET_LAYER_IDS),
    )


def test_sampled_rows_get_sparse_candidate_distributions():
    """Stochastic rows sample from the selector's candidates and expose q."""
    from omlx.utils.sampling import make_sampler

    mx.random.seed(5)
    drafter = _tiny_drafter()
    sampler = make_sampler(temp=1.0)
    rows = []
    for uid, seed in ((0, None), (1, sampler), (2, sampler)):
        rows.append(
            (
                SimpleNamespace(uid=uid),
                drafter._row(uid),
                mx.concatenate(_captured(3, seed=uid + 40), axis=-1),
                mx.array([uid + 1], dtype=mx.int32),
                seed,
            )
        )
    proposals = drafter._draft_batched(rows)
    assert len(proposals) == 3
    greedy_tokens, greedy_q = proposals[0]
    assert greedy_tokens.shape == (1, BLOCK - 1) and greedy_q == []
    for tokens, accept in proposals[1:]:
        assert tokens.shape == (1, BLOCK - 1)
        assert len(accept) == BLOCK - 1
        for position, q in enumerate(accept):
            assert q.shape == (VOCAB,)
            probs = mx.exp(q)
            # Mass lives on at most top_k candidates and includes the draft.
            assert (probs > 0).sum().item() <= drafter.model.candidate_selector.top_k
            assert abs(probs.sum().item() - 1.0) < 1e-3
            assert probs[tokens[0, position]].item() > 0


def test_short_context_matches_reference_draft_block():
    """Ring slots not yet written must stay out of attention."""
    for length in (1, 3, WINDOW - 1):
        drafter = _tiny_drafter()
        context = mx.concatenate(_captured(length, seed=60 + length), axis=-1)
        anchor = mx.array([7], dtype=mx.int32)
        cache = drafter.model.make_cache()
        for layer_cache in cache:
            layer_cache.offset = 0
        expected = drafter.model.draft_block(
            anchor, context, cache, BLOCK, lambda logits: mx.argmax(logits, axis=-1)
        )
        row = drafter._row(0)
        got = drafter._draft_batched(
            [(SimpleNamespace(uid=0), row, context, anchor, None)]
        )[0][0]
        assert got.tolist() == expected.tolist()


def test_prefill_capture_accepts_bound_prefill_of_the_model():
    class Model:
        def _omlx_prefill(self, *args, **kwargs):
            return None

    model = Model()
    model._omlx_drafter = _tiny_drafter()
    scheduler = SimpleNamespace(model=model)
    request = SimpleNamespace(prompt_token_ids=list(range(30)), request_id="r")
    kwargs = {}
    keep = Scheduler._dflash_prefill_capture(
        scheduler, request, model._omlx_prefill, 10, 15, kwargs
    )
    assert keep == 8 and kwargs["capture_layer_ids"] == TARGET_LAYER_IDS
    other = Model()
    assert (
        Scheduler._dflash_prefill_capture(
            scheduler, request, other._omlx_prefill, 10, 15, {}
        )
        is None
    )


class _Selector(nn.Module):
    def __init__(self, vocab, rank, hidden):
        super().__init__()
        self.top_k = 16
        self.predecessor_codebook = nn.Embedding(vocab, rank)
        self.successor_codebook = nn.Embedding(vocab, rank)
        self.hidden_projection = nn.Linear(hidden, rank, bias=False)


def test_fused_selector_matches_candidate_sampling():
    """One-launch selector: q equals ``_sample_candidates`` on the same path."""
    from omlx.utils.sampling import make_sampler, top_k_indices

    mx.random.seed(8)
    vocab, rank, hidden, batch, length = 600, 64, 32, 2, BLOCK - 1
    selector = _Selector(vocab, rank, hidden)
    selector.update(
        nn.utils.tree_map(lambda p: (p * 3).astype(mx.bfloat16), selector.parameters())
    )
    states = mx.random.normal((batch, length, hidden)).astype(mx.bfloat16)
    logits = (mx.random.normal((batch, length, vocab)) * 4).astype(mx.bfloat16)
    anchors = mx.array([3, 9], dtype=mx.int32)
    candidates = top_k_indices(logits, 16)
    unary = mx.take_along_axis(logits, candidates, axis=-1).astype(mx.float32)
    projected = selector.hidden_projection(states).astype(mx.float32)
    for sampler in (make_sampler(temp=1.0, top_p=0.9, top_k=12), None):
        assert dd._fused_select_eligible(selector, states)
        proposals = dd._select_fused(
            selector, states, logits, anchors, [sampler, sampler]
        )
        for row, (tokens, accept) in enumerate(proposals):
            tokens = tokens.reshape(-1).tolist()
            previous = int(anchors[row])
            for position in range(length):
                edges = mx.sum(
                    selector.predecessor_codebook.weight[previous].astype(mx.float32)
                    * projected[row, position]
                    * selector.successor_codebook.weight[
                        candidates[row, position]
                    ].astype(mx.float32),
                    axis=-1,
                )
                scores = (unary[row, position] + edges)[None]
                picked = candidates[row, position].tolist().index(tokens[position])
                if sampler is None:
                    assert picked == int(mx.argmax(scores[0]).item())
                else:
                    _, expected = dd._sample_candidates(scores, sampler)
                    got = accept.logq[position]
                    assert mx.allclose(got, expected[0], atol=1e-4).item()
                    assert got[picked].item() > -float("inf")
                previous = tokens[position]


def test_conv_kernel_matches_grouped_dynamic_convolve():
    from mlx_vlm.speculative.drafters.dflash2.dflash2 import (
        GroupedDynamicCausalConv,
    )

    mx.random.seed(4)
    conv = GroupedDynamicCausalConv(256, 2, 16)
    conv.base_kernel = (mx.random.normal((2, 2, 256)) * 0.5).astype(mx.bfloat16)
    conv.kernel_projection.weight = (mx.random.normal((64, 256)) * 0.05).astype(
        mx.bfloat16
    )
    x = mx.random.normal((3, BLOCK, 256)).astype(mx.bfloat16)
    expected, dynamic = conv.prepare(x)
    got, packed = dd._conv_prepare(conv, x)
    assert mx.array_equal(got, expected).item()
    y = expected * 0.5
    assert mx.array_equal(
        dd._conv_finish(conv, y, packed), conv.finish(y, dynamic)
    ).item()


def test_resolve_block_size_clamps_to_trained_block_and_mtp_limit():
    model = SimpleNamespace(config=SimpleNamespace(block_size=8))
    assert dd.resolve_block_size(model, None) == 8
    assert dd.resolve_block_size(model, 5) == 5
    assert dd.resolve_block_size(model, 16) == 8
    model.config.block_size = 16
    assert dd.resolve_block_size(model, None) == 8
    with pytest.raises(ValueError):
        dd.resolve_block_size(model, 1)


@pytest.mark.parametrize("hot_only", [True, False])
@pytest.mark.parametrize("fed", [0, 25])
@pytest.mark.parametrize("quantized", [False, True])
def test_prefix_context_restores_ring_pending_and_same_drafts(
    tmp_path, hot_only, fed, quantized
):
    """Warm hot/SSD contexts resume across ring wraps without replaying the target."""
    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    original, resumed = _tiny_drafter(), _tiny_drafter()
    if quantized:
        for drafter in (original, resumed):
            dd.quantize_drafter(drafter.model, bits=4, group_size=32)
    ssd = PagedSSDCacheManager(
        tmp_path,
        2**20,
        hot_cache_max_bytes=2**20 if hot_only else 0,
        hot_cache_only=hot_only,
        expected_model_name="tiny",
        expected_block_size=4,
    )
    prefix = BlockAwarePrefixCache(
        _tiny_target().language_model,
        PagedCacheManager(block_size=4, model_name="tiny"),
        ssd,
    )
    keys = {"extra_key_ranges": [(4, ("image-a",)), (24, ("image-a", "image-b"))]}
    try:
        original.seed_request("request", _captured(3, seed=22))
        original.bind_uid("request", 7)
        if fed:
            _run_cycles(original, [{7: (fed - 3, 2)}])
        original.seed(7, _captured(2, seed=33))
        boundary = fed + 2 if fed else 5
        tokens = list(range(boundary))
        assert original.store_prefix_context(
            prefix, "request", tokens, boundary, uid=7, **keys
        )
        tip = prefix._mtp_prefix_chain_tip(tokens, boundary, **keys)
        assert not resumed.restore_prefix_context(
            prefix, "wrong-boundary", tip, boundary + 1
        )
        assert resumed.restore_prefix_context(prefix, "resume", tip, boundary)
        resumed.bind_uid("resume", 17)
        # Changing either image identity cannot restore the other image's context.
        different = prefix._mtp_prefix_chain_tip(
            tokens, boundary, extra_key_ranges=[(4, ("image-c",))]
        )
        assert not resumed.restore_prefix_context(prefix, "other", different, boundary)
        states = []
        for drafter, uid in ((original, 7), (resumed, 17)):
            state = SimpleNamespace(uid=uid, stats=None)
            drafter.draft([(None, state, _captured(1, seed=55), mx.array([9]), None)])
            states.append(state)
        assert states[0].drafts.tolist() == states[1].drafts.tolist()
        assert original.context_length(7) == resumed.context_length(17) == boundary + 1
        original.release([7])
        assert original.store_prefix_context(
            prefix, "request", list(range(boundary + 1)), boundary + 1, **keys
        )
        original.release_request("request")
        assert "request" not in original._request_seeds
    finally:
        ssd.close()


def _glm_target_and_drafter(verify_mode=None):
    from omlx.patches.mlx_vlm_mtp import glm5_next_vlm_runtime
    from tests.test_mlx_vlm_glm5_next_compat import _tiny_config as glm_config

    assert glm5_next_vlm_runtime.apply()
    from mlx_vlm.models.glm5_next import Model

    mx.random.seed(31)
    target = Model(glm_config(with_vision=True))
    config = _tiny_config(
        vocab_size=128,
        num_target_layers=2,
        dflash_config={
            "block_size": 16,
            "mask_token_id": 127,
            "target_layer_ids": [0, 1],
            "conv_group_size": 8,
            "conv_kernel_size": 2,
            "selector_rank": 8,
            "selector_top_k": 4,
        },
    )
    model = DFlash2DraftModel(config)
    model.bind(target)
    mx.eval(target.parameters(), model.parameters())
    return target, dd.DFlashDrafter(
        model, block_size=16, source_path="tiny-glm", verify_mode=verify_mode
    )


@pytest.mark.parametrize("length", [5, 260])
def test_glm_captures_match_existing_dflash_adapter_and_observe_decode(length):
    from omlx.models.vlm import VLMModelAdapter
    from omlx.patches.dflash_glm5 import Glm5NextTargetOps

    target, drafter = _glm_target_and_drafter()
    adapter = VLMModelAdapter(target)
    tokens = (mx.arange(length, dtype=mx.uint32) % 64 + 1)[None]
    cache = target.language_model.make_cache()
    out = adapter(
        tokens, cache=cache, capture_layer_ids=[1, 0], capture_from=length - 3
    )
    ops = Glm5NextTargetOps()
    logits, captures = ops.forward_with_hidden_capture(
        target,
        input_ids=tokens,
        cache=target.language_model.make_cache(),
        capture_layer_ids={1, 2},
        logits_last_only=False,
    )
    assert mx.allclose(out.logits, logits, atol=1e-5).item()
    for got, index in zip(out.hidden_states, [2, 1]):
        assert mx.allclose(got, captures[index][:, -3:], atol=1e-5).item()
    dd.attach_drafter(target.language_model, drafter)
    try:
        with drafter.decode_scope([7]):
            adapter(mx.array([[6]], dtype=mx.uint32), cache=cache)
        assert drafter._pending_length(drafter._rows[7]) == 1
        result = dd.bg._call_backbone_captured(
            adapter,
            mx.array([[7, 8]], dtype=mx.uint32),
            cache,
            capture_layer_ids=[0, 1],
        )
        assert len(result[3]) == 2 and len(result[2]) == 1
    finally:
        dd.detach_drafter(target.language_model)


def test_missing_draft_sidecar_replays_only_a_window_and_preserves_blocks(tmp_path):
    from types import MethodType

    from mlx_lm.models.cache import KVCache

    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache

    drafter = _tiny_drafter()
    paged = PagedCacheManager(block_size=4, model_name="tiny")
    ssd = PagedSSDCacheManager(
        tmp_path,
        2**20,
        hot_cache_max_bytes=2**20,
        hot_cache_only=True,
        expected_block_size=4,
    )
    prefix = BlockAwarePrefixCache(_tiny_target().language_model, paged, ssd)
    tokens = list(range(40))
    cache = KVCache()
    data = mx.arange(40)[None, None, :, None].astype(mx.float32)
    cache.update_and_fetch(data, data)
    try:
        stored = prefix.store_cache("stored", tokens, [cache])
        paged.release_for_eviction(stored.block_ids)
        prefix.clear_request_entry("stored")
        table, _ = prefix.fetch_cache("new", tokens)
        scheduler = SimpleNamespace(
            model=SimpleNamespace(_omlx_drafter=drafter),
            block_aware_cache=prefix,
            paged_cache_manager=paged,
            config=SimpleNamespace(paged_cache_block_size=4),
        )
        scheduler._dflash_restore_prefix_context = MethodType(
            Scheduler._dflash_restore_prefix_context, scheduler
        )
        assert Scheduler._dflash_prepare_prefix_context(
            scheduler, SimpleNamespace(request_id="new"), table
        )
        assert (
            table.num_tokens == 28
        )  # 11 context rows, rounded down to 4-token blocks.
        assert all(
            paged.allocated_blocks[bid].ref_count == 1 for bid in table.block_ids
        )
        again, _ = prefix.fetch_cache("again", tokens)
        assert again.num_tokens == 40
        prefix.release_cache("new")
        prefix.release_cache("again")
    finally:
        ssd.close()


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("verify_mode", ["dflash", "adaptive"])
def test_glm_dflash_generation_and_warm_image_prefix_match_greedy(
    tmp_path, image, verify_mode, monkeypatch
):
    from mlx_lm.generate import BatchGenerator

    from omlx.cache.paged_cache import PagedCacheManager
    from omlx.cache.paged_ssd_cache import PagedSSDCacheManager
    from omlx.cache.prefix_cache import BlockAwarePrefixCache
    from omlx.cache.state import extract_cache_states
    from omlx.models.vlm import VLMModelAdapter

    target, drafter = _glm_target_and_drafter(verify_mode=verify_mode)
    adapter = VLMModelAdapter(target)
    assert dd.bg.apply()
    tokens = [1, 120 if image else 2] + list(range(3, 29))
    ids = mx.array([tokens], dtype=mx.uint32)
    embeds = (
        target.get_input_embeddings(
            ids, mx.ones((4, 24)), image_grid_thw=mx.array([[1, 2, 2]])
        ).inputs_embeds
        if image
        else None
    )
    reference_cache = target.language_model.make_cache()
    out = target.language_model(ids, inputs_embeds=embeds, cache=reference_cache)
    expected = []
    for _ in range(12):
        token = int(mx.argmax(out.logits[:, -1], axis=-1).item())
        expected.append(token)
        out = target.language_model(mx.array([[token]]), cache=reference_cache)

    # Deterministically low acceptance exercises the real policy hook and
    # width transitions without relying on random tiny weights.
    wrong = next(token for token in range(128) if token not in expected)
    draft_batched = drafter._draft_batched
    widths = []
    embed = drafter.model._embed_input_tokens

    def observed_embed(inputs):
        widths.append(int(inputs.shape[1]))
        return embed(inputs)

    def rejected_drafts(*args, **kwargs):
        return [
            (mx.full(tokens.shape, wrong, dtype=tokens.dtype), accept)
            for tokens, accept in draft_batched(*args, **kwargs)
        ]

    monkeypatch.setattr(drafter.model, "_embed_input_tokens", observed_embed)
    monkeypatch.setattr(drafter, "_draft_batched", rejected_drafts)

    dd.attach_drafter(target.language_model, drafter)
    rollbacks = []
    original_rollback = target.language_model.rollback_speculative_cache

    def rollback(cache, state, accepted, block_size):
        if block_size > 1:
            rollbacks.append(block_size)
        return original_rollback(cache, state, accepted, block_size)

    monkeypatch.setattr(target.language_model, "rollback_speculative_cache", rollback)
    ssd = PagedSSDCacheManager(
        tmp_path,
        2**24,
        hot_cache_max_bytes=2**24,
        hot_cache_only=True,
        expected_model_name="glm",
        expected_block_size=4,
    )
    prefix = BlockAwarePrefixCache(
        adapter, PagedCacheManager(block_size=4, model_name="glm"), ssd
    )
    cache_keys = {"extra_key_ranges": [(2, ("image-a",))]} if image else {}
    try:
        cache = target.language_model.make_cache()
        boundaries = {}
        for start in range(0, len(tokens) - 1, 4):
            end = min(start + 4, len(tokens) - 1)
            out = adapter(
                ids[:, start:end],
                cache=cache,
                **({"inputs_embeds": embeds[:, start:end]} if image else {}),
                capture_layer_ids=[0, 1],
                skip_lm_head=True,
            )
            assert out.logits is None
            drafter.seed_request("cold", out.hidden_states, start=start)
            states, geometry = extract_cache_states(cache)
            mx.eval([value["state"] for value in states])
            boundaries[end] = states
        table = prefix.store_cache(
            "stored",
            tokens[:-1],
            states,
            model_cache_config=geometry,
            boundary_snapshots=boundaries,
            _store_tail_terminal=True,
            **cache_keys,
        )
        assert table.num_tokens == len(tokens) - 1
        assert drafter.store_prefix_context(
            prefix, "cold", tokens, len(tokens) - 1, **cache_keys
        )
        prefix.paged_cache.release_for_eviction(table.block_ids)
        prefix.clear_request_entry("stored")
        for request_id in ("cold", "warm"):
            rollback_start = len(rollbacks)
            width_start = len(widths)
            if request_id == "warm":
                table, suffix = prefix.fetch_cache(request_id, tokens, **cache_keys)
                assert suffix == tokens[-1:]
                cache = adapter.restore_cache(prefix.reconstruct_cache(table))
                tip = prefix.paged_cache.allocated_blocks[
                    table.block_ids[-1]
                ].block_hash
                assert drafter.restore_prefix_context(
                    prefix, request_id, tip, len(tokens) - 1
                )
            generator = BatchGenerator(
                adapter, max_tokens=12, sampler=lambda lp: mx.argmax(lp, -1)
            )
            try:
                uid = generator.insert(
                    [tokens[-1:]], caches=[cache], all_tokens=[list(tokens)]
                )[0]
                drafter.bind_uid(request_id, uid)
                if image:
                    adapter.set_pending_embeddings(embeds, start_offset=len(tokens) - 1)
                got = []
                for _ in range(48):
                    _, responses = generator.next()
                    got.extend(r.token for r in responses)
                    if len(got) >= 12:
                        break
                assert got == expected
                assert rollbacks[rollback_start] == widths[width_start] == 8
                assert set(rollbacks[rollback_start:]) == (
                    {4, 8} if verify_mode == "adaptive" else {8}
                )
                assert set(widths[width_start:]) == (
                    {4, 8} if verify_mode == "adaptive" else {8}
                )
                # One initial draft and one next draft per verify cycle,
                # including the adaptive width transition.
                assert len(widths) - width_start == len(rollbacks) - rollback_start + 1
            finally:
                generator.close()
                drafter.release_request(request_id)
                prefix.release_cache(request_id)
        assert 8 in rollbacks and max(rollbacks) == 8
    finally:
        dd.detach_drafter(target.language_model)
        ssd.close()
