"""ERNIE 5.5 Flash DSpark speculative drafter.

ERNIE 5.5 Flash Preview bundles a DSpark draft head under ``model.mtp.*``:
three ERNIE decoder layers that attend over a sliding window only, a
projection of the target's last three layer outputs, a low-rank Markov head
and a final hyper-connection contraction. It reuses the shared DSpark
machinery of the DeepSeek-V4 drafter (``VanillaMarkov`` block sampling and the
``dflash`` round loop) and swaps in ERNIE layers.

Reference: https://github.com/baidu/vLLM-ERNIE (``models/ernie5_5/nvidia/dspark.py``).
"""

import re
from types import SimpleNamespace
from typing import Callable

import mlx.core as mx
import mlx.nn as nn

from ....models.base import scaled_dot_product_attention
from ....models.cache import RotatingKVCache
from ....models.deepseek_v4.hyper_connection import HyperHead, hc_expand
from ....models.deepseek_v4.language import LocalAttention
from ....models.ernie5_5_moe import decode_kernels
from ....models.ernie5_5_moe.language import (
    HyperConnection,
    LanguageModel,
    LatentMoE,
    VirtualHeads,
    _hc_boundary,
    _linear,
)
from ..deepseek_v4_dspark.deepseek_v4_dspark import DeepseekV4DsparkDraftModel
from .policy import CopyIndex, DepthPolicy

# Draft-level tensors named outside the per-layer ERNIE layout.
_HEAD_RENAMES = {
    "model.feature_projection.": "stages.0.main_proj.",
    "model.feature_norm.": "stages.0.main_norm.",
    "model.markov_head.": "markov_head.",
}


class DSparkAttention(LocalAttention):
    """ERNIE window attention in the DFlash cross-attention layout.

    The projected target hidden states are the accumulating KV context; the
    drafted block supplies the queries and a transient block KV. As in the
    reference, draft layers use plain RoPE at ``rope_theta`` and no compression.
    """

    def __init__(self, config):
        super().__init__(config, 0)
        self.o_gate = nn.Linear(
            config.q_lora_rank, config.o_groups * config.o_lora_rank, bias=False
        )
        self.vha = VirtualHeads(self.n_heads, config.vha_rank)

    def __call__(self, x: mx.array, x_ctx: mx.array, cache: RotatingKVCache):
        qr, q, block_kv = self._queries(x)
        out = self._attend(q, self._context(x_ctx), block_kv, cache)
        return self._project(out, qr)

    def rows(self, x, x_ctx, caches, bounds, ctx_bounds):
        """Several requests' blocks, laid end to end in ``x`` (1, N, D).

        ``bounds[r]`` is where request ``r``'s rows lie in ``x``,
        ``ctx_bounds[r]`` where its context lies in ``x_ctx`` and
        ``caches[r]`` is its own cache. The projections read their weights
        once for all the rows; each request attends over its own context.
        """
        qr, q, block_kv = self._queries(x)
        ctx_kv = self._context(x_ctx)
        out = [
            self._attend(q[:, s:e], ctx_kv[:, cs:ce], block_kv[:, s:e], cache)
            for cache, (s, e), (cs, ce) in zip(caches, bounds, ctx_bounds)
        ]
        return self._project(mx.concatenate(out, axis=1), qr)

    def _context(self, x_ctx):
        return self.kv_norm(_linear(self.wkv, x_ctx))

    def _queries(self, x):
        """The projections that need no position: ``(qr, q, block_kv)``."""
        B, L, _ = x.shape
        qr = self.q_norm(_linear(self.wq_a, x))
        q = _linear(self.wq_b, qr).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        return qr, q, self.kv_norm(_linear(self.wkv, x))

    def _attend(self, q, ctx_kv, block_kv, cache):
        """One request's rows over its own context: (B, L, heads, head_dim)."""
        B, S = ctx_kv.shape[:2]
        offset = cache.offset
        q = self.rope(q.transpose(0, 2, 1, 3), offset + S)
        ctx_kv = self.rope(ctx_kv[:, None], offset)
        block_kv = self.rope(block_kv[:, None], offset + S)
        kv, _ = cache.update_and_fetch(
            ctx_kv, mx.zeros((B, 1, S, 0), dtype=ctx_kv.dtype)
        )
        kv = mx.concatenate([kv, block_kv], axis=2)
        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=None,
            scale=self.scale,
            mask=None,
            sinks=self.attn_sink.astype(q.dtype),
        )
        return self.rope(out, offset + S, inverse=True).transpose(0, 2, 1, 3)

    def _project(self, out, qr):
        """Mix the heads of ``out`` (B, L, heads, D) and project them out."""
        B, L = out.shape[:2]
        out = self.vha(out).reshape(B, L, self.o_groups, -1).transpose(0, 2, 1, 3)
        out = self.wo_a(out).transpose(0, 2, 1, 3).flatten(-2)
        return _linear(self.wo_b, out * mx.sigmoid(_linear(self.o_gate, qr)))


class DSparkStage(nn.Module):
    """One draft stage: an ERNIE decoder layer whose attention reads the
    projected target hidden. Parameter names match the ERNIE layer so the
    checkpoint maps straight in."""

    def __init__(self, config, stage_id: int):
        super().__init__()
        text = config.text_config
        self.attn = DSparkAttention(text)
        self.ffn = LatentMoE(text)
        self.attn_norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.attn_hc = HyperConnection(text)
        self.ffn_hc = HyperConnection(text)
        if stage_id == 0:
            targets = max(len(config.target_layer_ids), 1)
            self.main_proj = nn.Linear(
                text.hidden_size * targets, text.hidden_size, bias=False
            )
            self.main_norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        if stage_id == config.n_mtp_layers - 1:
            self.hc_head = HyperHead(text)
            self.norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)

    def __call__(self, h: mx.array, main_x: mx.array, cache: RotatingKVCache):
        h, pending = self.short(h, main_x, cache)
        return hc_expand(pending[0], h, pending[1], pending[2])

    def short(self, h, main_x, cache, pending=None, bounds=None, ctx_bounds=None):
        """The stage with its last expand left to the caller, as the target's
        ``DecoderLayer.short`` does: every expand then shares a launch with
        the projection that follows it. With ``bounds``, ``h`` holds several
        requests' rows and ``cache`` is one cache for each of them."""
        h, x, post, comb = _hc_boundary(self.attn_hc, self.attn_norm, h, pending)
        if bounds is None:
            x = self.attn(x, main_x, cache)
        else:
            x = self.attn.rows(x, main_x, cache, bounds, ctx_bounds)
        h, x, post, comb = _hc_boundary(self.ffn_hc, self.ffn_norm, h, (x, post, comb))
        return h, (self.ffn(x), post, comb)


class Ernie55DsparkDraftModel(DeepseekV4DsparkDraftModel):
    stage_class = DSparkStage

    # Draft depth follows acceptance; set False to always draft the full block.
    adaptive_depth = True
    # Draft from an earlier occurrence when the reply repeats the request.
    copy_drafts = True
    # Reach probability below which a proposal is not drafted, on top of the
    # policy's own thresholds. A loop that verifies several requests in one
    # forward sets it for the round.
    depth_floor = None
    # What belongs to one request. ``request_state`` and ``use_request`` move
    # it out and in, so one drafter can serve several requests in turn.
    _REQUEST_FIELDS = (
        "accept_lens",
        "draft_lens",
        "_policy",
        "_policy_rounds",
        "_copy",
        "_copy_draft",
        "copy_rounds",
        "copy_accepted",
    )

    def __init__(self, config):
        super().__init__(config)
        self._multimax = None
        self._policy = None
        self._policy_rounds = 0
        self._copy = None
        self._copy_draft = None
        self.copy_rounds = 0
        self.copy_accepted = 0

    # -- per-request state ---------------------------------------------------

    def begin_request(self, prompt_tokens=None):
        """Start a request. ``prompt_tokens`` enables context-copy drafts."""
        self._copy = None
        self._copy_draft = None
        if prompt_tokens is not None and self.copy_drafts:
            self._copy = CopyIndex()
            self._copy.extend(prompt_tokens)

    def observe(self, token):
        """Record a committed token (the first one and every emitted one)."""
        if self._copy is not None:
            self._copy.append(int(token))

    def request_state(self):
        """The current request's state, to hand back to ``use_request``."""
        return {name: getattr(self, name) for name in self._REQUEST_FIELDS}

    def use_request(self, state, depth_floor=None):
        """Continue the request that ``state`` was taken from."""
        for name, value in state.items():
            setattr(self, name, value)
        self.depth_floor = depth_floor

    def reset(self, target_model):
        cache = super().reset(target_model)
        self._policy = DepthPolicy(max(1, int(self.config.block_size) - 1))
        self._policy_rounds = 0
        self._copy_draft = None
        self.copy_rounds = 0
        self.copy_accepted = 0
        return cache

    def choose_next_block_size(self, block_total: int) -> int:
        """Block (anchor + proposals) for the next round."""
        limit = max(1, int(block_total) - 1)
        policy = self._policy
        if policy is None:
            return block_total
        # Fold in the rounds verified since the last call.
        for accepted, drafted in zip(
            self.accept_lens[self._policy_rounds :],
            self.draft_lens[self._policy_rounds :],
        ):
            if self._copy_draft is not None:
                self.copy_accepted += int(accepted)
            else:
                policy.update(int(accepted), int(drafted))
        self._policy_rounds = len(self.accept_lens)
        self._copy_draft = None
        if self._copy is not None:
            candidate = self._copy.continuation(limit)
            if candidate:
                self._copy_draft = candidate
                return len(candidate) + 1
        if not self.adaptive_depth:
            return block_total
        return policy.proposals(limit, self.depth_floor) + 1

    def reach(self, proposals: int) -> float:
        """Estimated chance that the current request's round accepts its
        first ``proposals`` drafts; a caller sharing out verify rows between
        requests gives the next row to the one most likely to use it."""
        if self._copy_draft is not None and len(self._copy_draft) >= proposals:
            # A run the head agreed to copy is accepted nearly whole.
            return 0.9
        policy = self._policy
        if policy is None or proposals >= len(policy.reach):
            return 0.0
        return float(policy.reach[max(0, int(proposals))])

    def validate_target_compatibility(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        config = target.config
        own = self.args
        fields = ("hidden_size", "vocab_size", "num_hidden_layers", "hc_mult")
        if config.model_type != "ernie5_5_moe" or any(
            getattr(config, k) != getattr(own, k) for k in fields
        ):
            raise ValueError("DSpark drafter is incompatible with the target model")
        if max(self.config.target_layer_ids) >= config.num_hidden_layers:
            raise ValueError("DSpark target layers are outside the target model")

    def bind(self, target_model):
        self.validate_target_compatibility(target_model)
        lm = getattr(target_model, "language_model", target_model)
        self._input_embed = lm.model.embed_tokens
        # The target's fp32 projection, without MultiMax: that follows the
        # Markov bias.
        self._lm_head_fn = lm.project_head
        self._multimax = lm.multimax
        return self

    def _hidden(self, inputs, target_hidden, cache, bounds=None, ctx_bounds=None):
        first = self.stages[0]
        main_x = first.main_norm(first.main_proj(target_hidden))
        h = self._input_embed(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :], (h.shape[0], h.shape[1], self.hc_mult, h.shape[-1])
        )
        h = mx.contiguous(h)
        # With ``bounds``, ``cache`` holds one cache per request; a stage
        # then takes its own cache of every request.
        stage_caches = cache if bounds is None else zip(*cache)
        pending = None
        for stage, stage_cache in zip(self.stages, stage_caches):
            h, pending = stage.short(
                h, main_x, stage_cache, pending, bounds, ctx_bounds
            )
        h = hc_expand(pending[0], h, pending[1], pending[2])
        last = self.stages[-1]
        return last.norm(last.hc_head(h))

    def _logits(self, hidden: mx.array) -> mx.array:
        return self._lm_head_fn(hidden)

    def draft_block(
        self,
        last_bonus,
        hidden: mx.array,
        cache,
        block_size: int,
        sampler: Callable[[mx.array], mx.array],
        token_dtype: mx.Dtype = mx.int32,
        base: mx.array = None,
    ) -> mx.array:
        """Proposals for one block. ``base`` is this request's share of a
        ``draft_block_bases`` forward, which then is not run again."""
        if self._multimax is None:
            raise RuntimeError(
                "bind(target_model) must be called before draft_block()."
            )
        # Always run the head: it also appends the committed positions to the
        # draft context.
        tokens = self._draft_tokens(
            last_bonus, hidden, cache, block_size, sampler, base
        )
        tokens = tokens.astype(token_dtype)
        candidate = self._copy_draft
        if candidate is None:
            return tokens
        proposals = int(block_size) - 1
        anchor = last_bonus if isinstance(last_bonus, int) else None
        if (
            tokens.shape[0] != 1
            or len(candidate) != proposals
            or (anchor is not None and self._copy.tokens[-1] != anchor)
            or tokens[0, 0].item() != candidate[0]
        ):
            # The head disagrees on the next token, so this is not a verbatim
            # run. Its own draft is verified instead.
            self._copy_draft = None
            return tokens
        self.copy_rounds += 1
        return mx.array([candidate], dtype=token_dtype)

    def _inputs(self, anchor, proposals):
        """The anchor of each row followed by the masks it drafts into."""
        masks = mx.full(
            (anchor.shape[0], proposals - 1),
            int(self.config.mask_token_id),
            dtype=anchor.dtype,
        )
        return mx.concatenate([anchor[:, None], masks], axis=1)

    def draft_block_bases(self, anchors, hiddens, caches, block_sizes):
        """The head's logits for several requests' blocks, from one forward.

        Request ``r`` drafts ``block_sizes[r] - 1`` proposals after the token
        ``anchors[r]``, reading ``hiddens[r]`` into its own ``caches[r]``.
        The rows of all the requests go through the stages and the vocabulary
        head together; only the attention is per request. Returns one
        (1, proposals, vocab) array for each request, to pass to
        ``draft_block`` as ``base``, and ``None`` for a request that drafts
        nothing.
        """
        bases = [None] * len(block_sizes)
        live = [r for r, size in enumerate(block_sizes) if int(size) > 1]
        if not live:
            return bases
        inputs, bounds, ctx_bounds = [], [], []
        start = ctx_start = 0
        for r in live:
            proposals = int(block_sizes[r]) - 1
            anchor = mx.array([int(anchors[r])], dtype=mx.int32)
            inputs.append(self._inputs(anchor, proposals))
            bounds.append((start, start + proposals))
            start += proposals
            length = hiddens[r].shape[1]
            ctx_bounds.append((ctx_start, ctx_start + length))
            ctx_start += length
        hidden = self._hidden(
            mx.concatenate(inputs, axis=1),
            mx.concatenate([hiddens[r] for r in live], axis=1),
            [caches[r] for r in live],
            bounds,
            ctx_bounds,
        )
        logits = self._logits(hidden)
        for r, (s, e) in zip(live, bounds):
            bases[r] = logits[:, s:e]
        return bases

    def _draft_tokens(self, last_bonus, hidden, cache, block_size, sampler, base=None):
        """The head's proposals for one block, sampled left to right."""
        token_dtype = mx.int32
        proposals = int(block_size) - 1
        if proposals <= 0:
            batch = 1 if isinstance(last_bonus, int) else int(last_bonus.shape[0])
            return mx.zeros((batch, 0), dtype=token_dtype)
        anchor = (
            mx.array([last_bonus], dtype=token_dtype)
            if isinstance(last_bonus, int)
            else last_bonus.reshape(-1).astype(token_dtype)
        )
        if base is None:
            inputs = self._inputs(anchor, proposals)
            base = self._logits(self._hidden(inputs, hidden, cache))
        head, multimax = self.markov_head, self._multimax
        correction = head.markov_w2
        weight = correction.get("weight") if type(correction) is nn.Linear else None
        if weight is None or "bias" in correction or base.dtype != mx.float32:
            return head.sample_block(
                base,
                first_prev_tokens=anchor,
                sampler=lambda logits: sampler(multimax(logits)),
            )
        # Each step adds the Markov correction of the previous token to its
        # base logits and applies MultiMax, in one pass over the correction.
        shape = multimax.packed()
        previous = anchor
        sampled = []
        for step in range(proposals):
            embedded = head.get_prev_embeddings(previous).astype(mx.float32)
            logits = decode_kernels.vocab_head(embedded, weight, shape, base[:, step])
            if logits is None:
                logits = multimax(head.apply_step_logits(base[:, step], previous))
            previous = sampler(logits).reshape(-1)
            sampled.append(previous[:, None])
        return mx.concatenate(sampled, axis=1)

    def sanitize(self, weights: dict) -> dict:
        """Map draft tensors in the ERNIE layer layout onto the drafter.

        ``self`` only needs ``args`` (the ERNIE text config), so the splitter
        can call this without building the model. Expects the output of
        ``split.unbundle`` after ``prepare_preview_weights`` has repacked it.
        Weights that are already in the drafter layout pass through.
        """
        if not any(key.startswith("model.") for key in weights):
            return weights
        args = self.args
        renamed = {
            key: value
            for key, value in weights.items()
            # The confidence head is unused by the dflash round loop.
            if not key.startswith("model.confidence_head.")
        }
        stages = {
            int(m[1])
            for key in renamed
            if (m := re.match(r"model\.layers\.(\d+)\.", key))
        }
        if not stages:
            raise ValueError("No DSpark stage tensors found")
        last = max(stages)

        # Reuse the backbone's layer mapping on a config that has only the
        # draft stages, all of them non-MLA.
        context = SimpleNamespace(
            config=SimpleNamespace(
                num_hidden_layers=last + 1,
                o_groups=args.o_groups,
                o_lora_rank=args.o_lora_rank,
                n_routed_experts=args.n_routed_experts,
                compress_ratios=[0] * (last + 1),
            )
        )
        mapped = LanguageModel.sanitize(context, renamed)

        out = {}
        for key, value in mapped.items():
            if key == "model.markov_head.markov_w1.weight":
                # Stored as [rank, vocab]; the embedding wants [vocab, rank].
                value = mx.contiguous(value.T)
            if key.startswith("model.layers."):
                key = "stages." + key.removeprefix("model.layers.")
            elif key.startswith("model.hc_head."):
                key = f"stages.{last}.hc_head." + key.removeprefix("model.hc_head.")
            elif key.startswith("model.norm."):
                key = f"stages.{last}.norm." + key.removeprefix("model.norm.")
            else:
                for src, dst in _HEAD_RENAMES.items():
                    if key.startswith(src):
                        key = dst + key.removeprefix(src)
                        break
                else:
                    raise ValueError(f"Unexpected DSpark tensor: {key}")
            out[key] = value
        return out


Model = Ernie55DsparkDraftModel
