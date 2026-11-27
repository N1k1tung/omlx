import mlx.core as mx

from omlx.patches import ernie55_prefill


class _Stub:
    model_type = "ernie5_5_moe"

    def __init__(self):
        self.model = lambda inputs, cache, inputs_embeds, hidden_sink=None: mx.zeros(
            (1, 1, 4)
        )
        self.lm_head = lambda h: mx.zeros((1, 1, 8))
        self.multimax = lambda logits: logits


def test_skip_logits_drops_the_vocabulary_projection(monkeypatch):
    from mlx_vlm.models.ernie5_5_moe import language

    monkeypatch.setattr(ernie55_prefill, "_APPLIED", False)
    assert ernie55_prefill.apply_ernie55_prefill_patch(
        type("Model", (), {"model_type": "ernie5_5_moe"})()
    )

    stub = _Stub()
    ids = mx.zeros((1, 1), dtype=mx.int32)
    skipped = language.LanguageModel.__call__(stub, ids, cache=None, skip_logits=True)
    kept = language.LanguageModel.__call__(stub, ids, cache=None)

    assert skipped.logits is None
    assert kept.logits is not None
    assert tuple(kept.logits.shape) == (1, 1, 8)
