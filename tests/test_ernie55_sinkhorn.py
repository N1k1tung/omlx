from types import SimpleNamespace

import mlx.core as mx

from omlx.patches.ernie55_sinkhorn import apply_ernie55_sinkhorn_patch


def _hyper_connection():
    from mlx_vlm.models.ernie5_5_moe import language

    config = SimpleNamespace(
        hc_mult=4, hc_sinkhorn_iters=3, hc_eps=1e-6, rms_norm_eps=1e-5, hidden_size=8
    )
    module = language.HyperConnection(config)
    module.fn = mx.random.normal((24, 32)).astype(mx.float32)
    module.base = mx.random.normal((24,)).astype(mx.float32)
    module.scale = mx.array([1.0, 2.0, 0.5])
    return module


def test_fused_sinkhorn_keeps_the_collapsed_residual():
    from mlx_vlm.models.deepseek_v4.hyper_connection import _hc_ops

    module = _hyper_connection()
    x = mx.random.normal((1, 1, 4, 8)).astype(mx.bfloat16)
    flat = x.astype(mx.float32).flatten(-2)
    mixes = (flat @ module.fn.T) / (
        mx.sqrt(mx.mean(flat * flat, axis=-1, keepdims=True)) + module.norm_eps
    )
    reference = _hc_ops(
        x,
        x.astype(mx.float32),
        mixes,
        module.scale,
        module.base,
        module.mult,
        module.iters,
        module.eps,
    )

    assert apply_ernie55_sinkhorn_patch(SimpleNamespace(model_type="ernie5_5_moe"))
    collapsed, post, comb = module(x)

    assert mx.array_equal(collapsed, reference[0]).item()
    assert mx.max(mx.abs(post - reference[1])).item() <= 1e-6
    assert mx.max(mx.abs(comb - reference[2])).item() <= 1e-6
