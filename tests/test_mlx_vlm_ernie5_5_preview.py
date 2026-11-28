from pathlib import Path

import mlx.core as mx

from omlx.patches.mlx_vlm_ernie5_5_compat import (
    _VENDOR_MLX_VLM,
    apply_mlx_vlm_ernie5_5_compat_patch,
    is_applied,
)

CONFIG = {
    "model_type": "ernie5_5_moe",
    "quantization_config": {"quant_method": "fp8", "expert_dtype": "mxfp4"},
}


def _prepare_preview_weights():
    if not is_applied():
        assert apply_mlx_vlm_ernie5_5_compat_patch()
    from mlx_vlm.models.ernie5_5_moe import ernie5_5_moe

    assert str(Path(ernie5_5_moe.__file__).resolve()).startswith(
        str(_VENDOR_MLX_VLM.resolve())
    )
    return ernie5_5_moe.prepare_preview_weights


def test_prepare_preview_weights_repacks_both_layouts():
    prepare_preview_weights = _prepare_preview_weights()

    expert = "model.layers.1.block_sparse_moe.experts.0.w1"
    dense = "model.layers.0.mlp.up_proj"
    expert_weight = mx.arange(64 * 32, dtype=mx.uint8).reshape(64, 32)
    expert_scales = mx.full((64, 2), 127, dtype=mx.uint8)
    dense_weight = mx.arange(256 * 64, dtype=mx.uint8).reshape(256, 64)
    dense_scales = mx.array([[1.0], [0.5]])
    raw = {
        expert + ".weight": expert_weight.view(mx.int8),
        expert + ".weight_scale_inv": expert_scales,
        dense + ".weight": dense_weight,
        dense + ".weight_scale_inv": dense_scales,
        "model.mtp.0.block_sparse_moe.experts.0.w1.weight": mx.zeros((1,)),
        "model.embed_tokens.weight": mx.zeros((2, 2)),
    }

    weights, quantization = prepare_preview_weights(raw, CONFIG)

    assert weights[expert + ".weight"].view(mx.uint8).tolist() == (
        expert_weight.tolist()
    )
    assert weights[expert + ".scales"].tolist() == expert_scales.tolist()
    assert weights[dense + ".weight"].shape == (256, 16)
    dense_scales_out = weights[dense + ".scales"]
    assert dense_scales_out.shape == (256, 2)
    assert dense_scales_out[0].tolist() == [127, 127]
    assert dense_scales_out[127].tolist() == [127, 127]
    assert dense_scales_out[128].tolist() == [126, 126]
    assert dense_scales_out[255].tolist() == [126, 126]
    assert quantization == {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
        "language_model.model.layers.1.ffn.switch_mlp.gate_proj": {
            "group_size": 32,
            "bits": 4,
            "mode": "mxfp4",
        },
    }
    assert "model.mtp.0.block_sparse_moe.experts.0.w1.weight" not in weights
    assert "model.embed_tokens.weight" in weights


def test_prepare_preview_weights_ignores_base_checkpoints():
    prepare_preview_weights = _prepare_preview_weights()
    raw = {"model.layers.0.mlp.up_proj.weight": mx.zeros((2, 2))}
    config = {
        "model_type": "ernie5_5_moe",
        "quantization_config": {"quant_method": "fp8"},
    }

    assert prepare_preview_weights(raw, config) == (raw, None)
