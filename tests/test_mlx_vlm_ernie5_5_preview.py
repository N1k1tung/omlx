import mlx.core as mx

from omlx.patches.mlx_vlm_ernie5_5_preview import (
    is_preview_config,
    prepare_preview_weights,
)

CONFIG = {
    "model_type": "ernie5_5_moe",
    "quantization_config": {"quant_method": "fp8", "expert_dtype": "mxfp4"},
}


def test_prepare_preview_weights_repacks_both_layouts():
    expert = "model.layers.1.block_sparse_moe.experts.0.w1"
    dense = "model.layers.0.mlp.up_proj"
    grouped = "model.layers.0.self_attn.linear_o_group_proj"
    expert_weight = mx.arange(64 * 32, dtype=mx.uint8).reshape(64, 32)
    expert_scales = mx.full((64, 2), 127, dtype=mx.uint8)
    dense_weight = mx.arange(256 * 64, dtype=mx.uint8).reshape(256, 64)
    dense_scales = mx.array([[1.0], [0.5]])
    raw = {
        expert + ".weight": expert_weight.view(mx.int8),
        expert + ".weight_scale_inv": expert_scales,
        dense + ".weight": dense_weight,
        dense + ".weight_scale_inv": dense_scales,
        grouped: mx.zeros((128, 128), dtype=mx.uint8),
        grouped + "_scale_inv": mx.array([[1.0]]),
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
    assert weights[grouped].dtype == mx.bfloat16
    assert weights[grouped].shape == (128, 128)
    assert grouped + ".scales" not in weights


def test_prepare_preview_weights_ignores_base_checkpoints():
    raw = {"model.layers.0.mlp.up_proj.weight": mx.zeros((2, 2))}
    config = {"model_type": "ernie5_5_moe", "quantization_config": {"quant_method": "fp8"}}

    assert not is_preview_config(config)
    assert prepare_preview_weights(raw, config) == (raw, None)
