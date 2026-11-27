# Copyright © 2026 Neywa Labs Inc.

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from ...fp8 import MLX_MXFP8_QUANTIZATION, _quantize_fp8_weight
from ..base import InputEmbeddingsFeatures, LanguageModelOutput
from .config import ModelConfig
from .echo import EchoWeights, extract_echo_weights
from .language import LanguageModel


def _pack_block_fp8(weight, scales):
    """Expand Baidu's FP32 block scales onto the native MXFP8 grid."""
    rows, cols = weight.shape
    block_size = 128
    group_size = MLX_MXFP8_QUANTIZATION["group_size"]
    if (
        weight.dtype != mx.uint8
        or scales.dtype != mx.float32
        or cols % group_size
        or scales.shape
        != (
            (rows + block_size - 1) // block_size,
            (cols + block_size - 1) // block_size,
        )
    ):
        raise ValueError(
            f"Invalid block FP8 matrix or scales: {weight.shape}, {scales.shape}"
        )

    bits = scales.view(mx.uint32)
    exponent = bits >> 23
    if (
        mx.any((bits & 0x807FFFFF) != 0).item()
        or mx.any((exponent == 0) | (exponent == 255)).item()
    ):
        return _quantize_fp8_weight(weight, scales)

    exponent = mx.repeat(exponent.astype(mx.uint8), block_size, axis=0)[:rows]
    exponent = mx.repeat(exponent, block_size // group_size, axis=1)[
        :, : cols // group_size
    ]
    return weight.view(mx.uint32, stream=mx.cpu), exponent


def prepare_preview_weights(weights, config):
    """Repack Baidu's mixed FP8/MXFP4 Preview checkpoint for MLX."""
    source = config.get("quantization_config") or {}
    if source.get("expert_dtype") != "mxfp4":
        return weights, None

    # The three DSpark draft layers have a different input and cache contract
    # from the Base model's ECHO head. Only prepare the target here.
    weights = {
        key: value for key, value in weights.items() if not key.startswith("model.mtp.")
    }
    quantization = dict(MLX_MXFP8_QUANTIZATION)
    expert_quantization = {"group_size": 32, "bits": 4, "mode": "mxfp4"}
    projections = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    for key in tuple(weights):
        if not key.endswith("_scale_inv"):
            continue
        weight_key = key.removesuffix("_scale_inv")
        weight, scales = weights.pop(weight_key), weights.pop(key)
        if ".experts." in weight_key:
            if (
                weight.ndim != 2
                or weight.dtype not in (mx.int8, mx.uint8)
                or scales.dtype != mx.uint8
                or scales.shape != (weight.shape[0], weight.shape[1] // 16)
                or weight.shape[1] % 16
            ):
                raise ValueError(f"Invalid packed MXFP4 expert: {weight_key}")
            packed = weight.view(mx.uint32, stream=mx.cpu)
            layer = weight_key.split(".")[2]
            projection = projections[weight_key.rsplit(".", 2)[1]]
            path = f"language_model.model.layers.{layer}.ffn.switch_mlp.{projection}"
            quantization[path] = expert_quantization
        else:
            packed, scales = _pack_block_fp8(weight, scales)
        prefix = weight_key.removesuffix(".weight")
        weights[prefix + ".weight"] = packed
        weights[prefix + ".scales"] = scales
    return weights, quantization


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.language_model = LanguageModel(config)
        self.echo = None

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        return InputEmbeddingsFeatures(
            inputs_embeds=self.language_model.model.embed_tokens(input_ids)
        )

    def __call__(
        self,
        input_ids: mx.array,
        pixel_values: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        **kwargs,
    ) -> LanguageModelOutput:
        return self.language_model(input_ids, cache=cache, **kwargs)

    def sanitize(self, weights):
        auxiliary = extract_echo_weights(weights, self.config)
        if auxiliary:
            self.echo = EchoWeights(self.config)
        backbone = {
            k: v
            for k, v in weights.items()
            if not k.removeprefix("language_model.").startswith("echo.")
        }
        out = {
            f"language_model.{k}": v
            for k, v in self.language_model.sanitize(backbone).items()
        }
        out.update({f"echo.{k}": v for k, v in auxiliary.items()})
        return out

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def quant_predicate(self):
        return self.language_model.quant_predicate

    @property
    def cast_predicate(self):
        return self.language_model.cast_predicate

    def make_cache(self):
        return self.language_model.make_cache()

    def shard(self, group=None):
        self.language_model.shard(group)
