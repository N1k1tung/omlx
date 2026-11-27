# SPDX-License-Identifier: Apache-2.0
"""Load ERNIE 5.5 Flash Preview on the pinned mlx-vlm.

The Preview checkpoint stores dense projections as E4M3 bytes with FP32 128x128
block scales and its routed experts as packed MXFP4 with ue8m0 scales. Pinned
mlx-vlm only knows the Base layout, so its shared FP8 transform declines the
checkpoint and the sanitizer then tries to dequantize the packed expert bytes.

This rewrites both layouts into native MLX quantized tensors before that
transform runs, so the stock ``mlx_vlm.models.ernie5_5_moe`` module loads it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm.fp8 import _dequantize_fp8_weight, _quantize_fp8_weight

logger = logging.getLogger(__name__)

_MODEL_TYPE = "ernie5_5_moe"
_BLOCK_SIZE = 128
_GROUP_SIZE = 32
_MXFP8 = {"group_size": _GROUP_SIZE, "bits": 8, "mode": "mxfp8"}
_MXFP4 = {"group_size": _GROUP_SIZE, "bits": 4, "mode": "mxfp4"}
_PROJECTIONS = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}

_APPLIED = False


def is_preview_config(config: dict[str, Any]) -> bool:
    """True for the mixed FP8/MXFP4 Preview checkpoint."""
    quantization = config.get("quantization_config") or {}
    return (
        config.get("model_type") == _MODEL_TYPE
        and quantization.get("expert_dtype") == "mxfp4"
    )


def _pack_block_fp8(weight: mx.array, scales: mx.array) -> tuple[mx.array, mx.array]:
    """Expand Baidu's FP32 block scales onto the native MXFP8 grid."""
    if weight.dtype == mx.int8:
        weight = weight.view(mx.uint8)
    rows, cols = weight.shape
    if (
        weight.dtype != mx.uint8
        or scales.dtype != mx.float32
        or cols % _GROUP_SIZE
        or scales.shape
        != (
            (rows + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
            (cols + _BLOCK_SIZE - 1) // _BLOCK_SIZE,
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
        # Not a power-of-two scale, so requantize from the dequantized values.
        return _quantize_fp8_weight(weight, scales)

    exponent = mx.repeat(exponent.astype(mx.uint8), _BLOCK_SIZE, axis=0)[:rows]
    exponent = mx.repeat(exponent, _BLOCK_SIZE // _GROUP_SIZE, axis=1)[
        :, : cols // _GROUP_SIZE
    ]
    return weight.view(mx.uint32, stream=mx.cpu), exponent


def prepare_preview_weights(
    weights: dict[str, mx.array], config: dict[str, Any]
) -> tuple[dict[str, mx.array], dict[str, Any] | None]:
    """Repack a Preview checkpoint into native MLX quantized tensors."""
    if not is_preview_config(config):
        return weights, None

    # The DSpark draft layers have a different cache contract from the Base
    # model's ECHO head and are outside the autoregressive path.
    weights = {
        key: value for key, value in weights.items() if not key.startswith("model.mtp.")
    }
    quantization: dict[str, Any] = dict(_MXFP8)
    prepared: dict[str, mx.array] = {}
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
            projection = _PROJECTIONS[weight_key.rsplit(".", 2)[1]]
            path = f"language_model.model.layers.{layer}.ffn.switch_mlp.{projection}"
            quantization[path] = _MXFP4
        elif weight_key.endswith(".linear_o_group_proj"):
            # The grouped output projection keeps BF16 weights: the sanitizer
            # reshapes this suffixless tensor to [groups, rank, -1] for a
            # plain matmul.
            prepared[weight_key] = _dequantize_fp8_weight(weight, scales).astype(
                mx.bfloat16
            )
            continue
        else:
            packed, scales = _pack_block_fp8(weight, scales)
        prefix = weight_key.removesuffix(".weight")
        prepared[prefix + ".weight"] = packed
        prepared[prefix + ".scales"] = scales

    for key, value in weights.items():
        prepared.setdefault(key, value)
    return prepared, quantization


def _evaluate_layers(model: Any, progress_callback: Any) -> None:
    """Materialize one decoder layer at a time to bound GPU work and peak RSS."""
    language_model = getattr(model, "language_model", model)
    layers = getattr(language_model, "layers", [])
    for index, layer in enumerate(layers):
        mx.eval(layer.parameters())
        mx.clear_cache()
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "evaluating_layers",
                    "completed_layers": index + 1,
                    "total_layers": len(layers),
                }
            )
    if progress_callback is not None:
        progress_callback({"stage": "evaluating_remaining_parameters"})
    mx.eval(model.parameters())


def _load_preview_tokenizer(model_path: str | Path, trust_remote_code: bool) -> Any:
    """Load the checkpoint's Python chat tokenizer.

    The repository advertises it only through ``tokenizer_config.json``'s
    ``auto_map``, which ``AutoTokenizer`` refuses for an unknown model type.
    """
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    path = Path(model_path)
    config_path = path / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    if config.get("tokenizer_class") != "ErnieChatTokenizer":
        raise ValueError(f"{path} does not ship the ERNIE chat tokenizer")
    if not trust_remote_code:
        raise ValueError("ERNIE 5.5 Flash Preview's tokenizer requires trust_remote_code")
    reference = config["auto_map"]["AutoTokenizer"]
    if isinstance(reference, (list, tuple)):
        reference = next(entry for entry in reversed(reference) if entry)
    tokenizer_class = get_class_from_dynamic_module(reference, path)
    return tokenizer_class.from_pretrained(path, trust_remote_code=True)


def _load_preview_processor(
    model_path: str | Path,
    add_detokenizer: bool = True,
    eos_token_ids: Any = None,
    trust_remote_code: bool = False,
    **kwargs: Any,
) -> Any:
    from mlx_vlm.tokenizer_utils import load_tokenizer
    from mlx_vlm.utils import StoppingCriteria

    tokenizer = _load_preview_tokenizer(model_path, trust_remote_code)
    if add_detokenizer:
        detokenizer_class = load_tokenizer(model_path, return_tokenizer=False)
        tokenizer.detokenizer = detokenizer_class(tokenizer)
        tokenizer.stopping_criteria = StoppingCriteria(eos_token_ids, tokenizer)
    return tokenizer


def apply_mlx_vlm_ernie5_5_preview_patch() -> bool:
    """Teach pinned mlx-vlm the Preview weight layout. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_vlm.fp8 as fp8
        import mlx_vlm.utils as utils
    except ImportError:
        return False

    if not getattr(fp8.transform_fp8_weights, "_omlx_ernie55_preview", False):
        original_transform = fp8.transform_fp8_weights

        def transform_fp8_weights(
            weights: dict[str, mx.array],
            config: dict[str, Any],
            target_quantization: dict[str, Any] | None = None,
        ) -> tuple[dict[str, mx.array], dict[str, Any] | None]:
            if is_preview_config(config):
                return prepare_preview_weights(weights, config)
            return original_transform(weights, config, target_quantization)

        transform_fp8_weights._omlx_ernie55_preview = True
        fp8.transform_fp8_weights = transform_fp8_weights

    if not getattr(utils.evaluate_model, "_omlx_ernie55_preview", False):
        original_evaluate = utils.evaluate_model

        def evaluate_model(
            model: Any,
            progress_callback: Any = None,
            *,
            clear_cache: bool = False,
        ) -> None:
            if getattr(model, "model_type", None) != _MODEL_TYPE:
                return original_evaluate(
                    model, progress_callback, clear_cache=clear_cache
                )
            _evaluate_layers(model, progress_callback)

        evaluate_model._omlx_ernie55_preview = True
        utils.evaluate_model = evaluate_model

    if not getattr(utils.load_processor, "_omlx_ernie55_preview", False):
        original_load_processor = utils.load_processor

        def load_processor(
            model_path: str | Path,
            add_detokenizer: bool = True,
            eos_token_ids: Any = None,
            **kwargs: Any,
        ) -> Any:
            config_path = Path(model_path) / "config.json"
            config = json.loads(config_path.read_text()) if config_path.is_file() else {}
            if config.get("model_type") != _MODEL_TYPE:
                return original_load_processor(
                    model_path,
                    add_detokenizer=add_detokenizer,
                    eos_token_ids=eos_token_ids,
                    **kwargs,
                )
            return _load_preview_processor(
                model_path,
                add_detokenizer=add_detokenizer,
                eos_token_ids=eos_token_ids,
                trust_remote_code=bool(kwargs.get("trust_remote_code", False)),
            )

        load_processor._omlx_ernie55_preview = True
        utils.load_processor = load_processor

    _APPLIED = True
    logger.info("ERNIE 5.5 Flash Preview weight patch applied")
    return True
