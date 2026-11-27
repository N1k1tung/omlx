# SPDX-License-Identifier: Apache-2.0
"""Load ERNIE 5.5 Flash Preview on the pinned mlx-vlm.

The Preview checkpoint stores dense projections as E4M3 bytes with FP32 128x128
block scales and its routed experts as packed MXFP4 with ue8m0 scales. Pinned
mlx-vlm only knows the Base layout, so its shared FP8 transform declines the
checkpoint and the sanitizer then tries to dequantize the packed expert bytes.

The vendored model tree carries the repack (``prepare_preview_weights``), so
this wires it into the pinned loader and bounds the materialization of the
156 GB weight set, plus the checkpoint's Python chat tokenizer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_MODEL_TYPE = "ernie5_5_moe"
_APPLIED = False


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
    """Wire the Preview repack, bounded load and tokenizer into pinned mlx-vlm."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_vlm.fp8 as fp8
        import mlx_vlm.utils as utils
    except ImportError:
        return False

    from .mlx_vlm_ernie55_compat import apply_mlx_vlm_ernie55_compat_patch

    if not apply_mlx_vlm_ernie55_compat_patch():
        return False

    if not getattr(fp8.transform_fp8_weights, "_omlx_ernie55_preview", False):
        original_transform = fp8.transform_fp8_weights

        def transform_fp8_weights(
            weights: dict[str, mx.array],
            config: dict[str, Any],
            target_quantization: dict[str, Any] | None = None,
        ) -> tuple[dict[str, mx.array], dict[str, Any] | None]:
            if config.get("model_type") == _MODEL_TYPE:
                from mlx_vlm.models.ernie5_5_moe.ernie5_5_moe import (
                    prepare_preview_weights,
                )

                prepared, quantization = prepare_preview_weights(weights, config)
                if quantization is not None:
                    return prepared, quantization
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
    logger.info("ERNIE 5.5 Flash Preview loader patch applied")
    return True
