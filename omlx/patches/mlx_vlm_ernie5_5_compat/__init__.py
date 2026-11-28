# SPDX-License-Identifier: Apache-2.0
"""Serve ERNIE 5.5 Flash from oMLX's vendor tree on the pinned mlx-vlm.

The pinned mlx-vlm ships the Base model only. The vendored copy also loads the
Preview checkpoint (MXFP4 experts, block FP8 projections, Python chat
tokenizer, DSpark draft head) and carries the decode kernels. The loader
changes that copy needs in shared mlx-vlm files are installed here as wrappers.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import mlx.core as mx
import mlx_vlm.fp8 as vlm_fp8
import mlx_vlm.models as vlm_models
import mlx_vlm.speculative.dflash as vlm_dflash
import mlx_vlm.speculative.drafters as vlm_drafters
import mlx_vlm.utils as vlm_utils
from mlx_vlm.load_progress import LoadingProgress, evaluate_model
from mlx_vlm.models.deepseek_v4 import hyper_connection

_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
_MODEL_TYPE = "ernie5_5_moe"
_VENDORED_PACKAGES = (
    "mlx_vlm.models.ernie5_5_moe",
    "mlx_vlm.speculative.drafters.ernie5_5_mtp",
    "mlx_vlm.speculative.drafters.ernie5_5_dspark",
)
_APPLIED = False


def apply_mlx_vlm_ernie5_5_compat_patch() -> bool:
    """Put the vendored ERNIE 5.5 model and drafters ahead of mlx-vlm's own."""
    global _APPLIED
    if _APPLIED:
        return False

    vlm_models.__path__.insert(0, str(_VENDOR_MLX_VLM / "models"))
    vlm_drafters.__path__.insert(0, str(_VENDOR_MLX_VLM / "speculative" / "drafters"))
    # Model discovery may have imported the pinned copies already.
    for name in list(sys.modules):
        if name in _VENDORED_PACKAGES or name.startswith(
            tuple(package + "." for package in _VENDORED_PACKAGES)
        ):
            del sys.modules[name]

    vlm_drafters.DRAFTER_KIND_BY_MODEL_TYPE["ernie5_5_dspark"] = "dflash"
    _load_preview_weights(
        importlib.import_module("mlx_vlm.models.ernie5_5_moe.ernie5_5_moe")
    )
    _evaluate_layer_by_layer()
    _load_chat_tokenizer(
        importlib.import_module("mlx_vlm.models.ernie5_5_moe.processing_ernie5_5_moe")
    )
    _let_drafter_choose_block_size()
    _read_collapse_factors_as_fp32()

    _APPLIED = True
    return True


def is_applied() -> bool:
    return _APPLIED


def _load_preview_weights(model: ModuleType) -> None:
    stock = vlm_fp8.transform_fp8_weights

    def transform_fp8_weights(weights, config, target_quantization=None):
        weights, quantization = stock(weights, config, target_quantization)
        if quantization is None and config.get("model_type") == _MODEL_TYPE:
            return model.prepare_preview_weights(weights, config)
        return weights, quantization

    vlm_fp8.transform_fp8_weights = transform_fp8_weights


def _evaluate_layer_by_layer() -> None:
    stock = vlm_utils.load_model_weights

    def load_model_weights(
        model,
        config,
        weights,
        model_path,
        *,
        lazy=False,
        progress_callback=None,
        **kwargs,
    ):
        if config.get("model_type") != _MODEL_TYPE:
            return stock(
                model,
                config,
                weights,
                model_path,
                lazy=lazy,
                progress_callback=progress_callback,
                **kwargs,
            )
        model = stock(
            model,
            config,
            weights,
            model_path,
            lazy=True,
            progress_callback=progress_callback,
            **kwargs,
        )
        if not lazy:
            # mlx-vlm evaluates the whole model at once unless a caller watches
            # the load; the stacked experts are too much GPU work for one call.
            evaluate_model(model, LoadingProgress(model_path, progress_callback).update)
        return model

    vlm_utils.load_model_weights = load_model_weights


def _load_chat_tokenizer(processing: ModuleType) -> None:
    stock = vlm_utils.load_processor

    def load_processor(model_path, add_detokenizer=True, eos_token_ids=None, **kwargs):
        if vlm_utils.load_config(model_path).get("model_type") != _MODEL_TYPE:
            return stock(model_path, add_detokenizer, eos_token_ids, **kwargs)
        # mlx-vlm's AutoProcessor hook turns trust_remote_code on by default;
        # the Preview tokenizer is checkpoint Python and needs the caller's own.
        auto_processor = vlm_utils.AutoProcessor
        vlm_utils.AutoProcessor = processing.Ernie55Processor
        try:
            return stock(model_path, add_detokenizer, eos_token_ids, **kwargs)
        finally:
            vlm_utils.AutoProcessor = auto_processor

    vlm_utils.load_processor = load_processor


def _let_drafter_choose_block_size() -> None:
    stock = vlm_dflash._dflash_next_block_size

    def _dflash_next_block_size(
        draft_model, requested_block_total, remaining_budget, initial_block_size=None
    ):
        block_total = min(requested_block_total, remaining_budget)
        choose = getattr(draft_model, "choose_next_block_size", None)
        if block_total <= 1 or not callable(choose):
            return stock(
                draft_model, requested_block_total, remaining_budget, initial_block_size
            )
        return max(2, min(block_total, int(choose(block_total))))

    vlm_dflash._dflash_next_block_size = _dflash_next_block_size


def _read_collapse_factors_as_fp32() -> None:
    stock = hyper_connection._hc_kernel

    def _hc_kernel(x, y, mixes, scale, base, hc_mult, sinkhorn_iters, eps):
        # The kernel reads the factors as fp32; a model cast to half precision
        # holds them in its own type, and reading those as fp32 runs past the end.
        return stock(
            x,
            y,
            mixes.astype(mx.float32),
            scale.astype(mx.float32),
            base.astype(mx.float32),
            hc_mult,
            sinkhorn_iters,
            eps,
        )

    hyper_connection._hc_kernel = _hc_kernel
