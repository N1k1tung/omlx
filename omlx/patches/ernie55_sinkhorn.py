# SPDX-License-Identifier: Apache-2.0
"""Run ERNIE 5.5 Flash's hyper-connection sinkhorn in one Metal launch.

``HyperConnection`` normalizes the mHC mixers, then runs the sinkhorn as a
dozen separate elementwise ops plus a second pass for the collapse, twice per
layer. mlx-vlm's DeepSeek V4 package ships a fused sinkhorn/collapse kernel
for the same arithmetic; this points the model's hyper-connections at it when
the shape matches (``hc_mult`` 4, epsilon 1e-6, Metal GPU).

The collapse it writes is bit-identical to the op chain. The two mixer gates
it also emits differ from the op chain by one fp32 epsilon (the kernel sums
the four mixer rows in another order), so a long generation can diverge from
the reference at the same point a bf16 rounding would.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_MODEL_TYPE = "ernie5_5_moe"
_APPLIED = False


def apply_ernie55_sinkhorn_patch(model: Any) -> bool:
    """Swap in the fused sinkhorn/collapse kernel for the model's layers."""
    global _APPLIED
    if _APPLIED:
        return False
    if getattr(model, "model_type", None) != _MODEL_TYPE:
        return False

    try:
        from mlx_vlm.models.deepseek_v4.hyper_connection import _hc_kernel, _hc_ops
        from mlx_vlm.models.ernie5_5_moe import language
    except ImportError:
        return False

    def hyper_connection_call(self, x):
        flat = x.astype(mx.float32).flatten(-2)
        mixes = (flat @ self.fn.T) / (
            mx.sqrt(mx.mean(flat * flat, axis=-1, keepdims=True)) + self.norm_eps
        )
        use_ops = (
            self.mult != 4
            or self.training
            or mx.default_device() != mx.gpu
            or not mx.metal.is_available()
            or _hc_kernel is None
        )
        collapse = _hc_ops if use_ops else _hc_kernel
        return collapse(
            x,
            x.astype(mx.float32),
            mixes,
            self.scale,
            self.base,
            self.mult,
            self.iters,
            self.eps,
        )

    language.HyperConnection.__call__ = hyper_connection_call
    _APPLIED = True
    logger.info("ERNIE 5.5 fused sinkhorn installed")
    return True
