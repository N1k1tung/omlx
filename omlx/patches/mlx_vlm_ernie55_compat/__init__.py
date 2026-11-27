# SPDX-License-Identifier: Apache-2.0
"""Expose the fused ERNIE 5.5 Flash implementation from oMLX's vendor tree.

The pinned mlx-vlm ships only the Base ERNIE 5.5 module: it cannot load the
Preview checkpoint's mixed FP8/MXFP4 weights, it runs a decode step as ~72 GPU
launches per layer, and it lacks the fused hyper-connection and SiTU paths.
This tree carries the model files from the ``ernie55-preview/8-decode-
pipelining`` fork at the commit that added ``decode_kernels.py``, so oMLX
serves the model without patching the installed mlx-vlm.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SOURCE_URL = "https://github.com/arfung/mlx-vlm/tree/ernie55-preview/8-decode-pipelining"
SOURCE_SHA = "a0a8fcb0e7b52adb2db994b928dfb760bf2369f4"
_VENDOR_MLX_VLM = Path(__file__).resolve().parent / "vendor" / "mlx_vlm"
_APPLIED = False


def _append_package_path(package: Any, path: Path) -> None:
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        return
    path_string = str(path)
    if path_string not in package_path:
        package_path.insert(0, path_string)


def apply_mlx_vlm_ernie55_compat_patch() -> bool:
    """Expose ``mlx_vlm.models.ernie5_5_moe`` from oMLX's vendor tree."""
    global _APPLIED
    if _APPLIED:
        return False

    try:
        import mlx_vlm
        import mlx_vlm.models

        from omlx.patches.deepseek_v4 import apply_pooling_cache_support

        # The HCA layers keep their compressed KV in mlx-vlm's PoolingCache, so
        # the prefix / SSD cache needs that handler registered before any use.
        apply_pooling_cache_support()
        _append_package_path(mlx_vlm, _VENDOR_MLX_VLM)
        _append_package_path(mlx_vlm.models, _VENDOR_MLX_VLM / "models")
        # Discovery may import upstream first; reload from the vendor path.
        for name in [
            name
            for name in list(sys.modules)
            if name == "mlx_vlm.models.ernie5_5_moe"
            or name.startswith("mlx_vlm.models.ernie5_5_moe.")
        ]:
            del sys.modules[name]
        importlib.import_module("mlx_vlm.models.ernie5_5_moe")
        language = importlib.import_module("mlx_vlm.models.ernie5_5_moe.language")
        if not str(Path(language.__file__).resolve()).startswith(
            str(_VENDOR_MLX_VLM.resolve())
        ):
            logger.error(
                "ERNIE 5.5 compat patch did not take effect: language.py resolves to %s",
                language.__file__,
            )
            return False
    except Exception as exc:  # noqa: BLE001 - a broken vendor tree must not abort
        logger.warning("ERNIE 5.5 mlx-vlm registration failed: %s", exc)
        return False

    _APPLIED = True
    logger.info("ERNIE 5.5 mlx-vlm compatibility patch applied")
    return True


def is_applied() -> bool:
    return _APPLIED


__all__ = [
    "SOURCE_SHA",
    "SOURCE_URL",
    "apply_mlx_vlm_ernie55_compat_patch",
    "is_applied",
]
