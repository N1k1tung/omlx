# SPDX-License-Identifier: Apache-2.0
"""Skip ERNIE 5.5 Flash's vocabulary projection on intermediate prefill chunks.

The scheduler prefills a long prompt in chunks and only the last chunk needs
logits, but the model's forward always projects its hidden states over the
201k-token vocabulary and runs MultiMax in FP32: about 1.5 GiB for a 2,048-token
chunk. oMLX's cache-only prefill contract passes ``skip_logits`` for those
chunks; this makes the language model honor it like the other families in
``_SKIP_LOGITS_MODEL_TYPES``.
"""

from __future__ import annotations

import logging
from typing import Any

import mlx.core as mx

logger = logging.getLogger(__name__)

_MODEL_TYPE = "ernie5_5_moe"
_APPLIED = False


def apply_ernie55_prefill_patch(model: Any) -> bool:
    """Let the language model return no logits for cache-only prefill chunks."""
    global _APPLIED
    if _APPLIED:
        return False
    if getattr(model, "model_type", None) != _MODEL_TYPE:
        return False

    try:
        from mlx_vlm.models.base import LanguageModelOutput
        from mlx_vlm.models.ernie5_5_moe import language
    except ImportError:
        return False

    def call(self, inputs=None, cache=None, inputs_embeds=None, **kwargs):
        hidden_sink = [] if kwargs.get("return_hidden", False) else None
        h = self.model(inputs, cache, inputs_embeds, hidden_sink=hidden_sink)
        logits = None
        if not kwargs.get("skip_logits"):
            logits = self.multimax(self.lm_head(h.astype(mx.float32)))
        return LanguageModelOutput(
            logits=logits,
            hidden_states=hidden_sink,
            shared_kv_states={} if kwargs.get("return_shared_kv", False) else None,
        )

    language.LanguageModel.__call__ = call
    _APPLIED = True
    logger.info("ERNIE 5.5 cache-only prefill installed")
    return True
