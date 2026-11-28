# Copyright © 2026 Neywa Labs Inc.

import json
from pathlib import Path

from transformers import PreTrainedTokenizerFast
from transformers.dynamic_module_utils import get_class_from_dynamic_module


class Ernie55Processor:
    @classmethod
    def from_pretrained(cls, model_path, **kwargs):
        path = Path(model_path) / "tokenizer_config.json"
        config = json.loads(path.read_text()) if path.is_file() else {}
        if config.get("tokenizer_class") == "ErnieChatTokenizer":
            # The Preview checkpoint supplies a Python chat tokenizer. Honor
            # the caller's explicit opt-in while preserving Base completions.
            if not kwargs.get("trust_remote_code"):
                raise ValueError(
                    "ERNIE Preview's tokenizer requires trust_remote_code=True"
                )
            ref = config["auto_map"]["AutoTokenizer"]
            if isinstance(ref, (list, tuple)):
                ref = next(entry for entry in reversed(ref) if entry)
            tokenizer_cls = get_class_from_dynamic_module(ref, model_path)
            return tokenizer_cls.from_pretrained(model_path, **kwargs)
        return PreTrainedTokenizerFast.from_pretrained(model_path, **kwargs)
