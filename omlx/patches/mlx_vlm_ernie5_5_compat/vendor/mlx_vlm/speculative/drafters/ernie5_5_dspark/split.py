"""Load ERNIE 5.5 Flash Preview's bundled DSpark head as a drafter.

The draft tensors live under ``model.mtp.*`` in the target checkpoint, in the
same mixed block-FP8 / MXFP4 storage as the backbone. They are repacked for MLX
the same way and loaded in memory; no standalone drafter folder is required.
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import mlx.core as mx

from ....models.ernie5_5_moe.config import ModelConfig as TextConfig
from ....models.ernie5_5_moe.ernie5_5_moe import prepare_preview_weights
from ....utils import _load_safetensors
from ..mtp_split import MTPSplitter
from .ernie5_5_dspark import Ernie55DsparkDraftModel

_BUNDLED_PREFIX = "model.mtp."
_TARGET_QUANT_PREFIX = "language_model.model.layers."


def unbundle(tensors: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """Rename ``model.mtp.<stage>.*`` to the ERNIE layer layout."""
    out = {}
    for key, value in tensors.items():
        rest = key.removeprefix("language_model.").removeprefix(_BUNDLED_PREFIX)
        if rest.partition(".")[0].isdigit():
            out[f"model.layers.{rest}"] = value
        else:
            out[f"model.{rest}"] = value
    return out


class Ernie55DsparkSplitter(MTPSplitter):
    output_model_type = "ernie5_5_dspark"
    draft_model_cls = Ernie55DsparkDraftModel
    tokenizer_files = ()

    def select_keys(self, key: str, text_config: dict) -> bool:
        return key.removeprefix("language_model.").startswith(_BUNDLED_PREFIX)

    def load_shard(self, file: Path, keys: List[str]) -> Dict[str, mx.array]:
        # The released FP8 dtypes need the shared loader's header fallback.
        shard = _load_safetensors(str(file))
        return {key: shard[key] for key in keys}

    def sanitize_ctx(self, text_config: dict):
        return SimpleNamespace(args=TextConfig.from_dict(text_config))

    def extra_config(self, text_config: dict) -> dict:
        proposals = int(text_config.get("dspark_block_size", 0) or 0)
        layer_ids = list(text_config.get("dspark_target_layer_ids", []) or [])
        return {
            "n_mtp_layers": len(layer_ids),
            # The ids are zero-based target layers in capture-after semantics.
            "target_layer_ids": layer_ids,
            "mask_token_id": int(text_config.get("dspark_noise_token_id", 0) or 0),
            "markov_rank": int(text_config.get("dspark_markov_rank", 256) or 256),
            "block_size": proposals + 1 if proposals else 0,
        }

    def prepare(
        self,
        source_path: Path,
        *,
        source_config: Optional[dict] = None,
        block_size: Optional[int] = None,
        **quant_opts,
    ) -> Tuple[dict, Dict[str, mx.array]]:
        if source_config is None:
            with open(source_path / "config.json") as f:
                source_config = json.load(f)
        text_config = self.read_text_config(source_config)

        selected: Dict[str, mx.array] = {}
        for file, keys in self.iter_selected(source_path, text_config):
            selected.update(self.load_shard(file, keys))
        if not selected:
            raise ValueError(f"No DSpark tensors found in {source_path}.")

        packed, quantization = prepare_preview_weights(
            unbundle(selected), source_config
        )
        if quantization is None:
            raise ValueError("Expected the Preview's mixed FP8/MXFP4 DSpark weights")
        weights = self.run_sanitize(packed, text_config)
        quantization = {
            (
                "stages." + key.removeprefix(_TARGET_QUANT_PREFIX)
                if key.startswith(_TARGET_QUANT_PREFIX)
                else key
            ): value
            for key, value in quantization.items()
        }

        draft_config = {
            "model_type": self.output_model_type,
            "text_config": text_config,
            "tie_word_embeddings": False,
            "quantization": quantization,
            "quantization_config": quantization,
        }
        draft_config.update(self.extra_config(text_config))
        if block_size is not None:
            draft_config["block_size"] = int(block_size)
        return draft_config, weights
