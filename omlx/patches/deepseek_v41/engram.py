# SPDX-License-Identifier: MIT
"""Token-normalized Engram lookup with request-local lookback."""

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .gemv import mxfp8_gemv_each
from .quantization import QuantizedProjection, quantize_activation
from .storage import DECODE_ROWS, dedupe_tuple_ids


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with (no clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def _prime(n):
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


class NgramHash:
    """CPU int64 hashing matches the official NumPy RNG, not Qwen's SplitMix64.

    Only the last n-1 compressed token ids live in the request cache. Image
    positions stop lookback; they are not removed from the token timeline.
    """

    def __init__(self, config, token_map):
        self.config = config
        self.token_map = np.asarray(token_map, dtype=np.int64)
        vocab = int(self.token_map.max()) + 1
        if vocab != config.engram_compressed_vocab_size:
            raise ValueError(f"Engram compressed vocabulary mismatch: {vocab}")
        self.pad_id = int(self.token_map[config.engram_pad_id])
        primes, multipliers, seen = [], [], set()
        for layer_id in config.engram_layer_ids:
            groups = []
            for _ in range(config.engram_max_ngram_size - 1):
                current, group = config.engram_vocab_size - 1, []
                for _ in range(config.engram_n_heads):
                    current += 1
                    while not _prime(current) or current in seen:
                        current += 1
                    seen.add(current)
                    group.append(current)
                groups.append(group)
            primes.append(groups)
            rng = np.random.default_rng(10007 * layer_id)
            bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
            multipliers.append(
                rng.integers(0, bound, config.engram_max_ngram_size, dtype=np.int64) * 2
                + 1
            )
        self.primes = np.array(primes, dtype=np.int64)
        self.multipliers = np.array(multipliers, dtype=np.int64)
        flat = self.primes.reshape(len(primes), -1)
        self.offsets = np.cumsum(
            np.concatenate(
                [np.zeros((len(primes), 1), dtype=np.int64), flat[:, :-1]], -1
            ),
            -1,
        )
        if list(flat.sum(-1)) != list(config.engram_num_embeddings):
            raise ValueError("Engram table rows do not match the official prime layout")

    def __call__(self, ids, history=None, image_mask=None):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.min() < 0 or ids.max() >= len(self.token_map):
            raise ValueError("Token id outside Engram tokenizer vocabulary")
        tokens = self.token_map[ids]
        if image_mask is not None:
            tokens = np.where(np.asarray(image_mask), -1, tokens)
        depth = self.config.engram_max_ngram_size - 1
        if history is None:
            history = np.full((len(ids), depth), -1, dtype=np.int64)
        joined = np.concatenate([np.asarray(history), tokens], axis=1)
        positions = np.arange(tokens.shape[1]) + depth
        blocked = np.zeros_like(tokens, dtype=bool)
        lookback = []
        for shift in range(depth + 1):
            source = joined[:, positions - shift]
            blocked |= source == -1
            lookback.append(np.where(blocked, self.pad_id, source))
        product = np.stack(lookback, -1)[:, :, None, :] * self.multipliers
        rolling, hashes = product[..., 0], []
        for shift in range(1, depth + 1):
            rolling = np.bitwise_xor(rolling, product[..., shift])
            hashes.append(rolling[..., None] % self.primes[:, shift - 1])
        return np.concatenate(hashes, -1) + self.offsets, joined[:, -depth:]


# Inputs of up to this many rows (batched decode, MTP verify) keep each row's own
# arithmetic: MXFP8 with one weight read, packed GGUF with one read per three rows,
# any other layout row by row.
def _project_each(p, flat):
    """p applied to each row of flat ([count, k]) on its own, with one weight read for all
    rows (mxfp8_gemv_each), or None if p is not an MXFP8 projection."""
    if not (isinstance(p, QuantizedProjection) and p.mode == "mxfp8" and p.group_size == 32):
        return None
    x = quantize_activation(flat) if p.quantize_input else flat
    parts = [mxfp8_gemv_each(x[i : i + 8], p.weight, p.scales) for i in range(0, x.shape[0], 8)]
    if any(part is None for part in parts):
        return None
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, 0)


def _project_rows(p, flat):
    """p applied to each row of flat ([count, k]) with one weight read per few rows, or
    None unless p is a packed GGUF projection of 2..16 rows. Its rows kernel replays
    each row's single-token accumulation order (gguf_rows_nb), so the result matches the
    per-row calls bit for bit while reading the weight once instead of once per row.

    The chunk and the two-rows-a-simdgroup layout are measured on this projection
    ([25600, 6144] Q6_K, 129 MB, M5 Ultra): one read decodes in 0.33 ms whatever it
    feeds, and from six vectors a thread's accumulators spill, so six rows take 0.49 ms
    as 3+3 where one six-vector call takes 0.88 ms and the per-row calls it replaces took
    1.18 ms. Sixteen rows take 0.88 ms as 4x4 against 1.29 ms as 6x3 and 2.6 ms per row."""
    from .gguf_quant import GGUFProjection, rows_projection

    if not isinstance(p, GGUFProjection) or p.weight.ndim != 2:
        return None
    rows = flat.shape[0]
    if not 2 <= rows <= 16:
        return None
    chunk = 3 if rows <= 8 else 4
    parts = [
        rows_projection(p, flat[i : i + chunk], rows_per_simdgroup=2, simdgroups=4)
        for i in range(0, rows, chunk)
    ]
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, 0)


class Engram(nn.Module):
    def __init__(self, config, table_index):
        super().__init__()
        self.embed = nn.Embedding(
            config.engram_num_embeddings[table_index], config.engram_head_dim
        )
        self.wkv = nn.Linear(
            (config.engram_max_ngram_size - 1)
            * config.engram_n_heads
            * config.engram_head_dim,
            config.dim * (config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = mx.ones((config.hc_mult, config.dim))
        self.k_weight = mx.ones((config.hc_mult, config.dim))
        self._config = config

    def __call__(self, h, ids, image_mask=None):
        c = self._config
        ids_np = np.asarray(ids, dtype=np.int64)
        ids_host, inverse = dedupe_tuple_ids(ids_np)
        rows = self.embed(mx.array(ids_host)).flatten(-2)
        count = math.prod(ids_np.shape[:-1]) if ids_np.ndim >= 2 else 1
        requests = ids_np.shape[0] if ids_np.ndim >= 2 else 1
        if 1 < count <= 8 or (requests > 1 and count <= DECODE_ROWS):
            # Decode-sized inputs (one token for each of B requests, or an MTP verify block
            # of one request): project each row on its own, exactly as a single-token
            # decode does. A multi-row quantized_matmul runs a different MLX kernel whose
            # fp32 sums round differently (the reference leaves GEMM order free, but a
            # token's output would then depend on its batch or on speculation).
            kv = _project_each(self.wkv, rows.reshape(count, -1))
            if kv is None:
                # A packed GGUF weight has a rows kernel that keeps each row's
                # single-token order, so it may read the weight once for several rows.
                kv = _project_rows(self.wkv, rows.reshape(count, -1))
            if kv is None:
                flat = rows.reshape(count, 1, -1)
                kv = mx.concatenate([self.wkv(flat[r : r + 1]) for r in range(count)])
            kv = kv.reshape(*rows.shape[:2], -1)
        else:
            kv = self.wkv(rows)
        if inverse is not None:
            kv = kv[mx.array(inverse, dtype=mx.int32)]
        key, value = mx.split(kv, [c.dim * c.hc_mult], axis=-1)
        key = key.astype(mx.float32).reshape(*h.shape)
        x = h.astype(mx.float32)
        inv = mx.rsqrt(mx.mean(x * x, -1) + c.norm_eps) * mx.rsqrt(
            mx.mean(key * key, -1) + c.norm_eps
        )
        # Reference: weight = q_weight.float() * k_weight.float(); dot = (h * weight * key).sum(-1)
        weight = self.q_weight.astype(mx.float32) * self.k_weight.astype(mx.float32)
        dot = mx.sum(x * weight * key, -1) * inv * c.dim**-0.5
        gate = mx.sigmoid(mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
        # copysign(+sqrt, 0) is positive in the official reference.
        gate = mx.where(dot == 0, mx.sigmoid(mx.array(0.001)), gate)
        if image_mask is not None:
            gate = mx.where(image_mask[..., None], 0, gate)
        return (x + gate[..., None] * value.astype(mx.float32)[..., None, :]).astype(
            h.dtype
        )
