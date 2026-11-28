# Copyright © 2026 Neywa Labs Inc.

"""ERNIE 5.5 Flash: HCA/MLA, latent multi-router MoE and mHC.

Architecture reference: https://github.com/baidu/vLLM-ERNIE
"""

import re

import mlx.core as mx
import mlx.nn as nn
from mlx.nn.layers.distributed import shard_inplace
from mlx.utils import tree_flatten

from ...fp8 import MLX_MXFP8_QUANTIZATION, _dequantize_fp8_weight
from ...speculative.cache_state import (
    rollback_speculative_cache,
    start_speculative_cache,
)
from ..base import (
    LanguageModelOutput,
    create_attention_mask,
    project_logits,
    scaled_dot_product_attention,
)
from ..cache import CacheList, KVCache, PoolingCache, RotatingKVCache
from ..deepseek_v4.hyper_connection import (
    HyperHead,
    _hc_kernel,
    _hc_ops,
    _hc_sinkhorn_collapse_kernel,
    hc_expand,
)
from ..linear import DECODE_BLOCK_SIZE
from ..deepseek_v4.language import (
    CompressedAttention,
    Compressor,
    _align_local_mask,
    _extend_mask,
)
from ..mla import MultiLinear
from ..switch_layers import SwitchGLU
from . import decode_kernels
from .config import ModelConfig


# Decode and verify blocks hand each layer to the GPU as it is built. Building
# the whole forward first leaves the GPU idle for about 5 ms of the 23 to 44
# ms a block takes. 0 turns it off.
SUBMIT_LAYERS = 1


def _submit(index, caches, *arrays):
    """Start the GPU on the layers built so far.

    The layer's cache state goes with it. An array computed here keeps this
    call's Metal event alive for as long as it lives, and a cache update left
    lazy holds on to the arrays of every step since it was last evaluated:
    43 events a step, against a limit of 65,536.
    """
    if SUBMIT_LAYERS and (index + 1) % SUBMIT_LAYERS == 0:
        mx.async_eval(
            [a for a in arrays if isinstance(a, mx.array)],
            [cache.state for cache in caches if cache is not None],
        )


def _linear(layer, x):
    """A dense projection; a verify block reads the weights once for all rows."""
    if decode_kernels.projects_block(x):
        out = decode_kernels.fp8_linear(layer, x)
        if out is not None:
            return out
    return layer(x)


def _parallel_output(projection, x, group):
    if group is None:
        return _linear(projection, x)
    # Match a full dot product's accumulation before rounding back to BF16.
    out = projection(x.astype(mx.float32))
    return mx.distributed.all_sum(out, group=group).astype(x.dtype)


def _row_major(value):
    """Store a transposed checkpoint tensor in its own row-major buffer.

    A transpose is only a strided view. The hyper-connection projects its
    16384 inputs onto 24 outputs as ``x @ fn.T``; with ``fn`` left as a view of
    the checkpoint's (16384, 24) tensor that product takes MLX's slow
    transposed-GEMV path, 0.38 ms instead of 0.15 ms on an M5 Ultra, 86 times
    per decoded token.
    """
    return mx.contiguous(value)


@mx.compile
def _situ(up, gate, beta, linear_beta):
    dtype = gate.dtype
    gate, up = gate.astype(mx.float32), up.astype(mx.float32)
    gate = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    up = linear_beta * mx.tanh(up / linear_beta)
    return (gate * up).astype(dtype)


class SiTU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.beta = config.activation_situ_beta
        self.linear_beta = config.activation_situ_linear_beta
        self.output_dtype = None

    def __call__(self, up, gate):
        return _situ(up, gate, self.beta, self.linear_beta).astype(
            self.output_dtype or gate.dtype
        )


class MLP(nn.Module):
    def __init__(self, config, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.activation = SiTU(config)
        self.sharding_group = None

    def __call__(self, x):
        activation = self.activation
        hidden = None
        if (
            decode_kernels.projects_block(x)
            and self.sharding_group is None
            and type(activation) is SiTU
            and activation.output_dtype is None
        ):
            # Gate, up and the activation of a verify block in one weight pass.
            hidden = decode_kernels.fp8_gate_up(
                self.gate_proj, self.up_proj, x, activation.beta, activation.linear_beta
            )
        if hidden is None:
            hidden = activation(self.up_proj(x), self.gate_proj(x))
        return _parallel_output(self.down_proj, hidden, self.sharding_group)


class MultiRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.weight = mx.zeros((config.n_routed_experts, config.hidden_size))
        self.weight_1 = mx.zeros_like(self.weight)
        self.routed_scaling_factor_param = mx.full(
            (config.n_routed_experts,), config.routed_scaling_factor
        )
        self.top_k = config.num_experts_per_tok
        self.normalize = config.norm_topk_prob
        # Both routers stacked in fp32, see _scores.
        self._stacked_source = None
        self._stacked = None

    def stacked(self):
        """Both routers stacked in fp32, cast once.

        The released routers are half precision and are scored in fp32, so the
        plain form casts both and runs two products per call. Call this
        outside compiled code: it evaluates the copy when the weights change.
        """
        source = (self.weight, self.weight_1)
        cached = self._stacked_source
        if cached is None or cached[0] is not source[0] or cached[1] is not source[1]:
            self._stacked = mx.concatenate(
                [w.astype(mx.float32) for w in source], axis=0
            )
            mx.eval(self._stacked)
            self._stacked_source = source
        return self._stacked

    def _scores(self, x, stacked):
        """Sum of the two routers' sigmoid scores."""
        if stacked is None:
            scores = mx.sigmoid(x @ self.weight.astype(mx.float32).T)
            return scores + mx.sigmoid(x @ self.weight_1.astype(mx.float32).T)
        if x.ndim >= 2 and x.shape[-2] <= DECODE_BLOCK_SIZE:
            # Row-wise mat-vecs keep a verify block on the one-token path.
            raw = (stacked @ x[..., None]).squeeze(-1)
        else:
            raw = x @ stacked.T
        scores = mx.sigmoid(raw)
        experts = self.weight.shape[0]
        return scores[..., :experts] + scores[..., experts:]

    def __call__(self, x, bias, stacked=None):
        x = x.astype(mx.float32)
        if stacked is None and not self.training:
            stacked = self.stacked()
        scores = self._scores(x, stacked)
        ids = mx.argpartition(-(scores + bias), kth=self.top_k - 1, axis=-1)[
            ..., : self.top_k
        ]
        ids = mx.stop_gradient(ids)
        weights = mx.take_along_axis(scores, ids, axis=-1)
        if self.normalize:
            weights /= mx.maximum(weights.sum(axis=-1, keepdims=True), 1e-20)
        return ids, weights * self.routed_scaling_factor_param[ids]


class LatentMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = MultiRouter(config)
        self.e_score_correction_bias = mx.zeros((config.n_routed_experts,))
        self.fc1_latent_proj = nn.Linear(
            config.hidden_size, config.moe_latent_size, bias=False
        )
        self.switch_mlp = SwitchGLU(
            config.moe_latent_size,
            config.moe_intermediate_size,
            config.n_routed_experts,
            activation=SiTU(config),
        )
        self.latent_norm = nn.RMSNorm(config.moe_latent_size, eps=config.rms_norm_eps)
        self.fc2_latent_proj = nn.Linear(
            config.moe_latent_size, config.hidden_size, bias=False
        )
        self.shared_experts = MLP(
            config, config.n_shared_experts * config.moe_intermediate_size
        )
        self.sharding_group = None
        self._compiled = None
        # A host may offer the routed outputs of a prefill chunk through its
        # own kernels: ``backend(switch_mlp, latent, ids)`` returns them, or
        # None to decline.
        self.routed_backend = None

    def __call__(self, x, router_bias=None):
        bias = self.e_score_correction_bias if router_bias is None else router_bias
        if (
            not self.training
            and self.sharding_group is None
            and x.ndim == 3
            and x.shape[1] <= DECODE_BLOCK_SIZE
        ):
            # Decode and verify blocks run on the fused kernels, or failing
            # that as one compiled graph. The stacked router is prepared here
            # because compiled code cannot evaluate.
            out = self._decode(x, bias) if decode_kernels.ENABLED else None
            if out is not None:
                return out
            if self._compiled is None:
                self._compiled = mx.compile(self._forward, inputs=[self.state])
            return self._compiled(x, bias, self.gate.stacked())
        return self._forward(x, bias)

    def _decode(self, x, bias):
        """A decode block in seven launches besides the shared expert.

        Returns None when the kernels do not cover this layer.
        """
        gate, switch, norm = self.gate, self.switch_mlp, self.latent_norm
        activation = switch.activation
        if (
            type(activation) is not SiTU
            or activation.output_dtype is not None
            or type(norm) is not nn.RMSNorm
            or self.routed_backend is not None
        ):
            return None
        B, L, D = x.shape
        rows = x.reshape(B * L, D)
        scores = decode_kernels.router_scores(rows, gate.weight, gate.weight_1)
        if scores is None:
            return None
        picked = decode_kernels.router_select(
            scores, bias, gate.routed_scaling_factor_param, gate.top_k, gate.normalize
        )
        if picked is None:
            return None
        ids, weights = picked
        latent = _linear(self.fc1_latent_proj, rows)
        outputs = decode_kernels.routed_experts(
            switch, latent, ids, activation.beta, activation.linear_beta
        )
        if outputs is None:
            return None
        combined = decode_kernels.combine_norm(outputs, weights, norm.weight, norm.eps)
        if combined is None:
            return None
        routed = _linear(self.fc2_latent_proj, combined.reshape(B, L, -1))
        return self.shared_experts(x) + routed

    def _forward(self, x, bias, stacked=None):
        ids, weights = self.gate(x, bias, stacked)
        latent = self.fc1_latent_proj(x)
        outputs = None
        if self.routed_backend is not None and self.sharding_group is None:
            outputs = self.routed_backend(self.switch_mlp, latent, ids)
        if outputs is None:
            outputs = self.switch_mlp(latent, ids)
        if self.sharding_group is not None:
            # Reconstruct each expert before weighting and the latent RMSNorm.
            outputs = mx.distributed.all_sum(outputs, group=self.sharding_group)
            outputs = outputs.astype(latent.dtype)
        combined = (
            (outputs.astype(mx.float32) * weights[..., None])
            .sum(axis=-2)
            .astype(x.dtype)
        )
        return self.shared_experts(x) + self.fc2_latent_proj(self.latent_norm(combined))


class HyperConnection(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mult = config.hc_mult
        self.iters = config.hc_sinkhorn_iters
        self.eps = config.hc_eps
        self.norm_eps = config.rms_norm_eps
        count = (2 + self.mult) * self.mult
        self.fn = mx.zeros((count, self.mult * config.hidden_size), dtype=mx.float32)
        self.base = mx.zeros((count,), dtype=mx.float32)
        self.scale = mx.ones((3,), dtype=mx.float32)
        self._compiled = None

    def __call__(self, x):
        if not self.training and x.ndim == 4 and x.shape[1] <= DECODE_BLOCK_SIZE:
            # Decode and verify blocks run as one compiled graph, which fuses
            # the short elementwise chains around the projection and kernel.
            if self._compiled is None:
                self._compiled = mx.compile(self._mix, inputs=[self.state])
            return self._compiled(x)
        return self._mix(x)

    def _mix(self, x):
        flat = x.astype(mx.float32).flatten(-2)
        fn = self.fn.astype(mx.float32)
        if not self.training and x.shape[1] <= DECODE_BLOCK_SIZE:
            # Row-wise mat-vecs: a short verify block then costs what the same
            # tokens cost one at a time, instead of taking the GEMM path.
            projected = (fn @ flat[..., None]).squeeze(-1)
        else:
            projected = flat @ fn.T
        # ERNIE places epsilon outside sqrt in the sublayer mHC mixers.
        mixes = projected / (
            mx.sqrt(mx.mean(flat * flat, axis=-1, keepdims=True)) + self.norm_eps
        )
        use_kernel = (
            not self.training
            and self.mult == 4
            and self.eps == 1e-6
            and _hc_sinkhorn_collapse_kernel is not None
            and mx.default_device() == mx.gpu
        )
        collapse = _hc_kernel if use_kernel else _hc_ops
        return collapse(
            x,
            flat.reshape(x.shape),
            mixes,
            self.scale,
            self.base,
            self.mult,
            self.iters,
            self.eps,
        )


class VirtualHeads(nn.Module):
    def __init__(self, heads, rank):
        super().__init__()
        self.U = mx.zeros((heads, rank))
        self.V = mx.zeros((heads, rank))
        self.sharding_group = None
        self._folded_source = None
        self._folded = None

    def _mix(self):
        """``I + V U^T`` in fp32, folded once per weight load outside training."""
        if self.training:
            return (
                mx.eye(self.U.shape[0])
                + self.V.astype(mx.float32) @ self.U.astype(mx.float32).T
            )
        cached = self._folded_source
        if cached is None or cached[0] is not self.U or cached[1] is not self.V:
            self._folded = (
                mx.eye(self.U.shape[0])
                + self.V.astype(mx.float32) @ self.U.astype(mx.float32).T
            )
            mx.eval(self._folded)
            self._folded_source = (self.U, self.V)
        return self._folded

    def __call__(self, x):
        # x: B, L, heads, value_dim. Fold before applying to match the
        # reference's rounding order.
        mix = self._mix()
        heads = x.shape[-2]
        if self.sharding_group is not None:
            x = mx.distributed.all_gather(
                x.transpose(2, 0, 1, 3), group=self.sharding_group
            ).transpose(1, 2, 0, 3)
            start = self.sharding_group.rank() * heads
            mix = mix[start : start + heads]
        return mix.astype(x.dtype) @ x


def _attention_weights(scores, mask, sinks):
    """Softmax over the visible keys and a per-head sink.

    The sink is one more logit per head that takes its share of the softmax
    and contributes no value. ``mask`` is boolean; ``sinks`` must broadcast
    against the scores once it is given a trailing axis.
    """
    scores = mx.where(mask, scores, -mx.inf)
    sink = mx.broadcast_to(
        sinks.astype(scores.dtype)[..., None], (*scores.shape[:-1], 1)
    )
    weights = mx.softmax(mx.concatenate((scores, sink), axis=-1), axis=-1, precise=True)
    return weights[..., :-1]


def _window_bands(kv, earlier, window, blocks):
    """Regroup local keys so each block of ``window`` queries gets its own keys.

    ``kv`` is (B, 1, earlier + L, D): the ``earlier`` keys kept from before the
    chunk, then the chunk's own. Query ``p`` of block ``i`` sees the ``window``
    keys ending at itself, which all lie in the block before it or in its own,
    so block ``i`` reads keys ``(i - 1) * window`` to ``(i + 1) * window`` and
    the mask is the same band for every block. Returns the keys as
    (B, blocks, 2 * window, D) and the mask as (blocks, window, 2 * window).
    """
    B, _, length, dim = kv.shape
    front = window - earlier
    tail = (blocks + 1) * window - front - length
    kv = mx.pad(kv[:, 0], ((0, 0), (front, tail), (0, 0)))
    kv = kv.reshape(B, blocks + 1, window, dim)
    keys = mx.concatenate((kv[:, :-1], kv[:, 1:]), axis=2)
    own = mx.arange(window)[None] <= mx.arange(window)[:, None]
    band = mx.concatenate((~own, own), axis=1)
    # Only the first block reaches back before the earliest key held.
    held = (mx.arange(blocks)[:, None] > 0) | (mx.arange(2 * window)[None] >= front)
    return keys, band[None] & held[:, None]


class HCACompressor(Compressor):
    def __call__(self, x, pool_cache, offset):
        B, _, _ = x.shape
        # The reference accumulates the compression projections and pooling
        # buffers in fp32, even for a bf16 backbone.
        pair = None
        if not self.training and x.shape[1] <= DECODE_BLOCK_SIZE:
            # One launch that reads both half-precision weights in place,
            # instead of promoting them to fp32 on every decode step.
            pair = decode_kernels.pair_project(x, self.wkv.weight, self.wgate.weight)
        if pair is not None:
            kv = pair[0].reshape(B, x.shape[1], -1)
            gate = pair[1].reshape(B, x.shape[1], -1)
        else:
            kv = x.astype(mx.float32) @ self.wkv.weight.astype(mx.float32).T
            gate = x.astype(mx.float32) @ self.wgate.weight.astype(mx.float32).T
        if pool_cache is None:
            usable = kv.shape[1] // self.compress_ratio * self.compress_ratio
            kv, gate, base = kv[:, :usable], gate[:, :usable], offset
        else:
            kv, gate, base = pool_cache.accumulate_windows(kv, gate, offset)
        if kv.shape[1] == 0:
            pooled = mx.zeros((B, 0, self.head_dim), dtype=x.dtype)
        else:
            kv = kv.reshape(B, -1, self.compress_ratio, self.head_dim)
            gate = gate.reshape(B, -1, self.compress_ratio, self.head_dim)
            weights = mx.softmax(
                gate + self.ape.astype(mx.float32), axis=-2, precise=True
            )
            pooled = self.norm((kv * weights).sum(axis=-2))
            pooled = self.rope(pooled[:, None], offset=base).squeeze(1).astype(x.dtype)
        return pool_cache.update_and_fetch(pooled) if pool_cache is not None else pooled


class HCAAttention(CompressedAttention):
    # Attend long single-row chunks in window-sized bands.
    banded_window = True

    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.compressor = HCACompressor(config, self.compress_ratio, self.head_dim)
        self.o_gate = nn.Linear(
            config.q_lora_rank, config.o_groups * config.o_lora_rank, bias=False
        )
        self.vha = VirtualHeads(self.n_heads, config.vha_rank)

    def __call__(self, x, cache=None):
        qr, q, kv = self._queries(x)
        return self._project(self._attend(x, q, kv, cache), qr)

    def rows(self, x, caches, bounds):
        """Several requests' blocks, laid end to end in ``x`` (1, N, D).

        ``bounds[r]`` is the ``(start, end)`` of request ``r``'s rows and
        ``caches[r]`` its own layer cache. The projections on either side of
        the attention are token-wise and read their weights once for all the
        rows; each request attends over its own keys.
        """
        qr, q, kv = self._queries(x)
        out = [
            self._attend(x[:, s:e], q[:, s:e], kv[:, s:e], cache)
            for cache, (s, e) in zip(caches, bounds)
        ]
        return self._project(mx.concatenate(out, axis=1), qr)

    def _queries(self, x):
        """The projections that need no position: ``(qr, q, kv)``."""
        B, L, _ = x.shape
        qr = self.q_norm(_linear(self.wq_a, x))
        q = _linear(self.wq_b, qr).reshape(B, L, self.n_heads, self.head_dim)
        q = mx.fast.rms_norm(q, None, self.config.rms_norm_eps)
        return qr, q, self.kv_norm(_linear(self.wkv, x))

    def _attend(self, x, q, kv, cache=None):
        """One request's rows over its own cache: (B, L, heads, head_dim)."""
        B, L, _ = x.shape
        local = cache[0] if cache is not None else None
        pool = cache[1] if cache is not None else None
        offset = local.offset if local is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset
        mask = create_attention_mask(
            x, local, window_size=self.config.sliding_window, return_array=True
        )
        q = self.rope(q.transpose(0, 2, 1, 3), offset)
        kv = self.rope(kv[:, None], offset)
        if local is not None:
            kv, _ = local.update_and_fetch(kv, mx.zeros((B, 1, L, 0), dtype=x.dtype))
        pooled = self.compressor(x, pool, offset)
        pool_mask = None
        if pooled.shape[1]:
            if pool is not None:
                pool_mask = pool.make_mask(L, offset)
            else:
                pool_mask = mx.arange(pooled.shape[1])[None] < (
                    mx.arange(1, L + 1)[:, None] // self.compress_ratio
                )
        if self._banded(L, kv.shape[2] - L, local, mask, pooled, pool_mask):
            out = self._banded_attention(q, kv, pooled, pool_mask)
            return self.rope(out, offset, inverse=True).transpose(0, 2, 1, 3)
        mask = _align_local_mask(mask, kv.shape[2])
        if pooled.shape[1]:
            if mask is None and pool_mask is not None:
                mask = mx.ones((B, 1, L, kv.shape[2]), dtype=mx.bool_)
            kv = mx.concatenate((kv, pooled[:, None]), axis=2)
            mask = _extend_mask(mask, pool_mask, kv.shape[2])
        out = scaled_dot_product_attention(
            q,
            kv,
            kv,
            cache=local,
            scale=self.scale,
            mask=mask,
            sinks=self.attn_sink.astype(q.dtype),
        )
        return self.rope(out, offset, inverse=True).transpose(0, 2, 1, 3)

    def _project(self, out, qr):
        """Mix the heads of ``out`` (B, L, heads, D) and project them out."""
        B, L = out.shape[:2]
        out = self.vha(out).reshape(B, L, self.o_groups, -1)
        grouped = None
        if decode_kernels.projects_block(out[..., 0, :]):
            grouped = decode_kernels.fp8_grouped(self.wo_a, out)
        if grouped is None:
            grouped = self.wo_a(out.transpose(0, 2, 1, 3))
            grouped = grouped.transpose(0, 2, 1, 3).flatten(-2)
        out = grouped
        return _parallel_output(
            self.wo_b,
            out * mx.sigmoid(_linear(self.o_gate, qr)),
            self.vha.sharding_group,
        )

    def _banded(self, L, earlier, local, mask, pooled, pool_mask):
        """Whether this call can attend block by block instead of densely.

        The dense path scores every query of a chunk against every key of the
        chunk and masks all but ``sliding_window`` of them. That is the right
        shape for a decode step and for batched caches, whose masks carry
        padding; a long chunk behind a plain windowed mask attends in bands.
        The checks are on what the caches returned, not on their classes, so a
        cache rebuilt by a server from its own store qualifies too.
        """
        window = self.config.sliding_window
        if not self.banded_window or self.training or L < 4 * window:
            return False
        if local is not None and (
            getattr(local, "keep", 0) != 0 or getattr(local, "max_size", None) != window
        ):
            return False
        # The plain mask is (L, earlier + L) with at most window - 1 keys kept
        # from before the chunk; anything else carries per-row padding.
        if not isinstance(mask, mx.array) or mask.shape != (L, earlier + L):
            return False
        if not 0 <= earlier < window:
            return False
        count = pooled.shape[1]
        return pool_mask is None or pool_mask.shape == (L, count)

    def _banded_attention(self, q, kv, pooled, pool_mask):
        B, H, L, D = q.shape
        window = self.config.sliding_window
        blocks = -(-L // window)
        padded = blocks * window
        keys, mask = _window_bands(kv, kv.shape[2] - L, window, blocks)
        count = pooled.shape[1]
        if count:
            if pool_mask is None:
                pool_mask = mx.ones((L, count), dtype=mx.bool_)
            pool_mask = mx.pad(pool_mask, ((0, padded - L), (0, 0)))
            mask = mx.concatenate(
                (mask, pool_mask.reshape(blocks, window, count)), axis=-1
            )
            keys = mx.concatenate(
                (keys, mx.broadcast_to(pooled[:, None], (B, blocks, count, D))),
                axis=2,
            )
        q = mx.pad(q, ((0, 0), (0, 0), (0, padded - L), (0, 0)))
        keys = keys[:, None]
        q = q.reshape(B, H, blocks, window, D)
        scores = (q * self.scale) @ keys.swapaxes(-1, -2)
        weights = _attention_weights(scores, mask, self.attn_sink[:, None, None])
        out = weights.astype(keys.dtype) @ keys
        return out.reshape(B, H, padded, D)[:, :, :L]


def _make_index_kernel():
    """Weighted head sum of the indexer's scores, in one pass.

    ``scores`` holds every head's query-key product as (rows, heads, keys) in
    the model's precision. The plain form widens all of it to fp32, weights it
    and sums over heads, three passes over a tensor of 2 GB per 8K keys for a
    2048-row chunk; here each output reads its 64 values once.
    """
    if mx.default_device() != mx.gpu or not mx.metal.is_available():
        return None

    source = """
        ulong row = thread_position_in_grid.y;
        ulong key = thread_position_in_grid.x;
        ulong H = scores_shape[1];
        ulong S = scores_shape[2];
        const device float* w = weights + row * H;
        float acc = 0.0f;
        for (ulong h = 0; h < H; ++h) {
            acc += w[h] * metal::max((float)scores[(row * H + h) * S + key], 0.0f);
        }
        out[row * S + key] = acc;
    """
    return mx.fast.metal_kernel(
        name="ernie_index_scores",
        input_names=["scores", "weights"],
        output_names=["out"],
        source=source,
        ensure_row_contiguous=True,
    )


_index_kernel = _make_index_kernel()
# Rows scored per kernel launch; bounds the per-head score tensor.
_INDEX_ROWS = 512


class Indexer(nn.Module):
    # Sum the heads' scores in one pass. The composed form makes several
    # passes over heads x keys values, which for a decode step at 30K tokens
    # of context costs more than the rest of the layer.
    kernel_scores = True
    # Per-head scores (rows x heads x keys) one ranking pass may hold. A chunk
    # that needs more is ranked a block of rows at a time, each block finished
    # before the next, so the working memory stops growing with the context.
    score_elements = 1 << 31

    def __init__(self, config):
        super().__init__()
        self.heads = config.index_n_heads
        self.dim = config.index_head_dim
        self.topk = config.index_topk
        self.window = config.sliding_window
        self.wq_b = nn.Linear(
            config.hybrid_mla_q_lora_rank, self.heads * self.dim, bias=False
        )
        self.wk = nn.Linear(config.hidden_size, self.dim, bias=False)
        self.k_norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.weights_proj = nn.Linear(config.hidden_size, self.heads, bias=False)
        self.rope = nn.RoPE(
            config.hybrid_mla_qk_rope_head_dim,
            traditional=False,
            base=config.rope_theta,
        )

    def _scores(self, q, k, weights):
        """Each query's score for every key: (B, 1, L, S) in fp32.

        ``q`` is (B, heads, L, dim), ``k`` (B, 1, S, dim) and ``weights``
        (B, L, heads) in fp32.
        """
        B, _, L, _ = q.shape
        S = k.shape[2]
        if _index_kernel is None or self.training or not self.kernel_scores:
            scores = mx.maximum(q @ k.swapaxes(-1, -2), 0).astype(mx.float32)
            return (scores * weights.transpose(0, 2, 1)[..., None]).sum(
                axis=1, keepdims=True
            )
        q = q.transpose(0, 2, 1, 3)
        keys = k.swapaxes(-1, -2)
        parts = []
        for start in range(0, L, _INDEX_ROWS):
            rows = min(_INDEX_ROWS, L - start)
            scores = q[:, start : start + rows] @ keys
            parts.append(
                _index_kernel(
                    inputs=[
                        scores.reshape(B * rows, self.heads, S),
                        weights[:, start : start + rows].reshape(-1, self.heads),
                    ],
                    grid=(S, B * rows, 1),
                    threadgroup=(32, 1, 1),
                    output_shapes=[(B * rows, S)],
                    output_dtypes=[mx.float32],
                )[0].reshape(B, 1, rows, S)
            )
        return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=2)

    def _ranked(self, q, k, weights, mask, start, rows, L, S):
        """Ranked key ids for ``rows`` queries from ``start``, and which are real.

        Returns ``(ids, ranked)``, both (rows, topk). Only keys older than a
        query's recent window are ranked; the causal diagonal of query ``j``
        is physical column ``S - L + j``.
        """
        queries = mx.arange(S - L + start, S - L + start + rows)
        valid = mx.arange(S)[None, :] - queries[:, None] <= -self.window
        if mask is not None:
            valid = valid & mask[..., start : start + rows, :]
        scores = self._scores(
            q[:, :, start : start + rows], k, weights[:, start : start + rows]
        )
        scores = mx.where(valid, scores, -mx.inf)
        ids = mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., : self.topk]
        ranked = mx.take_along_axis(scores, ids, axis=-1) > -mx.inf
        return ids[0, 0], ranked[0, 0]

    def __call__(self, x, qr, mask, cache=None, gather=False):
        """Pick the keys each query attends to.

        Returns ``(mask, indices)``. Without ``gather`` the mask is the dense
        (B, 1, L, S) selection and ``indices`` names the selected keys only for
        a single query. With ``gather`` (single-row chunks) there is no dense
        mask: ``indices`` is ``(ids, usable)``, both (L, topk + window), every
        query's ranked keys followed by its recent window, and which of them
        it may actually attend to.
        """
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset
        q = _linear(self.wq_b, qr).reshape(B, L, self.heads, self.dim)
        q = q.transpose(0, 2, 1, 3)
        k = self.k_norm(self.wk(x))[:, None]
        q, k = self.rope(q, offset), self.rope(k, offset)
        # The release's orthonormal Hadamard leaves full-precision dot products
        # unchanged; it is only needed when quantizing the indexer cache.
        if cache is not None:
            k, _ = cache.update_and_fetch(k, mx.zeros((B, 1, L, 0), dtype=k.dtype))
        S = k.shape[2]
        if S <= self.topk + self.window:
            return mask, None
        weights = (
            self.weights_proj(x).astype(mx.float32) * (self.heads * self.dim) ** -0.5
        )
        if gather:
            block = max(DECODE_BLOCK_SIZE, self.score_elements // (self.heads * S))
            if block >= L:
                ids, ranked = self._ranked(q, k, weights, mask, 0, L, L, S)
            else:
                parts = []
                for start in range(0, L, block):
                    rows = min(block, L - start)
                    part = self._ranked(q, k, weights, mask, start, rows, L, S)
                    mx.eval(part)
                    parts.append(part)
                ids = mx.concatenate([part[0] for part in parts], axis=0)
                ranked = mx.concatenate([part[1] for part in parts], axis=0)
            # A query early in the context has fewer than topk older keys; the
            # rest of its ranked slots hold keys it must not see, and its
            # window may start before the first key.
            recent = (
                mx.arange(S - L, S)[:, None] - mx.arange(self.window - 1, -1, -1)[None]
            )
            inside = recent >= 0
            recent = mx.maximum(recent, 0)
            if mask is not None:
                inside = inside & mx.take_along_axis(
                    mx.broadcast_to(mask, (B, 1, L, S))[0, 0], recent, axis=-1
                )
            return None, (
                mx.concatenate((ids, recent), axis=-1),
                mx.concatenate((ranked, inside), axis=-1),
            )
        # Physical cache columns include batch left padding; the causal diagonal
        # is still S-L+j. Always include the recent window, rank only older keys.
        delta = mx.arange(S)[None, :] - mx.arange(S - L, S)[:, None]
        older = delta <= -self.window
        valid = older if mask is None else older & mask
        scores = mx.where(valid, self._scores(q, k, weights), -mx.inf)
        ids = mx.argpartition(-scores, kth=self.topk - 1, axis=-1)[..., : self.topk]
        recent = (delta <= 0) & (delta > -self.window)
        sparse = mx.put_along_axis(
            mx.zeros(scores.shape, dtype=mx.bool_), ids, mx.array(True), axis=-1
        )
        sparse = (sparse & older) | recent
        sparse = sparse if mask is None else sparse & mask
        if L == 1:
            recent_ids = mx.broadcast_to(
                mx.arange(S - self.window, S), (B, 1, 1, self.window)
            )
            return sparse, mx.concatenate((ids, recent_ids), axis=-1)
        return sparse, None


class MLAAttention(nn.Module):
    # Attend long single-row chunks over each query's selected keys only.
    gathered_keys = True
    # Context length from which gathering beats the dense product, for
    # chunks and for verify blocks.
    gathered_min_keys = 6144
    gathered_min_keys_short = 4096
    gathered_rows = 512

    def __init__(self, config):
        super().__init__()
        self.heads = config.hybrid_mla_num_attention_heads
        self.nope = config.hybrid_mla_qk_nope_head_dim
        self.rope_dim = config.hybrid_mla_qk_rope_head_dim
        self.latent = config.hybrid_mla_kv_lora_rank
        self.value_dim = config.hybrid_mla_v_head_dim
        qr = config.hybrid_mla_q_lora_rank
        self.scale = (self.nope + self.rope_dim) ** -0.5
        self.q_a_proj = nn.Linear(config.hidden_size, qr, bias=False)
        self.q_a_layernorm = nn.RMSNorm(qr, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(
            qr, self.heads * (self.nope + self.rope_dim), bias=False
        )
        self.kv_a_proj_with_mqa = nn.Linear(
            config.hidden_size, self.latent + self.rope_dim, bias=False
        )
        self.kv_a_layernorm = nn.RMSNorm(self.latent, eps=config.rms_norm_eps)
        self.embed_q = MultiLinear(self.nope, self.latent, self.heads)
        self.unembed_out = MultiLinear(self.latent, self.value_dim, self.heads)
        self.wo_b = nn.Linear(
            self.heads * self.value_dim, config.hidden_size, bias=False
        )
        self.o_gate = nn.Linear(qr, self.heads * self.value_dim, bias=False)
        self.vha = VirtualHeads(self.heads, config.vha_rank)
        self.indexer = Indexer(config)
        self.attn_sink = mx.zeros((self.heads,), dtype=mx.float32)
        self.rope = nn.RoPE(self.rope_dim, traditional=True, base=config.rope_theta)

    def __call__(self, x, cache=None):
        qr, q, kv, k_pe = self._queries(x)
        return self._finish(self._attend(x, qr, q, kv, k_pe, cache), qr)

    def rows(self, x, caches, bounds):
        """Several requests' blocks in ``x`` (1, N, D); see ``HCAAttention.rows``."""
        qr, q, kv, k_pe = self._queries(x)
        out = [
            self._attend(
                x[:, s:e], qr[:, s:e], q[:, s:e], kv[:, s:e], k_pe[:, s:e], cache
            )
            for cache, (s, e) in zip(caches, bounds)
        ]
        return self._finish(mx.concatenate(out, axis=1), qr)

    def _queries(self, x):
        """The projections that need no position: ``(qr, q, kv, k_pe)``."""
        B, L, _ = x.shape
        qr = self.q_a_layernorm(_linear(self.q_a_proj, x))
        q = _linear(self.q_b_proj, qr).reshape(B, L, self.heads, -1)
        kv, k_pe = mx.split(_linear(self.kv_a_proj_with_mqa, x), [self.latent], axis=-1)
        return qr, q, self.kv_a_layernorm(kv), k_pe

    def _finish(self, out, qr):
        """Mix the heads of ``out`` (B, L, heads, D) and project them out."""
        out = self.vha(out).flatten(-2)
        return _parallel_output(
            self.wo_b,
            out * mx.sigmoid(_linear(self.o_gate, qr)),
            self.vha.sharding_group,
        )

    def _attend(self, x, qr, q, kv, k_pe, cache=None):
        """One request's rows over its own cache: (B, L, heads, value_dim)."""
        B, L, _ = x.shape
        latent_cache = cache[0] if cache is not None else None
        index_cache = cache[1] if cache is not None else None
        offset = latent_cache.offset if latent_cache is not None else 0
        offset = mx.array(offset) if isinstance(offset, mx.array) else offset
        mask = create_attention_mask(x, latent_cache, return_array=True)
        q = q.transpose(0, 2, 1, 3)
        q_nope, q_pe = mx.split(q, [self.nope], axis=-1)
        kv = kv[:, None]
        q_pe = self.rope(q_pe, offset)
        k_pe = self.rope(k_pe[:, None], offset)
        if latent_cache is not None:
            kv, k_pe = latent_cache.update_and_fetch(kv, k_pe)
        gathered = self._gathered(B, L, kv.shape[2], latent_cache)
        mask, indices = self.indexer(x, qr, mask, index_cache, gather=gathered)
        if gathered:
            out = self._gathered_attention(q_nope, q_pe, kv, k_pe, *indices)
            return out.transpose(0, 2, 1, 3)
        if indices is not None:
            take = indices[:, :, 0, :, None]
            kv = mx.take_along_axis(kv, take, axis=2)
            k_pe = mx.take_along_axis(k_pe, take, axis=2)
            mask = mx.take_along_axis(mask, indices, axis=-1)
        pe_scores = (q_pe * self.scale) @ k_pe.swapaxes(-1, -2)
        if mask is not None:
            pe_scores = mx.where(mask, pe_scores, -mx.inf)
        # Folding the per-head projections into the queries costs L rows; the
        # other form projects the whole cached context on every call. Short
        # decode and verify blocks fold, prefill chunks project.
        folded = L == 1 or (not self.training and L <= DECODE_BLOCK_SIZE)
        if folded:
            q_nope = self.embed_q(q_nope)
            k = v = kv
        else:
            k = self.embed_q(kv, transpose=False)
            v = self.unembed_out(kv)
        out = scaled_dot_product_attention(
            q_nope,
            k,
            v,
            cache=latent_cache,
            scale=self.scale,
            mask=pe_scores,
            sinks=self.attn_sink.astype(q_nope.dtype),
        )
        if folded:
            out = self.unembed_out(out)
        return out.transpose(0, 2, 1, 3)

    def _gathered(self, B, L, S, latent_cache):
        """Whether this call attends over gathered keys instead of all of them.

        Each query keeps ``index_topk`` ranked keys and its recent window, yet
        the dense path scores it against the whole context and projects that
        context per head on every call, so a chunk costs more the longer the
        prompt already is. Gathering every query's own keys costs the same at
        any length; it pays once the context is a few times what a query keeps.
        """
        if not self.gathered_keys or self.training or B != 1:
            return False
        # One query takes its keys through the indexer's own list; a verify
        # block gathers like a chunk does, from a shorter context on.
        short = L <= DECODE_BLOCK_SIZE
        if L == 1 or S < (
            self.gathered_min_keys_short if short else self.gathered_min_keys
        ):
            return False
        if S <= self.indexer.topk + self.indexer.window:
            return False
        # By name, so a server's own plain cache of the same kind qualifies;
        # quantized and batched caches keep the dense path.
        return latent_cache is None or type(latent_cache).__name__ == "KVCache"

    def _gathered_attention(self, q_nope, q_pe, kv, k_pe, ids, usable):
        """Attention for a single-row chunk over each query's own keys.

        The per-head projections are folded into the queries, as for a decode
        step, so keys and values are both the gathered latents. Rows go in
        blocks to bound the gathered tensor (rows x keys x latent).
        """
        L = ids.shape[0]
        q = (self.embed_q(q_nope)[0] * self.scale).transpose(1, 0, 2)
        q_pe = (q_pe[0] * self.scale).transpose(1, 0, 2)
        kv, k_pe = kv[0, 0], k_pe[0, 0]
        parts = []
        for start in range(0, L, self.gathered_rows):
            rows = slice(start, start + self.gathered_rows)
            keys = mx.take(kv, ids[rows], axis=0)
            scores = q[rows] @ keys.swapaxes(-1, -2) + q_pe[rows] @ mx.take(
                k_pe, ids[rows], axis=0
            ).swapaxes(-1, -2)
            weights = _attention_weights(scores, usable[rows, None], self.attn_sink)
            parts.append(weights.astype(keys.dtype) @ keys)
        out = parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
        return self.unembed_out(out.transpose(1, 0, 2)[None])


# Decode and verify blocks expand through one compiled graph.
_hc_expand_decode = mx.compile(hc_expand)


def _hc_boundary(connection, norm, x, pending):
    """Expand ``pending`` into ``x``, then collapse and normalize for a sublayer.

    Returns ``(stream, normed, post, comb)``. Two launches when the kernels
    cover the block, the composed operations otherwise.
    """
    weight = norm.weight if type(norm) is nn.RMSNorm else None
    fused = (
        weight is not None and connection.mult == 4 and mx.default_device() == mx.gpu
    )
    mixes = None
    if fused and pending is not None:
        out = decode_kernels.hc_expand_project(
            pending[0], x, pending[1], pending[2], connection.fn, connection.norm_eps
        )
        if out is not None:
            x, mixes = out
            pending = None
    if pending is not None:
        x = _hc_expand_decode(pending[0], x, pending[1], pending[2])
    if fused and mixes is None:
        mixes = decode_kernels.hc_project(x, connection.fn, connection.norm_eps)
    out = None
    if mixes is not None:
        out = decode_kernels.hc_collapse_norm(
            x,
            mixes,
            connection.scale,
            connection.base,
            weight,
            connection.iters,
            connection.eps,
            norm.eps,
        )
    if out is None:
        h, post, comb = connection(x)
        return x, norm(h), post, comb
    return x, out[0], out[1], out[2]


class DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = (
            MLAAttention(config)
            if config.compress_ratios[layer_idx] == -2
            else HCAAttention(config, layer_idx)
        )
        self.ffn = (
            MLP(config, config.intermediate_size)
            if layer_idx < config.first_k_dense_replace
            else LatentMoE(config)
        )
        self.attn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = HyperConnection(config)
        self.ffn_hc = HyperConnection(config)

    def __call__(self, x, cache=None, router_bias=None):
        short = not self.training and x.shape[1] <= DECODE_BLOCK_SIZE
        expand = _hc_expand_decode if short else hc_expand
        h, post, comb = self.attn_hc(x)
        h = self.attn_norm(h)
        h = self.attn(h, cache)
        x = expand(h, x, post, comb)
        h, post, comb = self.ffn_hc(x)
        h = self.ffn_norm(h)
        h = self.ffn(h) if router_bias is None else self.ffn(h, router_bias)
        return expand(h, x, post, comb)

    def short(self, x, cache=None, pending=None, bounds=None):
        """A decode or verify block, with the last expand left to the caller.

        ``pending`` is the ``(h, post, comb)`` of the sublayer before this
        layer, not yet expanded into ``x``. Returns the stream this layer's
        feed-forward read and its own pending expand, so that every expand
        runs in the same launch as the projection that follows it.

        With ``bounds`` the rows are several requests' blocks and ``cache``
        holds one layer cache per request.
        """
        x, h, post, comb = _hc_boundary(self.attn_hc, self.attn_norm, x, pending)
        h = self.attn(h, cache) if bounds is None else self.attn.rows(h, cache, bounds)
        x, h, post, comb = _hc_boundary(self.ffn_hc, self.ffn_norm, x, (h, post, comb))
        return x, (self.ffn(h), post, comb)


class ErnieModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.hc_head = HyperHead(config)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs,
        cache=None,
        inputs_embeds=None,
        hidden_sink=None,
        skip_final_norm=False,
        capture_layer_ids=None,
        capture_sink=None,
    ):
        h = self.embed_tokens(inputs) if inputs_embeds is None else inputs_embeds
        h = mx.broadcast_to(
            h[:, :, None, :], (*h.shape[:2], self.config.hc_mult, h.shape[-1])
        )
        if cache is None:
            cache = [None] * len(self.layers)
        if len(cache) != len(self.layers):
            raise ValueError("Expected one cache per decoder layer")
        capture = set(capture_layer_ids) if capture_layer_ids else ()
        # Decode and verify blocks carry each layer's last expand into the
        # next layer's first projection.
        short = not self.training and h.shape[1] <= DECODE_BLOCK_SIZE
        last = len(self.layers) - 1
        pending = None
        for index, (layer, layer_cache) in enumerate(zip(self.layers, cache)):
            if short:
                h, pending = layer.short(h, layer_cache, pending)
                if index in capture or index == last:
                    h = _hc_expand_decode(pending[0], h, pending[1], pending[2])
                    pending = None
                _submit(index, (layer_cache,), h, *(pending or ()))
            else:
                # A prefill chunk does not gain from it (1,575 against 1,637 tok/s).
                h = layer(h, layer_cache)
            if index in capture:
                # DSpark reads each target layer's output averaged over the
                # hyper-connection streams.
                capture_sink.append(h.mean(axis=2))
        if hidden_sink is not None:
            hidden_sink.append(h.flatten(-2))
        if skip_final_norm:
            return h
        return self.norm(self.hc_head(h))

    def rows(self, inputs, caches, bounds, capture=(), capture_sink=None):
        """Several requests' decode or verify blocks in one pass.

        ``inputs`` (1, N) holds the requests' tokens one after another, at
        most ``DECODE_BLOCK_SIZE`` in all; ``bounds[r]`` is where request
        ``r``'s lie and ``caches[r]`` is its own cache. Everything but the
        attention is token-wise, so it runs once over all N rows as it would
        for one request's block of that size. Returns the streams
        (1, N, hc_mult, D) before the final norm.
        """
        h = self.embed_tokens(inputs)
        h = mx.broadcast_to(
            h[:, :, None, :], (*h.shape[:2], self.config.hc_mult, h.shape[-1])
        )
        last = len(self.layers) - 1
        pending = None
        for index, layer in enumerate(self.layers):
            layer_caches = [cache[index] for cache in caches]
            h, pending = layer.short(h, layer_caches, pending, bounds)
            if index in capture or index == last:
                h = _hc_expand_decode(pending[0], h, pending[1], pending[2])
                pending = None
            _submit(index, layer_caches, h, *(pending or ()))
            if index in capture:
                capture_sink.append(h.mean(axis=2))
        return h


class MultiMax(nn.Module):
    def __init__(self):
        super().__init__()
        self.multimax_ranges = mx.zeros((4,), dtype=mx.float32)
        self.multimax_ts = mx.zeros((4,), dtype=mx.float32)
        self._packed_source = None
        self._packed = None

    def packed(self):
        """Ranges then factors as one fp32 array, for the head kernel."""
        source = (self.multimax_ranges, self.multimax_ts)
        cached = self._packed_source
        if cached is None or cached[0] is not source[0] or cached[1] is not source[1]:
            self._packed = mx.concatenate(source).astype(mx.float32)
            mx.eval(self._packed)
            self._packed_source = source
        return self._packed

    def __call__(self, logits):
        logits = logits.astype(mx.float32)
        r, t = self.multimax_ranges, self.multimax_ts
        return (
            logits
            + t[0] * mx.maximum(r[0] - logits, 0)
            + t[1] * mx.maximum(logits - r[1], 0)
            + t[2] * mx.maximum(r[2] - logits, 0) ** 2
            + t[3] * mx.maximum(logits - r[3], 0) ** 2
        )


class LanguageModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_type = config.model_type
        self.model = ErnieModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.multimax = MultiMax()
        self.sharding_group = None

    def __call__(self, inputs=None, cache=None, inputs_embeds=None, **kwargs):
        if kwargs.pop("speculative_verify", False):
            from ...speculative.common import verify_forward

            output, transaction = verify_forward(self, inputs, cache, **kwargs)
            output.gdn_states = transaction
            return output
        hidden_sink = [] if kwargs.get("return_hidden", False) else None
        capture_layer_ids = kwargs.get("capture_layer_ids")
        capture_sink = [] if capture_layer_ids else None
        skip_logits = kwargs.get("skip_logits", False)
        h = self.model(
            inputs,
            cache,
            inputs_embeds,
            hidden_sink=hidden_sink,
            skip_final_norm=skip_logits,
            capture_layer_ids=capture_layer_ids,
            capture_sink=capture_sink,
        )
        # The upstream model computes the vocabulary projection and MultiMax
        # in fp32, independently of the backbone dtype.
        logits = (
            None
            if skip_logits
            else project_logits(h, self._project_logits, kwargs.get("logits_to_keep"))
        )
        return LanguageModelOutput(
            logits=logits,
            hidden_states=capture_sink if capture_sink is not None else hidden_sink,
            shared_kv_states={} if kwargs.get("return_shared_kv", False) else None,
        )

    def project_head(self, hidden, multimax=False):
        """Vocabulary logits in fp32, before MultiMax unless ``multimax``.

        The released weight is half precision and the projection is fp32. A
        plain matmul promotes all vocab_size x hidden_size values on every
        call; the kernel reads the weight in place, accumulates in fp32 and
        can apply MultiMax as it writes.
        """
        hidden = hidden.astype(mx.float32)
        head = self.lm_head
        weight = head.get("weight") if type(head) is nn.Linear else None
        logits = None
        if not self.training and weight is not None and hidden.ndim >= 2:
            rows = hidden.reshape(-1, hidden.shape[-1])
            shape = self.multimax.packed() if multimax else None
            logits = decode_kernels.vocab_head(rows, weight, shape)
        if logits is None:
            logits = head(hidden)
            return self.multimax(logits) if multimax else logits
        return logits.reshape(*hidden.shape[:-1], weight.shape[0])

    def _project_logits(self, hidden):
        return self.project_head(hidden, multimax=True)

    def speculative_logits_from_hidden(self, hidden):
        h = hidden.reshape(
            *hidden.shape[:2], self.config.hc_mult, self.config.hidden_size
        )
        h = self.model.norm(self.model.hc_head(h))
        return self._project_logits(h)

    def speculative_verify_hidden(self, inputs, cache):
        transaction = start_speculative_cache(cache, inputs.shape[1])
        try:
            h = self.model(inputs, cache=cache, skip_final_norm=True)
        except BaseException:
            transaction.abort()
            raise
        return h.flatten(-2), {}, transaction

    def speculative_verify_logits(self, inputs, cache, sampler):
        hidden, shared, rollback = self.speculative_verify_hidden(inputs, cache)
        return (
            hidden,
            shared,
            rollback,
            sampler(self.speculative_logits_from_hidden(hidden)),
        )

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        return rollback_speculative_cache(caches, gdn_states, accepted, block_size)

    # Rows one ``speculative_verify_rows`` forward can hold.
    verify_rows_limit = DECODE_BLOCK_SIZE

    def speculative_verify_rows(self, inputs, caches, capture_layer_ids=None):
        """Verify several requests' blocks in one forward.

        ``inputs[r]`` is request ``r``'s block (1, R_r) and ``caches[r]`` its
        own cache; the blocks may differ in length. A forward mostly costs the
        weights it reads, and one forward over all the rows reads them once.
        No request sees another's keys, so nothing is padded and each cache
        commits or rolls back on its own.

        Returns, per request, ``(logits, captured, transaction)``: the logits
        (1, R_r, vocab), the outputs of ``capture_layer_ids`` as a list of
        (1, R_r, D), and the open cache transaction to commit.
        """
        lengths = [int(block.shape[1]) for block in inputs]
        if not lengths or min(lengths) < 1 or sum(lengths) > self.verify_rows_limit:
            raise ValueError(
                f"Expected 1 to {self.verify_rows_limit} rows in all, got {lengths}"
            )
        bounds, start = [], 0
        for length in lengths:
            bounds.append((start, start + length))
            start += length
        transactions = []
        try:
            for cache, length in zip(caches, lengths):
                transactions.append(start_speculative_cache(cache, length))
            sink = [] if capture_layer_ids else None
            h = self.model.rows(
                mx.concatenate(inputs, axis=1),
                caches,
                bounds,
                set(capture_layer_ids or ()),
                sink,
            )
            logits = self._project_logits(self.model.norm(self.model.hc_head(h)))
        except BaseException:
            for transaction in transactions:
                transaction.abort()
            raise
        return [
            (logits[:, s:e], [part[:, s:e] for part in sink or ()], transaction)
            for (s, e), transaction in zip(bounds, transactions)
        ]

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        return [
            (
                CacheList(KVCache(), KVCache())
                if ratio == -2
                else CacheList(
                    RotatingKVCache(max_size=self.config.sliding_window),
                    PoolingCache(ratio),
                )
            )
            for ratio in self.config.compress_ratios
        ]

    def shard(self, group=None):
        """Partition attention heads and FFN intermediate dimensions."""
        group = group or mx.distributed.init()
        size = group.size()
        if size == 1:
            return
        if self.sharding_group is not None:
            raise ValueError("The language model is already sharded")

        partitions = []

        def partition(module, axis):
            for name, value in tree_flatten(module.parameters()):
                # Splitting packed columns and their scales together requires
                # each rank to own complete quantization groups.
                if value.shape[axis] % size:
                    raise ValueError(
                        f"Cannot shard {type(module).__name__}.{name} "
                        f"with shape {value.shape} on axis {axis} across {size} ranks"
                    )
            partitions.append((module, axis))

        def partition_mlp(mlp):
            partition(mlp.gate_proj, -2)
            partition(mlp.up_proj, -2)
            partition(mlp.down_proj, -1)

        # Validate every dimension before mutating any weights or head counts.
        for layer in self.layers:
            attn = layer.attn
            if isinstance(attn, HCAAttention):
                if attn.o_groups % size or attn.n_heads % size:
                    raise ValueError(
                        "Rank count must divide HCA heads and output groups"
                    )
                partition(attn.wq_b, -2)
                partition(attn.wo_a, 0)
            else:
                if attn.heads % size:
                    raise ValueError("Rank count must divide MLA heads")
                partition(attn.q_b_proj, -2)
                partition(attn.embed_q, 0)
                partition(attn.unembed_out, 0)
            partition(attn.o_gate, -2)
            partition(attn.wo_b, -1)
            ffn = layer.ffn
            if isinstance(ffn, LatentMoE):
                partition_mlp(ffn.switch_mlp)
                partition_mlp(ffn.shared_experts)
            else:
                partition_mlp(ffn)

        for module, axis in partitions:
            shard_inplace(module, lambda path, value, axis=axis: axis, group=group)
        for layer in self.layers:
            attn = layer.attn
            attn.vha.sharding_group = group
            attn.attn_sink = mx.split(attn.attn_sink, size)[group.rank()]
            if isinstance(attn, HCAAttention):
                attn.n_heads //= size
                attn.o_groups //= size
            else:
                attn.heads //= size
            layer.ffn.sharding_group = group
            if isinstance(layer.ffn, LatentMoE):
                layer.ffn.shared_experts.sharding_group = group
                # Keep down-projection partial sums in FP32 until all-reduce,
                # while retaining the activation's original BF16 rounding.
                layer.ffn.switch_mlp.activation.output_dtype = mx.float32
        self.sharding_group = group

    def sanitize(self, weights):
        weights = {k.removeprefix("language_model."): v for k, v in weights.items()}
        out = {}
        for key, value in weights.items():
            match = re.search(r"model\.layers\.(\d+)\.", key)
            if match and int(match[1]) >= self.config.num_hidden_layers:
                continue  # ECHO is not part of the next-token backbone.
            if key.endswith("_scale_inv"):
                if key.removesuffix("_scale_inv") not in weights:
                    raise ValueError(f"FP8 scale without its weight: {key}")
                continue
            scale_key = key + "_scale_inv"
            if scale_key in weights:
                value = _dequantize_fp8_weight(value, weights[scale_key]).astype(
                    mx.bfloat16
                )
            elif value.dtype == mx.uint8 and not key.endswith(".scales"):
                raise ValueError(f"FP8 weight without its scale: {key}")

            if key == "model.mhc_contract.hc_head_fn":
                value = _row_major(value.T)
            key = key.replace("model.lm_head.", "multimax.")
            key = key.replace("model.mhc_contract.hc_head_", "model.hc_head.")
            for src, dst in (("self_attention", "attn"), ("mlp", "ffn")):
                prefix = f".{src}_hyper_connection."
                if prefix in key:
                    key = key.replace(prefix, f".{dst}_hc.")
                    if key.endswith("mapping_proj.weight"):
                        key = key.replace("mapping_proj.weight", "fn")
                        value = _row_major(value.T)
                    key = key.replace(f".{dst}_hc.bias", f".{dst}_hc.base")
            remap = {
                ".input_layernorm.": ".attn_norm.",
                ".post_attention_layernorm.": ".ffn_norm.",
                ".block_sparse_moe.": ".ffn.",
                ".mlp.": ".ffn.",
                ".self_attn.": ".attn.",
                ".core_attention.": ".",
                ".linear_q_down_proj.": ".wq_a.",
                ".linear_q_up_proj.": ".wq_b.",
                ".linear_kv_proj.": ".wkv.",
                ".q_layernorm.": ".q_norm.",
                ".kv_layernorm.": ".kv_norm.",
                ".compressor.linear_wkv.": ".compressor.wkv.",
                ".compressor.linear_wgate.": ".compressor.wgate.",
                ".attn.gate_proj.": ".attn.o_gate.",
                ".attn.o_proj.": ".attn.wo_b.",
                ".attn.vha_postmix_U": ".attn.vha.U",
                ".attn.vha_postmix_V": ".attn.vha.V",
                ".attn.softmax_offset": ".attn.attn_sink",
            }
            for src, dst in remap.items():
                key = key.replace(src, dst)
            if ".attn.linear_o_group_proj" in key:
                if key.endswith(".linear_o_group_proj"):
                    key += ".weight"
                key = key.replace(".linear_o_group_proj.", ".wo_a.")
                value = value.reshape(self.config.o_groups, self.config.o_lora_rank, -1)
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                key = key.replace(f".shared_experts.{src}.", f".shared_experts.{dst}.")
            out[key] = value

        for i in range(self.config.num_hidden_layers):
            prefix = f"model.layers.{i}"
            for kind in ("attn", "ffn"):
                keys = [
                    f"{prefix}.{kind}_hc.alpha_{part}"
                    for part in ("pre", "post", "res")
                ]
                if keys[0] in out:
                    out[f"{prefix}.{kind}_hc.scale"] = mx.concatenate(
                        [out.pop(k).reshape(-1) for k in keys]
                    )
            for src, dst in (
                ("w1", "gate_proj"),
                ("w2", "down_proj"),
                ("w3", "up_proj"),
            ):
                for suffix in ("weight", "scales", "biases"):
                    ep = f"{prefix}.ffn.experts"
                    if f"{ep}.0.{src}.{suffix}" in out:
                        out[f"{prefix}.ffn.switch_mlp.{dst}.{suffix}"] = mx.stack(
                            [
                                out.pop(f"{ep}.{e}.{src}.{suffix}")
                                for e in range(self.config.n_routed_experts)
                            ]
                        )
        weights = out
        for i, ratio in enumerate(self.config.compress_ratios):
            key = f"model.layers.{i}.attn.kv_b_proj.weight"
            if ratio == -2 and key in weights:
                attn = self.layers[i].attn
                w = weights.pop(key)
                scale_key = key.removesuffix("weight") + "scales"
                if scale_key in weights:
                    # The shared loader converts released FP8 linears to MXFP8.
                    # Restore this fused projection before transposing its K part,
                    # following the MLA layout conversion used by DeepSeek.
                    scales = weights.pop(scale_key)
                    biases = weights.pop(key.removesuffix("weight") + "biases", None)
                    w = mx.dequantize(
                        w,
                        scales,
                        biases,
                        group_size=attn.latent // scales.shape[-1],
                        bits=w.shape[-1] * 32 // attn.latent,
                        mode="mxfp8" if scales.dtype == mx.uint8 else "affine",
                    )
                w = w.reshape(attn.heads, attn.nope + attn.value_dim, attn.latent)
                k, v = mx.split(w, [attn.nope], axis=1)
                prefix = f"model.layers.{i}.attn."
                weights[prefix + "embed_q.weight"] = _row_major(k.swapaxes(-1, -2))
                weights[prefix + "unembed_out.weight"] = v
        return weights

    @property
    def quant_predicate(self):
        def predicate(path, module, *, quantization=None):
            if ".switch_mlp." in path:
                requested = quantization or {
                    "group_size": 64,
                    "bits": 4,
                    "mode": "affine",
                }
                if requested["bits"] == 8:
                    return dict(MLX_MXFP8_QUANTIZATION)
                return dict(requested)
            if (
                ".shared_experts." in path
                or path.endswith(("fc1_latent_proj", "fc2_latent_proj"))
                or path in ("echo.e_proj", "echo.h_proj")
            ):
                return dict(MLX_MXFP8_QUANTIZATION)
            return False

        return predicate

    @property
    def cast_predicate(self):
        return lambda path: not any(
            part in path
            for part in (
                "_hc.",
                "hc_head.",
                ".gate.",
                "e_score_correction_bias",
                "attn_sink",
                ".compressor.ape",
                "multimax.",
                "lm_head.",
            )
        )
