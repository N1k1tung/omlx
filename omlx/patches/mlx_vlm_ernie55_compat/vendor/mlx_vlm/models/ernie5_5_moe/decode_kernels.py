"""Fused Metal kernels for ERNIE 5.5 decode steps and short verify blocks.

A one-token forward streams about 270 MB of weights per layer through some
seventeen projections, and around them runs some fifty-five small operations:
casts, norms, the router, the expert combine, the hyper-connection algebra.
Each of those is its own GPU launch of a few microseconds, and together they
cost about as much as the weights do. The kernels here do the same arithmetic
in a handful of launches per layer.

Every function returns ``None`` when it does not cover its arguments; callers
then run the composed operations. Rounding follows the composed path where it
materializes a half-precision value; sums that the composed path accumulates in
fp32 are accumulated in fp32 here too, in a different order.
"""

import struct
from functools import lru_cache

import mlx.core as mx

MAX_ROWS = 8
# Set False to run every decode block through the composed operations.
ENABLED = True

_HALF = (mx.bfloat16, mx.float16)


def available():
    return ENABLED and mx.metal.is_available() and mx.default_device() == mx.gpu


def projects_block(x):
    """Whether ``x`` (..., K) is a block the MXFP8 kernels take over from MLX."""
    return ENABLED and FP8_MIN_ROWS <= x.size // x.shape[-1] <= MAX_ROWS


def _bits(value):
    return struct.unpack("<i", struct.pack("<f", float(value)))[0]


def _rows(x, trailing):
    count = 1
    for size in x.shape[: x.ndim - trailing]:
        count *= size
    return count


# --------------------------------------------------------------------------
# Hyper-connections
# --------------------------------------------------------------------------

# One threadgroup per (row, mix): the dot of one ``fn`` row with the row's
# HC * D stream values and their sum of squares. With EXPAND the stream is
# first rebuilt from the previous sublayer's output:
#   stream[s] = post[s] * h + sum_t comb[t][s] * res[t]
# rounded to T as the composed path stores it, and written by mix 0.
_HC_PROJECT = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint group = threadgroup_position_in_grid.x;
    const uint row = group / MIX;
    const uint m = group % MIX;
    constexpr uint D4 = D / 4;
    constexpr uint N4 = HC * D4;
    using T4 = vec<T, 4>;
    const device float4* f4 = (const device float4*)(fn + size_t(m) * HC * D);
    const device T4* r4 = (const device T4*)(res + size_t(row) * HC * D);
    EXPAND_SETUP
    float acc = 0;
    float sq = 0;
    for (uint c = tid; c < N4; c += TG) {
        EXPAND_VALUE
        acc += dot(f4[c], v);
        sq += dot(v, v);
    }
    acc = simd_sum(acc);
    sq = simd_sum(sq);
    threadgroup float part_acc[32];
    threadgroup float part_sq[32];
    if (lane == 0) {
        part_acc[sg] = acc;
        part_sq[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float a = 0;
        float q = 0;
        for (uint i = 0; i < TG / 32; ++i) {
            a += part_acc[i];
            q += part_sq[i];
        }
        const float eps = as_type<float>(uint(EPS_BITS));
        mixes[row * MIX + m] = a / (metal::precise::sqrt(q / float(HC * D)) + eps);
    }
"""

_EXPAND_SETUP = r"""
    const device T4* h4 = (const device T4*)(h + size_t(row) * D);
    device T4* o4 = (device T4*)(stream + size_t(row) * HC * D);
"""

_EXPAND_VALUE = r"""
        const uint s = c / D4;
        const uint d = c % D4;
        float4 e = float4(r4[d]) * comb[row * HC * HC + s];
        for (uint t = 1; t < HC; ++t) {
            e += float4(r4[t * D4 + d]) * comb[(row * HC + t) * HC + s];
        }
        e += float4(h4[d]) * post[row * HC + s];
        const T4 stored = T4(e);
        if (m == 0) {
            o4[c] = stored;
        }
        const float4 v = float4(stored);
"""


@lru_cache(maxsize=None)
def _hc_project_kernel(expand):
    source = _HC_PROJECT.replace(
        "EXPAND_SETUP", _EXPAND_SETUP if expand else ""
    ).replace(
        "EXPAND_VALUE", _EXPAND_VALUE if expand else "const float4 v = float4(r4[c]);"
    )
    return mx.fast.metal_kernel(
        name="ernie_hc_expand_project" if expand else "ernie_hc_project",
        input_names=["res", "fn", "h", "post", "comb"] if expand else ["res", "fn"],
        output_names=["mixes", "stream"] if expand else ["mixes"],
        source=source,
    )


def _project_threads(values):
    # 16 float4 loads per thread on the real shape; never more threads than
    # there is work.
    return 32 * max(1, min(8, (values // 4 + 31) // 32))


def _hc_shapes_ok(x, fn):
    if x.ndim != 4 or fn.ndim != 2:
        return False
    B, L, HC, D = x.shape
    return (
        available()
        and x.dtype in _HALF
        and fn.dtype == mx.float32
        and D % 4 == 0
        and fn.shape[1] == HC * D
        and 1 <= B * L <= MAX_ROWS
    )


def hc_project(x, fn, norm_eps):
    """Normalized hyper-connection mixes of ``x`` (B, L, HC, D)."""
    if not _hc_shapes_ok(x, fn):
        return None
    B, L, HC, D = x.shape
    mix = fn.shape[0]
    threads = _project_threads(HC * D)
    return _hc_project_kernel(False)(
        inputs=[x, fn],
        template=[
            ("T", x.dtype),
            ("HC", HC),
            ("D", D),
            ("MIX", mix),
            ("TG", threads),
            ("EPS_BITS", _bits(norm_eps)),
        ],
        grid=(B * L * mix * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(B, L, mix)],
        output_dtypes=[mx.float32],
    )[0]


def hc_expand_project(h, residual, post, comb, fn, norm_eps):
    """Expand a sublayer output into the stream and project the result.

    Returns ``(stream, mixes)``: the stream ``hc_expand`` would produce and
    its normalized mixes under ``fn``, the next sublayer's projection.
    """
    if not _hc_shapes_ok(residual, fn) or h.dtype != residual.dtype:
        return None
    B, L, HC, D = residual.shape
    if (
        h.shape != (B, L, D)
        or post.shape != (B, L, HC)
        or comb.shape != (B, L, HC, HC)
        or post.dtype != mx.float32
        or comb.dtype != mx.float32
    ):
        return None
    mix = fn.shape[0]
    threads = _project_threads(HC * D)
    mixes, stream = _hc_project_kernel(True)(
        inputs=[residual, fn, h, post, comb],
        template=[
            ("T", residual.dtype),
            ("HC", HC),
            ("D", D),
            ("MIX", mix),
            ("TG", threads),
            ("EPS_BITS", _bits(norm_eps)),
        ],
        grid=(B * L * mix * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(B, L, mix), (B, L, HC, D)],
        output_dtypes=[mx.float32, residual.dtype],
    )
    return stream, mixes


# One threadgroup per row. Simdgroup 0 runs the Sinkhorn iterations exactly as
# ``hc_sinkhorn_collapse`` does; then every thread collapses its share of the
# streams, rounds it to T, and the row is RMS-normalized with the layer norm's
# weight.
_HC_COLLAPSE_NORM = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;

    constexpr int MIX = (2 + HC) * HC;
    constexpr int BASE_OFF = 2 * HC;
    constexpr float EPS = EPS_INT * 1e-9;

    const device float* mix = mixes + row * MIX;
    device float* post_out = post + row * HC;
    device float* comb_out = comb + row * HC * HC;

    threadgroup float pre_shared[HC];

    if (sg == 0) {
        const float pre_scale = scale[0];
        const float post_scale = scale[1];
        const float comb_scale = scale[2];

        const float active = (lane < (uint)HC) ? 1.0f : 0.0f;
        const uint llane = metal::min(lane, (uint)(HC - 1));

        float pre_z = mix[llane] * pre_scale + base[llane];
        float post_z = mix[HC + llane] * post_scale + base[HC + llane];
        float pre_v = 1.0f / (1.0f + metal::fast::exp(-pre_z)) + EPS;
        float post_v = 2.0f / (1.0f + metal::fast::exp(-post_z));

        if (lane < (uint)HC) {
            pre_shared[lane] = pre_v;
            post_out[lane] = post_v;
        }

        float4 v = (*(const device float4*)(mix + BASE_OFF + llane * HC) * comb_scale
                    + *(const device float4*)(base + BASE_OFF + llane * HC))
                   * active;
        float row_max = metal::max(metal::max(v.x, v.y), metal::max(v.z, v.w));
        float4 e = metal::fast::exp(v - row_max) * active;
        float4 r = e * (1.0f / (e.x + e.y + e.z + e.w + EPS)) + EPS * active;

        float4 col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y),
                                        simd_sum(r.z), simd_sum(r.w)) + EPS);
        r *= col_inv;
        for (int iter = 1; iter < ITERS; ++iter) {
            r *= (1.0f / (r.x + r.y + r.z + r.w + EPS)) * active;
            col_inv = 1.0f / (float4(simd_sum(r.x), simd_sum(r.y),
                                     simd_sum(r.z), simd_sum(r.w)) + EPS);
            r *= col_inv;
        }
        if (lane < (uint)HC) {
            *(device float4*)(comb_out + lane * HC) = r;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float p0 = pre_shared[0];
    const float p1 = pre_shared[1];
    const float p2 = pre_shared[2];
    const float p3 = pre_shared[3];

    using T4 = vec<T, 4>;
    using W4 = vec<W, 4>;
    constexpr uint D4 = D / 4;
    const device T4* x0 = (const device T4*)(x + size_t(row) * HC * D);
    const device T4* x1 = x0 + D4;
    const device T4* x2 = x1 + D4;
    const device T4* x3 = x2 + D4;
    const device W4* w4 = (const device W4*)weight;
    device T4* o4 = (device T4*)(out + size_t(row) * D);

    float4 kept[CHUNKS];
    float acc = 0;
    uint n = 0;
    for (uint d = tid; d < D4; d += TG, ++n) {
        const float4 value = fma(float4(p0), float4(x0[d]),
                             fma(float4(p1), float4(x1[d]),
                             fma(float4(p2), float4(x2[d]), float4(p3) * float4(x3[d]))));
        const float4 stored = float4(T4(value));
        kept[n] = stored;
        acc += dot(stored, stored);
    }
    acc = simd_sum(acc);
    threadgroup float partial[32];
    threadgroup float inv_rms[1];
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint i = 0; i < TG / 32; ++i) {
            total += partial[i];
        }
        inv_rms[0] = metal::precise::rsqrt(
            total / float(D) + as_type<float>(uint(NORM_EPS_BITS)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_rms[0];
    n = 0;
    for (uint d = tid; d < D4; d += TG, ++n) {
        o4[d] = T4(float4(w4[d]) * float4(T4(kept[n] * inv)));
    }
"""


@lru_cache(maxsize=None)
def _hc_collapse_norm_kernel():
    return mx.fast.metal_kernel(
        name="ernie_hc_collapse_norm",
        input_names=["x", "mixes", "scale", "base", "weight"],
        output_names=["out", "post", "comb"],
        source=_HC_COLLAPSE_NORM,
    )


def hc_collapse_norm(x, mixes, scale, base, weight, iters, eps, norm_eps):
    """Sinkhorn, collapse and layer norm for ``x`` (B, L, 4, D).

    Returns ``(normed, post, comb)``: ``norm(collapsed)``, and the factors the
    expand step needs.
    """
    if x.ndim != 4 or not available():
        return None
    B, L, HC, D = x.shape
    if (
        HC != 4
        or x.dtype not in _HALF
        or D % 4
        or not 1 <= B * L <= MAX_ROWS
        or mixes.shape != (B, L, (2 + HC) * HC)
        or mixes.dtype != mx.float32
        or scale.dtype != mx.float32
        or base.dtype != mx.float32
        or weight.shape != (D,)
        or weight.dtype not in (*_HALF, mx.float32)
    ):
        return None
    threads = 32 * max(1, min(32, (D // 4 + 31) // 32))
    chunks = (D // 4 + threads - 1) // threads
    return _hc_collapse_norm_kernel()(
        inputs=[x, mixes, scale, base, weight],
        template=[
            ("T", x.dtype),
            ("W", weight.dtype),
            ("HC", HC),
            ("D", D),
            ("ITERS", int(iters)),
            ("EPS_INT", round(eps / 1e-9)),
            ("NORM_EPS_BITS", _bits(norm_eps)),
            ("TG", threads),
            ("CHUNKS", chunks),
        ],
        grid=(B * L * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(B, L, D), (B, L, HC), (B, L, HC, HC)],
        output_dtypes=[x.dtype, mx.float32, mx.float32],
    )


# --------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------

# MLX unary_ops.h Sigmoid.
_SIGMOID = r"""
inline float ernie_sigmoid(float x) {
    float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
    return (x < 0.0f) ? y : 1.0f - y;
}
"""

# One threadgroup per output: the rows of two half-precision matrices against
# every token row, read where they are stored and accumulated in fp32.
# Simdgroup g covers a contiguous quarter (two per matrix) of the K values.
_PAIR_PROJECT = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint e = threadgroup_position_in_grid.x;
    using W4 = vec<W, 4>;
    using T4 = vec<T, 4>;
    constexpr uint K4 = K / 4;
    constexpr uint HALF = K4 / 2;
    constexpr uint SPAN = HALF / 32;
    const uint which = sg / 2;
    const uint first = (sg % 2) * HALF + lane * SPAN;
    const device W4* w = (const device W4*)(which == 0 ? w0 : w1) + size_t(e) * K4 + first;
    const device T4* x4 = (const device T4*)x + first;
    float acc[ROWS];
    for (int r = 0; r < ROWS; ++r) {
        acc[r] = 0;
    }
    for (uint i = 0; i < SPAN; ++i) {
        const float4 wv = float4(w[i]);
        for (int r = 0; r < ROWS; ++r) {
            acc[r] += dot(wv, float4(x4[size_t(r) * K4 + i]));
        }
    }
    threadgroup float part[4][ROWS];
    for (int r = 0; r < ROWS; ++r) {
        const float total = simd_sum(acc[r]);
        if (lane == 0) {
            part[sg][r] = total;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < (uint)ROWS) {
        const float a = part[0][lane] + part[1][lane];
        const float b = part[2][lane] + part[3][lane];
        FINISH
    }
"""


@lru_cache(maxsize=None)
def _pair_project_kernel(scores):
    finish = (
        "out[size_t(lane) * E + e] = ernie_sigmoid(a) + ernie_sigmoid(b);"
        if scores
        else "out[size_t(lane) * E + e] = a; out[size_t(ROWS + lane) * E + e] = b;"
    )
    return mx.fast.metal_kernel(
        name="ernie_router_scores" if scores else "ernie_pair_project",
        input_names=["x", "w0", "w1"],
        output_names=["out"],
        header=_SIGMOID,
        source=_PAIR_PROJECT.replace("FINISH", finish),
    )


def _pair_project(x, w0, w1, scores):
    if not available() or x.ndim < 2 or w0.shape != w1.shape or w0.ndim != 2:
        return None
    width = x.shape[-1]
    rows = _rows(x, 1)
    outputs = w0.shape[0]
    if (
        x.dtype not in _HALF
        or w0.dtype not in _HALF
        or w1.dtype != w0.dtype
        or w0.shape[1] != width
        or width % 256
        or not 1 <= rows <= MAX_ROWS
        or rows * outputs < 8
    ):
        return None
    return _pair_project_kernel(scores)(
        inputs=[x, w0, w1],
        template=[
            ("T", x.dtype),
            ("W", w0.dtype),
            ("K", width),
            ("E", outputs),
            ("ROWS", rows),
        ],
        grid=(outputs * 128, 1, 1),
        threadgroup=(128, 1, 1),
        output_shapes=[(rows, outputs) if scores else (2, rows, outputs)],
        output_dtypes=[mx.float32],
    )[0]


def router_scores(x, w0, w1):
    """``sigmoid(x @ w0.T) + sigmoid(x @ w1.T)`` in fp32, ``(rows, experts)``.

    ``x`` holds up to eight token rows in half precision.
    """
    return _pair_project(x, w0, w1, True)


def pair_project(x, w0, w1):
    """``x @ w0.T`` and ``x @ w1.T`` in fp32, stacked as ``(2, rows, N)``."""
    return _pair_project(x, w0, w1, False)


# One threadgroup per token row, one thread per expert. TOPK rounds of a
# threadgroup argmax on score + bias (ties to the lower index), then the
# selected scores are normalized and scaled.
_ROUTER_SELECT = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    constexpr uint GROUPS = TG / 32;
    constexpr uint NONE = 0xFFFFFFFFu;

    const bool real = tid < (uint)E;
    const float score = real ? scores[size_t(row) * E + tid] : 0.0f;
    float key = real ? score + bias[tid] : -INFINITY;

    threadgroup float group_key[GROUPS];
    threadgroup uint group_id[GROUPS];
    threadgroup float picked_score[TOPK];
    threadgroup uint picked_id[TOPK];

    for (int k = 0; k < TOPK; ++k) {
        const float best_here = simd_max(key);
        const uint id_here = simd_min(key == best_here ? tid : NONE);
        if (lane == 0) {
            group_key[sg] = best_here;
            group_id[sg] = id_here;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float best = group_key[0];
        uint id = group_id[0];
        for (uint g = 1; g < GROUPS; ++g) {
            const float other = group_key[g];
            const uint other_id = group_id[g];
            if (other > best || (other == best && other_id < id)) {
                best = other;
                id = other_id;
            }
        }
        if (tid == id) {
            picked_score[k] = score;
            picked_id[k] = tid;
            key = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 0) {
        const bool used = lane < (uint)TOPK;
        const uint slot = used ? lane : 0;
        const float mine = used ? picked_score[slot] : 0.0f;
        const float total = simd_sum(mine);
        if (used) {
            const uint id = picked_id[slot];
            float weight = mine;
            if (NORMALIZE) {
                weight = weight / metal::max(total, 1e-20f);
            }
            ids[size_t(row) * TOPK + slot] = id;
            weights[size_t(row) * TOPK + slot] = weight * scaling[id];
        }
    }
"""


@lru_cache(maxsize=None)
def _router_select_kernel():
    return mx.fast.metal_kernel(
        name="ernie_router_select",
        input_names=["scores", "bias", "scaling"],
        output_names=["ids", "weights"],
        source=_ROUTER_SELECT,
    )


def router_select(scores, bias, scaling, top_k, normalize):
    """Top ``top_k`` experts of each row by ``scores + bias`` and their weights.

    Returns ``(ids, weights)`` shaped ``(rows, top_k)``: the weights are the
    selected scores, normalized to sum to one when ``normalize`` is set, times
    ``scaling`` of each selected expert. Experts come out best first.
    """
    if not available() or scores.ndim != 2:
        return None
    rows, experts = scores.shape
    if (
        scores.dtype != mx.float32
        or bias.dtype != mx.float32
        or scaling.dtype != mx.float32
        or bias.shape != (experts,)
        or scaling.shape != (experts,)
        or not 1 <= rows <= MAX_ROWS
        or not 1 <= top_k <= min(32, experts)
        or experts > 1024
        or experts < 8
    ):
        return None
    threads = 32 * ((experts + 31) // 32)
    ids, weights = _router_select_kernel()(
        inputs=[scores, bias, scaling],
        template=[
            ("E", experts),
            ("TOPK", int(top_k)),
            ("TG", threads),
            ("NORMALIZE", 1 if normalize else 0),
        ],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, top_k), (rows, top_k)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return ids, weights


# --------------------------------------------------------------------------
# Routed MXFP4 experts
# --------------------------------------------------------------------------

# The two kernels below follow oMLX's shared FP4 expert kernels
# (omlx/patches/fp4_expert_rows.py): each output row is the reduction of MLX's
# ``fp_qmv_fast_impl``, so the sums match ``gather_qmm`` bit for bit, and an
# expert picked by several rows of a block is streamed once for all of them.
_FP4 = r"""
// MLX fp4_e2m1 nibble -> float (via half).
inline float fp4_value(uint bits) {
    half value = as_type<half>(ushort((bits & 7) << 9));
    value *= 16384.0;
    return float((bits & 8) ? -value : value);
}

// MLX fp8_e8m0, the MXFP4 group scale.
inline float fp4_scale(uint8_t bits) {
    return as_type<float>(bits == 0 ? 0x400000u : (uint(bits) << 23));
}

// MLX fp_quantized.h qdot<float, 16, 4> on 8 packed bytes.
inline float fp4_qdot16(uint2 w, thread const float* x, float scale) {
    ushort ws[4] = {
        ushort(w.x & 0xFFFF), ushort(w.x >> 16), ushort(w.y & 0xFFFF), ushort(w.y >> 16)};
    float accum = 0;
    for (int i = 0; i < 4; ++i) {
        accum += (x[4 * i] * fp4_value(ws[i]) + x[4 * i + 1] * fp4_value(ws[i] >> 4)
                  + x[4 * i + 2] * fp4_value(ws[i] >> 8)
                  + x[4 * i + 3] * fp4_value(ws[i] >> 12));
    }
    return scale * accum;
}
"""

# Slot u of a block's plan names one distinct expert and the pairs
# (row * TOPK + k) that picked it, so an expert shared by several rows is
# streamed once. A one-row block needs no plan: every pair is its own slot.
_FP4_PLAN = r"""
    const uint p = thread_position_in_threadgroup.x;
    constexpr uint NONE = 0xFFFFFFFFu;
    constexpr uint PAIRS = ROWS * TOPK;
    constexpr uint WIDTH = ROWS + 1;
    threadgroup uint first[PAIRS];
    const bool real = p < PAIRS;
    const uint e = real ? inds[p] : NONE;
    bool is_first = real;
    for (uint q = 0; q < p && q < PAIRS; ++q) {
        if (inds[q] == e) {
            is_first = false;
        }
    }
    if (real) {
        first[p] = is_first ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!real) {
        return;
    }
    uint before = 0;
    uint total = 0;
    for (uint q = 0; q < PAIRS; ++q) {
        total += first[q];
        if (q < p) {
            before += first[q];
        }
    }
    if (is_first) {
        plan[before * WIDTH] = e;
        uint count = 0;
        for (uint q = p; q < PAIRS; ++q) {
            if (inds[q] == e && count < (uint)ROWS) {
                plan[before * WIDTH + 1 + count] = q;
                ++count;
            }
        }
        for (; count < (uint)ROWS; ++count) {
            plan[before * WIDTH + 1 + count] = NONE;
        }
    }
    if (p >= total) {
        plan[p * WIDTH] = NONE;
    }
"""

_FP4_SLOT_PLANNED = r"""
    constexpr uint NONE = 0xFFFFFFFFu;
    const uint slot = threadgroup_position_in_grid.x;
    const uint e = plan[slot * (ROWS + 1)];
    if (e == NONE) {
        return;
    }
    uint members[ROWS];
    int count = 0;
    for (int j = 0; j < ROWS; ++j) {
        const uint member = plan[slot * (ROWS + 1) + 1 + j];
        if (member != NONE) {
            members[count++] = member;
        }
    }
"""

_FP4_SLOT_SINGLE = r"""
    const uint slot = threadgroup_position_in_grid.x;
    const uint e = plan[slot];
    uint members[1] = {slot};
    constexpr int count = 1;
"""

_FP4_PAIRS = r"""
    SLOT
    const uint simdgroup = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int n0 = (int(threadgroup_position_in_grid.y) * NSG + int(simdgroup)) * RPS;
    constexpr int KW = KDIM / 8;
    constexpr int KS = KDIM / 32;
    constexpr int BLOCK = 512;
    constexpr int SB = BLOCK / 32;
    const int lane_scale = int(lane) / 2;
"""

_FP4_GATE_UP = _FP4_PAIRS + r"""
    const device uint32_t* gw = wg + (size_t(e) * NOUT + n0) * KW + lane * 2;
    const device uint32_t* uw = wu + (size_t(e) * NOUT + n0) * KW + lane * 2;
    const device uint8_t* gs = sg + (size_t(e) * NOUT + n0) * KS + lane_scale;
    const device uint8_t* us = su + (size_t(e) * NOUT + n0) * KS + lane_scale;

    float acc_g[ROWS][RPS];
    float acc_u[ROWS][RPS];
    for (int j = 0; j < ROWS; ++j) {
        for (int r = 0; r < RPS; ++r) {
            acc_g[j][r] = 0;
            acc_u[j][r] = 0;
        }
    }
    for (int b = 0; b < KDIM / BLOCK; ++b) {
        uint2 wgv[RPS];
        uint2 wuv[RPS];
        float gscale[RPS];
        float uscale[RPS];
        for (int r = 0; r < RPS; ++r) {
            wgv[r] = *(const device uint2*)(gw + r * KW + b * 64);
            wuv[r] = *(const device uint2*)(uw + r * KW + b * 64);
            gscale[r] = fp4_scale(gs[r * KS + b * SB]);
            uscale[r] = fp4_scale(us[r * KS + b * SB]);
        }
        for (int j = 0; j < ROWS; ++j) {
            if (j < count) {
                const device T* xp =
                    x + size_t(members[j] / TOPK) * KDIM + b * BLOCK + lane * 16;
                float xt[16];
                for (int i = 0; i < 16; ++i) {
                    xt[i] = float(xp[i]);
                }
                for (int r = 0; r < RPS; ++r) {
                    acc_g[j][r] += fp4_qdot16(wgv[r], xt, gscale[r]);
                    acc_u[j][r] += fp4_qdot16(wuv[r], xt, uscale[r]);
                }
            }
        }
    }
    const float beta = as_type<float>(uint(BETA_BITS));
    const float linear_beta = as_type<float>(uint(LINEAR_BETA_BITS));
    for (int j = 0; j < ROWS; ++j) {
        if (j < count) {
            for (int r = 0; r < RPS; ++r) {
                const float g = simd_sum(acc_g[j][r]);
                const float u = simd_sum(acc_u[j][r]);
                if (lane == 0) {
                    // SiTU on the half-precision projections, in fp32.
                    const float gt = float(static_cast<T>(g));
                    const float ut = float(static_cast<T>(u));
                    const float gate = beta * metal::precise::tanh(gt / beta)
                                       * ernie_sigmoid(gt);
                    const float up = linear_beta * metal::precise::tanh(ut / linear_beta);
                    act[size_t(members[j]) * NOUT + n0 + r] = static_cast<T>(gate * up);
                }
            }
        }
    }
"""

_FP4_DOWN = _FP4_PAIRS + r"""
    const device uint32_t* dw = w + (size_t(e) * NOUT + n0) * KW + lane * 2;
    const device uint8_t* ds = s + (size_t(e) * NOUT + n0) * KS + lane_scale;

    float acc[ROWS][RPS];
    for (int j = 0; j < ROWS; ++j) {
        for (int r = 0; r < RPS; ++r) {
            acc[j][r] = 0;
        }
    }
    for (int b = 0; b < KDIM / BLOCK; ++b) {
        uint2 wv[RPS];
        float scale[RPS];
        for (int r = 0; r < RPS; ++r) {
            wv[r] = *(const device uint2*)(dw + r * KW + b * 64);
            scale[r] = fp4_scale(ds[r * KS + b * SB]);
        }
        for (int j = 0; j < ROWS; ++j) {
            if (j < count) {
                const device T* ap = a + size_t(members[j]) * KDIM + b * BLOCK + lane * 16;
                float at[16];
                for (int i = 0; i < 16; ++i) {
                    at[i] = float(ap[i]);
                }
                for (int r = 0; r < RPS; ++r) {
                    acc[j][r] += fp4_qdot16(wv[r], at, scale[r]);
                }
            }
        }
    }
    for (int j = 0; j < ROWS; ++j) {
        if (j < count) {
            for (int r = 0; r < RPS; ++r) {
                const float v = simd_sum(acc[j][r]);
                if (lane == 0) {
                    y[size_t(members[j]) * NOUT + n0 + r] = static_cast<T>(v);
                }
            }
        }
    }
"""

# Output rows per simdgroup; two measured fastest on an M5 Ultra.
GATE_UP_RPS = 2
DOWN_RPS = 2
_FP4_NSG = 2


def _slot(source, planned):
    return source.replace("SLOT", _FP4_SLOT_PLANNED if planned else _FP4_SLOT_SINGLE)


@lru_cache(maxsize=None)
def _fp4_plan_kernel():
    return mx.fast.metal_kernel(
        name="ernie_fp4_plan",
        input_names=["inds"],
        output_names=["plan"],
        source=_FP4_PLAN,
    )


@lru_cache(maxsize=None)
def _fp4_gate_up_kernel(planned):
    return mx.fast.metal_kernel(
        name="ernie_fp4_gate_up" + ("_planned" if planned else ""),
        input_names=["x", "plan", "wg", "sg", "wu", "su"],
        output_names=["act"],
        header=_FP4 + _SIGMOID,
        source=_slot(_FP4_GATE_UP, planned),
    )


@lru_cache(maxsize=None)
def _fp4_down_kernel(planned):
    return mx.fast.metal_kernel(
        name="ernie_fp4_down" + ("_planned" if planned else ""),
        input_names=["a", "plan", "w", "s"],
        output_names=["y"],
        header=_FP4,
        source=_slot(_FP4_DOWN, planned),
    )


def _fp4_layer_ok(layer, width):
    weight, scales = layer.get("weight"), layer.get("scales")
    return (
        getattr(layer, "mode", None) == "mxfp4"
        and getattr(layer, "bits", None) == 4
        and getattr(layer, "group_size", None) == 32
        and layer.get("biases") is None
        and layer.get("bias") is None
        and weight is not None
        and scales is not None
        and weight.ndim == 3
        and weight.dtype == mx.uint32
        and scales.dtype == mx.uint8
        and weight.shape[2] * 8 == width
        and width % 512 == 0
    )


def routed_experts(switch, x, ids, beta, linear_beta):
    """``switch(x, ids)`` for a decode block of MXFP4 experts with SiTU.

    ``x`` (rows, K) half precision, ``ids`` (rows, top_k) uint32. Returns the
    expert outputs ``(rows, top_k, K)``.
    """
    if not available() or x.ndim != 2 or ids.ndim != 2:
        return None
    rows, width = x.shape
    top_k = ids.shape[1]
    gate, up, down = switch.gate_proj, switch.up_proj, switch.down_proj
    if (
        x.dtype not in _HALF
        or ids.dtype != mx.uint32
        or ids.shape[0] != rows
        or not 1 <= rows <= MAX_ROWS
        or rows * top_k > 1024
        or rows * top_k < 8
        or not _fp4_layer_ok(gate, width)
        or not _fp4_layer_ok(up, width)
        or gate.weight.shape != up.weight.shape
    ):
        return None
    hidden = gate.weight.shape[1]
    if (
        not _fp4_layer_ok(down, hidden)
        or hidden % (_FP4_NSG * GATE_UP_RPS)
        or down.weight.shape[1] % (_FP4_NSG * DOWN_RPS)
    ):
        return None
    pairs = rows * top_k
    planned = rows > 1
    plan = ids
    if planned:
        threads = 32 * ((pairs + 31) // 32)
        plan = _fp4_plan_kernel()(
            inputs=[ids],
            template=[("ROWS", rows), ("TOPK", top_k)],
            grid=(threads, 1, 1),
            threadgroup=(threads, 1, 1),
            output_shapes=[(pairs, rows + 1)],
            output_dtypes=[mx.uint32],
        )[0]
    common = [("T", x.dtype), ("TOPK", top_k), ("ROWS", rows), ("NSG", _FP4_NSG)]
    act = _fp4_gate_up_kernel(planned)(
        inputs=[x, plan, gate.weight, gate.scales, up.weight, up.scales],
        template=common
        + [
            ("KDIM", width),
            ("NOUT", hidden),
            ("RPS", GATE_UP_RPS),
            ("BETA_BITS", _bits(beta)),
            ("LINEAR_BETA_BITS", _bits(linear_beta)),
        ],
        grid=(32 * pairs, _FP4_NSG * (hidden // (_FP4_NSG * GATE_UP_RPS)), 1),
        threadgroup=(32, _FP4_NSG, 1),
        output_shapes=[(pairs, hidden)],
        output_dtypes=[x.dtype],
    )[0]
    out = down.weight.shape[1]
    y = _fp4_down_kernel(planned)(
        inputs=[act, plan, down.weight, down.scales],
        template=common + [("KDIM", hidden), ("NOUT", out), ("RPS", DOWN_RPS)],
        grid=(32 * pairs, _FP4_NSG * (out // (_FP4_NSG * DOWN_RPS)), 1),
        threadgroup=(32, _FP4_NSG, 1),
        output_shapes=[(pairs, out)],
        output_dtypes=[x.dtype],
    )[0]
    return y.reshape(rows, top_k, out)


# One threadgroup per row: the score-weighted sum of the expert outputs in
# fp32, rounded to T as the composed path stores it, then RMS-normalized.
_COMBINE_NORM = r"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint row = threadgroup_position_in_grid.x;
    using T4 = vec<T, 4>;
    using W4 = vec<W, 4>;
    constexpr uint D4 = D / 4;
    const device T4* y4 = (const device T4*)(y + size_t(row) * TOPK * D);

    float w[TOPK];
    for (int k = 0; k < TOPK; ++k) {
        w[k] = weights[row * TOPK + k];
    }
    float4 kept[CHUNKS];
    float acc = 0;
    uint n = 0;
    for (uint d = tid; d < D4; d += TG, ++n) {
        float4 value = float4(y4[d]) * w[0];
        for (int k = 1; k < TOPK; ++k) {
            value += float4(y4[k * D4 + d]) * w[k];
        }
        const float4 stored = float4(T4(value));
        kept[n] = stored;
        acc += dot(stored, stored);
    }
    acc = simd_sum(acc);
    threadgroup float partial[32];
    threadgroup float inv_rms[1];
    if (lane == 0) {
        partial[sg] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0;
        for (uint i = 0; i < TG / 32; ++i) {
            total += partial[i];
        }
        inv_rms[0] = metal::precise::rsqrt(
            total / float(D) + as_type<float>(uint(NORM_EPS_BITS)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = inv_rms[0];
    const device W4* w4 = (const device W4*)weight;
    device T4* o4 = (device T4*)(out + size_t(row) * D);
    n = 0;
    for (uint d = tid; d < D4; d += TG, ++n) {
        o4[d] = T4(float4(w4[d]) * float4(T4(kept[n] * inv)));
    }
"""


@lru_cache(maxsize=None)
def _combine_norm_kernel():
    return mx.fast.metal_kernel(
        name="ernie_moe_combine_norm",
        input_names=["y", "weights", "weight"],
        output_names=["out"],
        source=_COMBINE_NORM,
    )


def combine_norm(y, weights, weight, norm_eps):
    """``norm((y * weights[..., None]).sum(-2))`` for ``y`` (rows, top_k, D)."""
    if not available() or y.ndim != 3:
        return None
    rows, top_k, width = y.shape
    if (
        y.dtype not in _HALF
        or weights.shape != (rows, top_k)
        or weights.dtype != mx.float32
        or weight.shape != (width,)
        or weight.dtype not in (*_HALF, mx.float32)
        or width % 4
        or not 1 <= rows <= MAX_ROWS
        or rows * top_k < 8
    ):
        return None
    threads = 32 * max(1, min(32, (width // 4 + 31) // 32))
    chunks = (width // 4 + threads - 1) // threads
    return _combine_norm_kernel()(
        inputs=[y, weights, weight],
        template=[
            ("T", y.dtype),
            ("W", weight.dtype),
            ("D", width),
            ("TOPK", top_k),
            ("TG", threads),
            ("CHUNKS", chunks),
            ("NORM_EPS_BITS", _bits(norm_eps)),
        ],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[(rows, width)],
        output_dtypes=[y.dtype],
    )[0]


# --------------------------------------------------------------------------
# Dense MXFP8 projections
# --------------------------------------------------------------------------

# MLX's ``fp_qmv`` arithmetic for MXFP8, as oMLX's MiniMax decode kernels
# reproduce it: lane l takes values [256 b + 8 l, +8) of every 256-wide block
# b, an 8-term dot in order times the E8M0 scale of its 32-value group, summed
# block by block. An E4M3 code placed in a half's sign, exponent and mantissa
# is the code's value times 2^-8.
_FP8 = r"""
inline void fp8x4(uint p, thread float* v) {
    const float2 a = float2(as_type<half2>(((p & 0x00800080u) << 8) | ((p & 0x007F007Fu) << 7)));
    const uint q = p >> 8;
    const float2 b = float2(as_type<half2>(((q & 0x00800080u) << 8) | ((q & 0x007F007Fu) << 7)));
    v[0] = a.x * 256.0f;
    v[1] = b.x * 256.0f;
    v[2] = a.y * 256.0f;
    v[3] = b.y * 256.0f;
}

inline float fp8_scale(uint8_t bits) {
    return as_type<float>(bits == 0 ? 0x400000u : (uint(bits) << 23));
}
"""

# Simdgroup s computes output rows [n0, n0 + RPS), one after the other, for
# every token row, so a weight is decoded once however long the block is and a
# lane holds one accumulator per token row. With XG input groups, output row n
# reads group n / N of the token's (XG, K) input, which is how the grouped
# output projection runs without transposing.
_FP8_ROWS = r"""
    const uint lane = thread_index_in_simdgroup;
    const int n0 = (int(threadgroup_position_in_grid.y) * NSG
                    + int(simdgroup_index_in_threadgroup)) * RPS;
    constexpr int NTOTAL = XG * N;
    if (n0 >= NTOTAL) {
        return;
    }
    const int g = n0 / N;
    constexpr int KS = K / 32;
    const device T* xg = x + size_t(g) * K + lane * 8;
    FP8_SETUP
    for (int r = 0; r < RPS; ++r) {
        const size_t n = size_t(n0 + r);
        float acc[ACCS][ROWS];
        for (int a = 0; a < ACCS; ++a) {
            for (int j = 0; j < ROWS; ++j) {
                acc[a][j] = 0;
            }
        }
        for (int a = 0; a < ACCS; ++a) {
            const device uint8_t* wp = (const device uint8_t*)(a == 0 ? w : w1) + n * K + lane * 8;
            const device uint8_t* sp = (a == 0 ? s : s1) + n * KS + lane / 4;
            for (int k = 0; k < K; k += 256) {
                const uint2 wv = *(const device uint2*)(wp + k);
                float v[8];
                fp8x4(wv.x, v);
                fp8x4(wv.y, v + 4);
                const float scale = fp8_scale(sp[k / 32]);
                for (int j = 0; j < ROWS; ++j) {
                    const device T* xp = xg + size_t(j) * XG * K + k;
                    float sum = 0;
                    for (int i = 0; i < 8; ++i) {
                        sum += float(xp[i]) * v[i];
                    }
                    acc[a][j] += scale * sum;
                }
            }
        }
        for (int j = 0; j < ROWS; ++j) {
            FP8_STORE
        }
    }
"""

_FP8_ONE = r"""
    constexpr int ACCS = 1;
    const device uint32_t* w1 = w;
    const device uint8_t* s1 = s;
"""

_FP8_ONE_STORE = r"""
            const float total = simd_sum(acc[0][j]);
            if (lane == 0) {
                y[size_t(j) * NTOTAL + n] = static_cast<T>(total);
            }
"""

_FP8_TWO = r"""
    constexpr int ACCS = 2;
    const float beta = as_type<float>(uint(BETA_BITS));
    const float linear_beta = as_type<float>(uint(LINEAR_BETA_BITS));
"""

# ``w`` is the gate projection and ``w1`` the up projection; SiTU as in the
# routed experts.
_FP8_TWO_STORE = r"""
            const float gsum = simd_sum(acc[0][j]);
            const float usum = simd_sum(acc[1][j]);
            if (lane == 0) {
                const float gt = float(static_cast<T>(gsum));
                const float ut = float(static_cast<T>(usum));
                const float gate = beta * metal::precise::tanh(gt / beta) * ernie_sigmoid(gt);
                const float up = linear_beta * metal::precise::tanh(ut / linear_beta);
                y[size_t(j) * NTOTAL + n] = static_cast<T>(gate * up);
            }
"""

_FP8_NSG = 2
# Weights each lane multiplies per token row, across its simdgroup's outputs.
FP8_LANE_WORK = 64
# Rows a block needs before these kernels beat MLX's.
FP8_MIN_ROWS = 3


@lru_cache(maxsize=None)
def _fp8_kernel(gate_up):
    source = _FP8_ROWS.replace("FP8_SETUP", _FP8_TWO if gate_up else _FP8_ONE).replace(
        "FP8_STORE", _FP8_TWO_STORE if gate_up else _FP8_ONE_STORE
    )
    return mx.fast.metal_kernel(
        name="ernie_fp8_gate_up" if gate_up else "ernie_fp8_rows",
        input_names=["x", "w", "s", "w1", "s1"] if gate_up else ["x", "w", "s"],
        output_names=["y"],
        header=_FP8 + _SIGMOID,
        source=source,
    )


def _fp8_weights(layer):
    """``(weight, scales)`` of an MXFP8 layer these kernels read, or None."""
    weight, scales = layer.get("weight"), layer.get("scales")
    if (
        getattr(layer, "mode", None) != "mxfp8"
        or getattr(layer, "bits", None) != 8
        or getattr(layer, "group_size", None) != 32
        or layer.get("biases") is not None
        or layer.get("bias") is not None
        or weight is None
        or scales is None
        or weight.dtype != mx.uint32
        or scales.dtype != mx.uint8
    ):
        return None
    return weight, scales


def _fp8_launch(x, rows, groups, outputs, width, arrays, extra):
    if rows < FP8_MIN_ROWS:
        # MLX's own one-row kernel is the faster one for a row or two.
        return None
    rps = max(1, FP8_LANE_WORK * 32 // (width * (2 if extra else 1)))
    while outputs % (_FP8_NSG * rps):
        rps //= 2
        if rps == 0:
            return None
    total = groups * outputs
    return _fp8_kernel(bool(extra))(
        inputs=[x, *arrays],
        template=[
            ("T", x.dtype),
            ("K", width),
            ("N", outputs),
            ("XG", groups),
            ("ROWS", rows),
            ("RPS", rps),
            ("NSG", _FP8_NSG),
            *extra,
        ],
        grid=(32, _FP8_NSG * (total // (_FP8_NSG * rps)), 1),
        threadgroup=(32, _FP8_NSG, 1),
        output_shapes=[(rows, total)],
        output_dtypes=[x.dtype],
    )[0]


def _fp8_input_ok(x, width, trailing):
    return (
        available()
        and x.dtype in _HALF
        and x.ndim >= trailing
        and x.shape[-1] == width
        and width % 256 == 0
        and 1 <= _rows(x, trailing) <= MAX_ROWS
    )


def fp8_linear(layer, x):
    """``layer(x)`` for an MXFP8 linear layer and up to eight rows ``x``.

    Each row's result is MLX's one-row product, whatever the block size.
    """
    found = _fp8_weights(layer)
    if found is None or found[0].ndim != 2:
        return None
    weight, scales = found
    outputs, width = weight.shape[0], weight.shape[1] * 4
    if not _fp8_input_ok(x, width, 1):
        return None
    y = _fp8_launch(x, _rows(x, 1), 1, outputs, width, (weight, scales), ())
    return None if y is None else y.reshape(*x.shape[:-1], outputs)


def fp8_grouped(layer, x):
    """A grouped MXFP8 projection of ``x`` (..., groups, K) to (..., groups * N)."""
    found = _fp8_weights(layer)
    if found is None or found[0].ndim != 3:
        return None
    weight, scales = found
    groups, outputs, width = weight.shape[0], weight.shape[1], weight.shape[2] * 4
    if x.ndim < 2 or x.shape[-2] != groups or not _fp8_input_ok(x, width, 2):
        return None
    y = _fp8_launch(x, _rows(x, 2), groups, outputs, width, (weight, scales), ())
    return None if y is None else y.reshape(*x.shape[:-2], groups * outputs)


def fp8_gate_up(gate, up, x, beta, linear_beta):
    """SiTU of an MXFP8 gate and up projection of ``x``, in one weight pass."""
    g, u = _fp8_weights(gate), _fp8_weights(up)
    if g is None or u is None or g[0].ndim != 2 or g[0].shape != u[0].shape:
        return None
    outputs, width = g[0].shape[0], g[0].shape[1] * 4
    if not _fp8_input_ok(x, width, 1):
        return None
    y = _fp8_launch(
        x,
        _rows(x, 1),
        1,
        outputs,
        width,
        (g[0], g[1], u[0], u[1]),
        (("BETA_BITS", _bits(beta)), ("LINEAR_BETA_BITS", _bits(linear_beta))),
    )
    return None if y is None else y.reshape(*x.shape[:-1], outputs)


# --------------------------------------------------------------------------
# Vocabulary head
# --------------------------------------------------------------------------

# One simdgroup per VR vocabulary rows, taken one after the other: each lane
# accumulates its share of the half-precision row against every hidden row in
# fp32. MULTIMAX applies the model's logit reshaping as the value is written,
# which the composed path spends a dozen vocabulary-sized launches on. A lone
# hidden row is accumulated twice: on an M5 Ultra one accumulator per lane
# runs about three times slower than two, here and in MLX's own matmul.
_HEAD = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint first = threadgroup_position_in_grid.x * VR;
    constexpr uint K4 = K / 4;
    constexpr int WORK = ROWS < 2 ? 2 : ROWS;
    using W4 = vec<W, 4>;
    for (uint v = first; v < first + VR && v < (uint)V; ++v) {
        const device W4* w4 = (const device W4*)weight + size_t(v) * K4;
        float acc[WORK];
        for (int r = 0; r < WORK; ++r) {
            acc[r] = 0;
        }
        for (uint i = lane; i < K4; i += 32) {
            const float4 wv = float4(w4[i]);
            for (int r = 0; r < WORK; ++r) {
                const int source = r < ROWS ? r : ROWS - 1;
                acc[r] += dot(wv, *(const device float4*)(hidden + size_t(source) * K + 4 * i));
            }
        }
        for (int r = 0; r < ROWS; ++r) {
            float value = simd_sum(acc[r]);
            BASE
            MULTIMAX
            if (lane == 0) {
                logits[size_t(r) * V + v] = value;
            }
        }
    }
"""

_HEAD_MULTIMAX = r"""
            {
                const float low = metal::max(shape[0] - value, 0.0f);
                const float high = metal::max(value - shape[1], 0.0f);
                const float low2 = metal::max(shape[2] - value, 0.0f);
                const float high2 = metal::max(value - shape[3], 0.0f);
                value = value + shape[4] * low + shape[5] * high
                        + shape[6] * (low2 * low2) + shape[7] * (high2 * high2);
            }
"""

# Vocabulary rows per simdgroup.
HEAD_ROWS_PER_GROUP = 4


@lru_cache(maxsize=None)
def _head_kernel(multimax, based):
    inputs = ["hidden", "weight"]
    if multimax:
        inputs.append("shape")
    if based:
        inputs.append("base")
    return mx.fast.metal_kernel(
        name="ernie_vocab_head"
        + ("_multimax" if multimax else "")
        + ("_base" if based else ""),
        input_names=inputs,
        output_names=["logits"],
        source=_HEAD.replace("MULTIMAX", _HEAD_MULTIMAX if multimax else "").replace(
            "BASE", "value += base[size_t(r) * V + v];" if based else ""
        ),
    )


def vocab_head(hidden, weight, shape=None, base=None):
    """fp32 logits of fp32 hidden rows against a half-precision vocabulary head.

    ``shape`` holds MultiMax's four ranges followed by its four factors; with
    it the logits come out reshaped. ``base`` (rows, vocab) is added before
    that, which is how the draft head applies its Markov correction. Rows are
    projected eight at a time, the weight streamed once for each eight.
    """
    if not available() or hidden.ndim != 2 or weight.ndim != 2:
        return None
    count, width = hidden.shape
    vocab = weight.shape[0]
    if (
        hidden.dtype != mx.float32
        or weight.dtype not in _HALF
        or weight.shape[1] != width
        or width % 128
        or count < 1
        or (shape is not None and (shape.shape != (8,) or shape.dtype != mx.float32))
        or (
            base is not None
            and (
                base.shape != (count, vocab)
                or base.dtype != mx.float32
                or count > MAX_ROWS
            )
        )
    ):
        return None
    per = HEAD_ROWS_PER_GROUP
    groups = (vocab + per - 1) // per
    parts = []
    for start in range(0, count, MAX_ROWS):
        chunk = hidden if count <= MAX_ROWS else hidden[start : start + MAX_ROWS]
        rows = chunk.shape[0]
        inputs = [chunk, weight]
        if shape is not None:
            inputs.append(shape)
        if base is not None:
            inputs.append(base)
        parts.append(
            _head_kernel(shape is not None, base is not None)(
                inputs=inputs,
                template=[
                    ("W", weight.dtype),
                    ("K", width),
                    ("V", vocab),
                    ("ROWS", rows),
                    ("VR", per),
                ],
                grid=(groups * 32, 1, 1),
                threadgroup=(32, 1, 1),
                output_shapes=[(rows, vocab)],
                output_dtypes=[mx.float32],
            )[0]
        )
    return parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)
