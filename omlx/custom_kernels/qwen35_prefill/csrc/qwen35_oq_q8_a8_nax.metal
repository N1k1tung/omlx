// SPDX-License-Identifier: Apache-2.0
// oQ Q8 W8A8 GEMM on the M5 NAX tensor units.
//
// Packed Q8 GS64 affine weights are read as the checkpoint stores them (bytes
// in uint32 words), centered to signed INT8 in registers, and multiplied
// against dynamically quantized INT8 activations through the int8 x int8 ->
// int32 datapath. No unpacked weight matrix is written to device memory.
//
// MLX affine dequantization is w = s * q + b with q in [0, 255]. Flipping the
// top bit of each byte turns q into q - 128 as a two's-complement INT8, so
//
//   sum_k a_k w_k = s * (acc + 128 * r) + b * r
//
// per GS64 group, where acc = sum a_k * (q_k - 128) is the INT32 tensor-op
// result and r = sum a_k is the group sum Stage A already produces. The
// integers acc and 128 * r are far below 2^24, so the FP32 add is exact.
//
// A lane's 16 codes of one group are four words, one per micro-K step, so the
// activation is read in checkpoint K order. That is not Stage A v8's permuted
// order, which the Q4/Q5 kernels need to pick nibbles out of a word; for Q8 it
// would cost a byte gather per step.

#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)

// clang-format off
#include <metal_stdlib>

#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"

#include "oq_a8_decode.h"
// clang-format on

using namespace metal;
using namespace mlx::steel;
using namespace omlx::oq_a8;

constant constexpr int kFragM = 16;
constant constexpr int kFragN = 32;
constant constexpr int kFragK = 16;
constant constexpr int kElemsPerFrag = 8;
constant constexpr int kDestElems = 2 * kElemsPerFrag;
constant constexpr int kStepsPerGroup = kGroupSize / kFragK; // 4
constant constexpr uint32_t kCenter = 0x80808080u;

template <typename T, int ACT_MODE, int WM, int WN>
[[kernel]] void oq_q8_a8_qmm_t_nax(
    const device int8_t* qa [[buffer(0)]],
    const device float* sa [[buffer(1)]],
    const device short* ra [[buffer(2)]],
    const device uint32_t* w [[buffer(3)]],
    const device T* scales [[buffer(4)]],
    const device T* biases [[buffer(5)]],
    device T* out [[buffer(6)]],
    const constant int& K [[buffer(7)]],
    const constant int& N [[buffer(8)]],
    const constant int& M [[buffer(9)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]]) {
  constexpr int TM = 2;
  constexpr int BM = TM * kFragM * WM;
  constexpr int BN = kFragN * WN;
  constexpr int words = oq_group_words(8);

  const int groups = K / kGroupSize;
  const int sg_m = int(simd_gid) % WM;
  const int sg_n = int(simd_gid) / WM;
  const int row_base = int(tid.y) * BM + sg_m * (TM * kFragM);
  const int col_base = int(tid.x) * BN + sg_n * kFragN;

  const short2 coord = BaseNAXFrag::get_coord();

  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
      kFragM,
      kFragN,
      kFragK,
      /* transpose_left = */ false,
      /* transpose_right = */ true,
      /* relaxed_precision = */ false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  constexpr auto desc_set = mpp::tensor_ops::matmul2d_descriptor(
      kFragM,
      kFragN,
      kFragK,
      /* transpose_left = */ false,
      /* transpose_right = */ true,
      /* relaxed_precision = */ false,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply);

  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  mpp::tensor_ops::matmul2d<desc_set, metal::execution_simdgroup> op_set;

  auto ct_a =
      op.template get_left_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto ct_b =
      op.template get_right_input_cooperative_tensor<int8_t, int8_t, int32_t>();
  auto acc0 = op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      int32_t>();
  auto acc1 = op.template get_destination_cooperative_tensor<
      decltype(ct_a),
      decltype(ct_b),
      int32_t>();

  const int n_run0 = col_base + int(coord.x);
  const int m_base = row_base + int(coord.y);

  float Cf[TM][kDestElems];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kDestElems; ++e) {
      Cf[i][e] = 0.0f;
    }
  }

  const int n_lane = col_base + int(coord.y);
  const device uint32_t* wbase = w + size_t(n_lane) * size_t(groups) * words;
  const int w_row = groups * words;
  const int w_stride8 = 8 * w_row;
  const int w_stride16 = kFragM * w_row;

  const int m0_base = row_base + int(coord.y);
  // Which 16-code run of the affine group this lane owns: 0..3.
  const int cx = int(coord.x) >> 2;

  for (int g = 0; g < groups; ++g) {
    // The lane's 16 codes of one group are four words, so one aligned vector
    // load per operand row: group bases are 64-byte aligned.
    uint4 wg[4];
    STEEL_PRAGMA_UNROLL
    for (int q = 0; q < 4; ++q) {
      const device uint32_t* wr = wbase + (q & 1) * w_stride8 +
          (q >> 1) * w_stride16 + size_t(g) * words;
      wg[q] = reinterpret_cast<const device uint4*>(wr)[cx];
    }

    STEEL_PRAGMA_UNROLL
    for (int t = 0; t < kStepsPerGroup; ++t) {
      STEEL_PRAGMA_UNROLL
      for (int q = 0; q < 4; ++q) {
        const int base = (q >> 1) * kElemsPerFrag + (q & 1) * 4;
        const char4 quad = as_type<char4>(wg[q][t] ^ kCenter);
        ct_b[base + 0] = quad.x;
        ct_b[base + 1] = quad.y;
        ct_b[base + 2] = quad.z;
        ct_b[base + 3] = quad.w;
      }

      STEEL_PRAGMA_UNROLL
      for (int hf = 0; hf < 2; ++hf) {
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
          // Rows past M read row M-1; their accumulators are never stored.
          const int m = min(m0_base + hf * kFragM + r * 8, M - 1);
          const char4 quad = as_type<char4>(
              *reinterpret_cast<const device uint32_t*>(
                  qa + size_t(m) * size_t(K) + size_t(g) * kGroupSize +
                  size_t(cx) * 16 + size_t(t) * 4));
          ct_a[r * 4 + 0] = quad.x;
          ct_a[r * 4 + 1] = quad.y;
          ct_a[r * 4 + 2] = quad.z;
          ct_a[r * 4 + 3] = quad.w;
        }
        if (t == 0) {
          if (hf == 0) {
            op_set.run(ct_a, ct_b, acc0);
          } else {
            op_set.run(ct_a, ct_b, acc1);
          }
        } else {
          if (hf == 0) {
            op.run(ct_a, ct_b, acc0);
          } else {
            op.run(ct_a, ct_b, acc1);
          }
        }
      }
    }

    const device T* srow = scales + size_t(g) * size_t(N);
    const device T* brow = biases + size_t(g) * size_t(N);
    const device short* rrow = ra + size_t(g) * size_t(M);

    vec<T, 4> sv[2];
    vec<T, 4> bv[2];
    STEEL_PRAGMA_UNROLL
    for (int h = 0; h < 2; ++h) {
      const int n0 = n_run0 + h * kFragM;
      sv[h] = *reinterpret_cast<const device vec<T, 4>*>(srow + n0);
      bv[h] = *reinterpret_cast<const device vec<T, 4>*>(brow + n0);
    }

    float r_g[TM][2];
    float r_c[TM][2];
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < TM; ++i) {
      STEEL_PRAGMA_UNROLL
      for (int r = 0; r < 2; ++r) {
        const int m = min(m_base + i * kFragM + r * 8, M - 1);
        r_g[i][r] = float(rrow[m]);
        r_c[i][r] = 128.0f * r_g[i][r];
      }
    }

    if (ACT_MODE == 0) {
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kDestElems; ++e) {
        const int r = ((e & 7) >> 2);
        const float swc = float(sv[e >> 3][e & 3]);
        const float bwc = float(bv[e >> 3][e & 3]);
        Cf[0][e] = metal::fma(
            swc,
            float(acc0[e]) + r_c[0][r],
            metal::fma(bwc, r_g[0][r], Cf[0][e]));
        Cf[1][e] = metal::fma(
            swc,
            float(acc1[e]) + r_c[1][r],
            metal::fma(bwc, r_g[1][r], Cf[1][e]));
      }
    } else {
      const device float* arow = sa + size_t(g) * size_t(M);
      float s_g[TM][2];
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < TM; ++i) {
        STEEL_PRAGMA_UNROLL
        for (int r = 0; r < 2; ++r) {
          const int m = min(m_base + i * kFragM + r * 8, M - 1);
          s_g[i][r] = arow[m];
        }
      }
      STEEL_PRAGMA_UNROLL
      for (int e = 0; e < kDestElems; ++e) {
        const int r = ((e & 7) >> 2);
        const float swc = float(sv[e >> 3][e & 3]);
        const float bwc = float(bv[e >> 3][e & 3]);
        Cf[0][e] = metal::fma(
            s_g[0][r],
            metal::fma(swc, float(acc0[e]) + r_c[0][r], bwc * r_g[0][r]),
            Cf[0][e]);
        Cf[1][e] = metal::fma(
            s_g[1][r],
            metal::fma(swc, float(acc1[e]) + r_c[1][r], bwc * r_g[1][r]),
            Cf[1][e]);
      }
    }
  }

  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < TM; ++i) {
    STEEL_PRAGMA_UNROLL
    for (int e = 0; e < kDestElems; ++e) {
      const int ee = e & 7;
      const int r = ee >> 2;
      const int m = row_base + i * kFragM + int(coord.y) + r * 8;
      if (m < M) {
        const int n = col_base + (e >> 3) * kFragM + int(coord.x) + (ee & 3);
        const float v = ACT_MODE == 0 ? sa[m] * Cf[i][e] : Cf[i][e];
        out[size_t(m) * size_t(N) + size_t(n)] = static_cast<T>(v);
      }
    }
  }
}

#define instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, wm, wn)                 \
  instantiate_kernel(                                                         \
      "oq_q8_a8_qmm_t_nax_am" #act_mode "_" #type "_wm_" #wm "_wn_" #wn,      \
      oq_q8_a8_qmm_t_nax,                                                     \
      type,                                                                   \
      act_mode,                                                               \
      wm,                                                                     \
      wn)

// Tile variants index the same table as oq_a8_nax_variant() in
// qwen35_oq_a8.cpp, from the 800 base.
#define instantiate_oq_q8_a8_qmm_t_nax_tiles(act_mode, type)                   \
  instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, 2, 2);                        \
  instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, 4, 2);                        \
  instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, 2, 4);                        \
  instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, 4, 4);                        \
  instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, 1, 4);                        \
  instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, 8, 2);                        \
  instantiate_oq_q8_a8_qmm_t_nax(act_mode, type, 1, 2)

instantiate_oq_q8_a8_qmm_t_nax_tiles(0, float16_t);
instantiate_oq_q8_a8_qmm_t_nax_tiles(0, bfloat16_t);
instantiate_oq_q8_a8_qmm_t_nax_tiles(1, float16_t);
instantiate_oq_q8_a8_qmm_t_nax_tiles(1, bfloat16_t);

#endif // __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
