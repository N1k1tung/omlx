# SPDX-License-Identifier: Apache-2.0
"""Whether an ANE/GPU split of DeepSeek-V4.1's dense prefill projections can pay.

The upstream prefill path dequantizes each dense GGUF projection (Q8_0 here) to a BF16
copy at load and runs one matmul per chunk, so prefill on this class of Mac is
compute-bound, not bandwidth-bound. An ANE split only pays if the ANE's slice of the
projection costs less than the GPU's, which is what this measures: the GPU's BF16 dense
path against the ANE's INT8 program for a quarter of the output channels, at the chunk
widths a prefill uses. Two client rules bite:

* ``ane_compile_linear`` reads ``weight.data<float>()`` from the array, so the weight
  must be materialized (``mx.eval``) first -- a lazy slice segfaults the process.
* ``ane_affine_qmm_t`` wants the GPU suffix in affine q8/q4/q6 (uint32 packed) with
  scales/biases in the activation dtype, while ``mx.quantize`` returns FP32 metadata.
  And the ANE program's sequence length must equal the chunk's row count exactly, so a
  caller would tile every chunk to the compiled shape.

Run with ``.venv/bin/python tests/check_ane_prefill_split.py``; it prints per-shape
verdicts and never asserts, because the answer is machine-dependent.
"""

import time

import mlx.core as mx
import numpy as np

from omlx.custom_kernels.quant_matmul import ane

# DeepSeek-V4.1-Flash dense projections per layer, as (label, out, in).
SHAPES = (
    ("shared gate 5120x2304", 5120, 2304),
    ("q_b 32768x1280", 32768, 1280),
    ("o_a 4096x8192", 4096, 8192),
    ("o_b 8192x5120", 8192, 5120),
)
SEQUENCES = (1024, 2048)
# The share the ANE would take if the tuner split it evenly with the GPU.
SHARE = 0.25


def timeit(call, n=8):
    mx.eval(call())
    start = time.perf_counter()
    for _ in range(n):
        mx.eval(call())
    return (time.perf_counter() - start) / n * 1e3


def affine_q8(weight):
    """Packed q8 for the GPU suffix, metadata in the activation dtype (client rule)."""
    packed, scales, biases = mx.quantize(mx.contiguous(weight), group_size=64, bits=8)
    scales, biases = scales.astype(mx.bfloat16), biases.astype(mx.bfloat16)
    mx.eval(packed, scales, biases)
    return packed, scales, biases


def main():
    if not ane.ane_available():
        print("Private ANE runtime is unavailable; nothing to measure.")
        return
    rng = np.random.default_rng(0)
    print(f"ANE share {SHARE:.0%} of the output channels; prefill is compute-bound.")
    print(f"{'projection':22s} {'seq':>5} {'GPU bf16':>9} {'ANE slice':>10} {'GPU share':>10}  verdict")
    verdicts = []
    for label, out_d, in_d in SHAPES:
        weight = mx.array(rng.normal(0, 0.02, (out_d, in_d)).astype(np.float32))
        mx.eval(weight)
        cut = int(out_d * SHARE) // 64 * 64
        ane_weight = mx.contiguous(weight[:cut])
        mx.eval(ane_weight)  # the native compiler reads this pointer
        for seq in SEQUENCES:
            model = ane.ane_compile_linear(ane_weight, seq, 0)
            model.warmup()
            x = mx.array(rng.normal(0, 1, (seq, in_d)).astype(np.float32)).astype(mx.bfloat16)
            mx.eval(x)
            gpu_ms = timeit(lambda x=x, weight=weight: x @ weight.T.astype(mx.bfloat16))
            packed, scales, biases = affine_q8(weight[cut:])
            # The ANE slice plus a 64-channel GPU suffix isolates the ANE's own cost.
            tail, tail_s, tail_b = affine_q8(weight[out_d - 64:])
            ane_ms = timeit(
                lambda x=x, tail=tail, tail_s=tail_s, tail_b=tail_b, model=model:
                ane.ane_affine_qmm_t(x, tail, tail_s, tail_b, model, 8, 8, 64)
            )
            share_ms = timeit(
                lambda x=x, packed=packed, scales=scales, biases=biases:
                mx.quantized_matmul(x, packed, scales, biases, group_size=64, bits=8)
            )
            # The ANE would replace GPU work of `share_ms`; it wins only if it is faster.
            gflops = seq * in_d * cut * 2 / 1e9
            ane_rate, gpu_rate = gflops / ane_ms * 1e3, gflops / share_ms * 1e3
            verdict = "ANE wins" if ane_ms < share_ms else f"ANE {ane_ms / share_ms:.1f}x slower"
            verdicts.append(ane_ms < share_ms)
            print(
                f"{label:22s} {seq:5d} {gpu_ms * 1000:8.0f}us {ane_ms * 1000:9.0f}us"
                f" {share_ms * 1000:9.0f}us  {verdict}"
                f" (ANE {ane_rate:.0f} vs GPU {gpu_rate:.0f} GFLOP/s on the slice)"
            )
    print(
        "verdict: "
        + ("the ANE can take a share of the dense projections" if any(verdicts)
           else "keep the GPU's dense path; an ANE split would be slower")
    )


if __name__ == "__main__":
    main()
