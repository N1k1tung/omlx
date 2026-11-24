# SPDX-License-Identifier: Apache-2.0
"""The ANE's fixed evaluation floor, which decides whether it can help a decode step.

A verify forward runs 40 layers of dependent projections; putting the ANE on any of them
serializes one ANE round trip per layer, so the round trip's fixed cost (pack, event
handshake, output copy) is the gate. Measured on an M5 Ultra with this client's fp16
programs: a 64x64 matmul already costs ~0.4-0.65 ms, the ANE needs a sequence width of at
least 32 (so a 6-row verify block pads to 32 rows, 5.3x the work), and the GPU does each
projection 2-4x faster. Run with ``.venv/bin/python tests/check_ane_eval_floor.py``.
"""

import time

import mlx.core as mx
import numpy as np

from omlx.custom_kernels.quant_matmul import ane, fast

# DeepSeek-V4.1-Flash projections a verify block would hand over, as (label, out, in).
SHAPES = (
    ("tiny 64x64", 64, 64),
    ("q_a 5120x1280", 5120, 1280),
    ("gate 5120x2304", 5120, 2304),
    ("o_b 8192x5120", 8192, 5120),
    ("q_b 32768x1280", 32768, 1280),
)


def timeit(call, n=20):
    mx.eval(call())
    start = time.perf_counter()
    for _ in range(n):
        mx.eval(call())
    return (time.perf_counter() - start) / n * 1e3


def rounds(call, n=3):
    return [timeit(call) for _ in range(n)]


def main():
    if not ane.ane_available():
        print("Private ANE runtime is unavailable; nothing to measure.")
        return
    planar = fast._ext.ane_planar
    rng = np.random.default_rng(0)
    print("-- one evaluation, fp16 weights, sequence 32 (a verify block is 6 rows)")
    print(f"{'program':16s} {'compile':>8s} {'ANE round trip':>15s} {'GPU same matmul':>16s}")
    for label, out_d, in_d in SHAPES:
        weight = mx.array(rng.normal(0, 0.02, (out_d, in_d)).astype(np.float32))
        x = mx.array(rng.normal(0, 1, (in_d, 32)).astype(np.float32)).astype(mx.float16)
        mx.eval(x)
        start = time.perf_counter()
        model = ane.ane_compile_fp16_linear(weight, 32)
        compile_ms = (time.perf_counter() - start) * 1e3
        model.warmup()
        mx.eval(planar(x, model))
        ane_ms = min(rounds(lambda x=x, model=model: planar(x, model)))
        gpu_ms = min(rounds(lambda weight=weight, x=x: weight @ x))
        print(
            f"{label:16s} {compile_ms:7.0f}ms {ane_ms * 1000:9.0f} us"
            f"   {gpu_ms * 1000:9.0f} us  ({ane_ms / gpu_ms:.1f}x)"
        )

    print("-- fixed floor breakdown (gate 5120x2304, sequence 32)")
    weight = mx.array(rng.normal(0, 0.02, (5120, 2304)).astype(np.float32))
    x = mx.array(rng.normal(0, 1, (2304, 32)).astype(np.float32)).astype(mx.float16)
    mx.eval(x)
    model = ane.ane_compile_fp16_linear(weight, 32)
    model.warmup()
    mx.eval(planar(x, model))
    ane.ane_profile_reset()
    ane.ane_profile_set_enabled(True)
    for _ in range(20):
        mx.eval(planar(x, model))
    snap = ane.ane_profile_snapshot().get("mlp", {})
    ops = max(snap.get("operations", 0), 1)
    total = timeit(lambda x=x, model=model: planar(x, model))
    pack = snap.get("pack_ns", 0) / 1e3 / ops
    evaluate = snap.get("ane0_eval_ns", 0) / 1e3 / ops
    print(
        f"   ~{total * 1000:.0f} us round trip = pack {pack:.0f} us +"
        f" ANE eval {evaluate:.0f} us + handshake/copy {total * 1000 - pack - evaluate:.0f} us"
    )

    print("-- round trip against sequence width (gate 5120x2304)")
    print(f"{'seq':>5} {'ANE us':>8} {'GPU us':>8} {'ratio':>7}")
    for seq in (32, 64, 128, 256, 512):
        xs = mx.array(rng.normal(0, 1, (2304, seq)).astype(np.float32)).astype(mx.float16)
        mx.eval(xs)
        model = ane.ane_compile_fp16_linear(weight, seq)
        model.warmup()
        mx.eval(planar(xs, model))
        ane_ms = min(rounds(lambda xs=xs, model=model: planar(xs, model), n=2))
        gpu_ms = min(rounds(lambda weight=weight, xs=xs: weight @ xs, n=2))
        print(f"{seq:5d} {ane_ms * 1000:8.0f} {gpu_ms * 1000:8.0f} {ane_ms / gpu_ms:6.1f}x")

    print("-- minimum sequence width")
    for seq in (6, 16, 32):
        try:
            model = ane.ane_compile_fp16_linear(weight, seq)
            xs = mx.array(rng.normal(0, 1, (2304, seq)).astype(np.float32)).astype(mx.float16)
            mx.eval(xs)
            model.warmup()
            mx.eval(planar(xs, model))
            print(f"   seq {seq}: ok")
        except Exception as exc:  # the private framework refuses narrow programs
            print(f"   seq {seq}: {type(exc).__name__}: {str(exc).splitlines()[0][:60]}")


if __name__ == "__main__":
    main()
