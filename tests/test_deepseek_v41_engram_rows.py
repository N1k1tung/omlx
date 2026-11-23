"""Engram wkv: one packed-GGUF weight read for a verify block's rows.

A verify block of six rows used to project the Engram ``wkv`` one row at a time, so a
129 MB Q6_K weight was read six times per Engram layer (two layers per forward).
``gguf_rows_nb`` keeps each row's single-token accumulation order -- same lane stride,
same eight products per block in the same order, one fp32 chain per output row, then
``simd_sum`` -- so the rows kernel must match the per-row calls bit for bit while
reading each block once. These tests pin that equality at the Engram's own shapes and
prove ``Engram.__call__`` takes the shared-read path for a six-row verify block.

Run alone for the exactness checks plus a shared-versus-per-row timing at the real
Q6_K Engram shape: ``python tests/test_deepseek_v41_engram_rows.py``.
"""

import time

import mlx.core as mx
import numpy as np
import pytest

from omlx.patches.deepseek_v41 import engram as engram_mod
from omlx.patches.deepseek_v41.engram import Engram
from omlx.patches.deepseek_v41.gguf import FORMATS, row_bytes
from omlx.patches.deepseek_v41.gguf_quant import GGUFProjection, rows_projection
from tests.test_deepseek_v41 import tiny

pytestmark = pytest.mark.skipif(
    mx.default_device() != mx.gpu, reason="GGUF rows kernels are Metal-only"
)

# GGML type: (offset of the block's fp16 scale, a finite scale value).
_FINITE_SCALE = {8: (0, 0.02), 14: (208, 0.002)}
# The converted checkpoint's Engram wkv, as (output, input): [dim * (hc_mult + 1),
# (ngram - 1) * heads * head_dim] = [25600, 6144], 129 MB packed Q6_K.
_ENGRAM_SHAPE = (25600, 6144)


def packed(qtype, output_dims, input_dims, seed=0):
    """Packed rows of ``qtype``: random codes with each block's fp16 scale made finite,
    so no decoded value is a NaN (random scale bytes would make equality untestable)."""
    size, offset = FORMATS[qtype][1], _FINITE_SCALE[qtype][0]
    rng = np.random.default_rng(seed)
    data = rng.integers(0, 256, (output_dims, row_bytes(input_dims, qtype)), dtype=np.uint8)
    blocks = data.reshape(output_dims, -1, size)
    blocks[..., offset : offset + 2] = np.frombuffer(
        np.float16(_FINITE_SCALE[qtype][1]).tobytes(), np.uint8
    )
    return data


def projection(qtype, output_dims, input_dims, seed=0):
    return GGUFProjection(
        mx.array(packed(qtype, output_dims, input_dims, seed)),
        qtype=qtype,
        input_dims=input_dims,
    )


def rows_input(rows, input_dims, seed=1):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 0.5, (rows, input_dims)).astype(np.float32)
    return mx.array(values).astype(mx.bfloat16)


def bits(a):
    """The raw 16-bit patterns: value equality would accept -0.0 for 0.0."""
    return np.array(a.view(mx.uint16))


def per_row(p, x):
    """The per-row reference: what Engram called before the rows kernel, in input order."""
    rows, input_dims = x.shape
    return mx.concatenate(
        [p(x[r : r + 1].reshape(1, 1, input_dims)) for r in range(rows)], axis=0
    ).reshape(rows, p.output_dims)


@pytest.mark.parametrize("qtype", [8, 14])
@pytest.mark.parametrize("rows", [2, 5, 6, 8, 16])
@pytest.mark.parametrize("output_dims", [64, 6])
def test_rows_projection_matches_per_row(qtype, rows, output_dims):
    input_dims = 512 if qtype == 8 else 2560
    p = projection(qtype, output_dims, input_dims)
    x = rows_input(rows, input_dims)
    shared = rows_projection(p, x)
    assert shared.shape == (rows, output_dims)
    np.testing.assert_array_equal(bits(shared), bits(per_row(p, x)))


def test_rows_projection_matches_per_row_at_the_engram_shape():
    output_dims, input_dims = _ENGRAM_SHAPE
    p = projection(14, output_dims, input_dims, seed=3)
    x = rows_input(6, input_dims, seed=4)
    np.testing.assert_array_equal(bits(rows_projection(p, x)), bits(per_row(p, x)))


@pytest.mark.parametrize("rows", [2, 6, 16])
def test_engram_verify_block_reads_wkv_once(monkeypatch, rows):
    c = tiny(
        dim=64,
        engram_layer_ids=(1,),
        engram_num_embeddings=(97,),
        engram_vocab_size=5,
        engram_max_ngram_size=4,
        engram_n_heads=2,
        engram_head_dim=32,
        engram_compressed_vocab_size=64,
    )
    model = Engram(c, 0)
    rng = np.random.default_rng(7)
    model.embed.weight = mx.array(
        rng.normal(0, 1, model.embed.weight.shape).astype(np.float32)
    ).astype(mx.bfloat16)
    hashes = (c.engram_max_ngram_size - 1) * c.engram_n_heads
    model.wkv = projection(
        8, c.dim * (c.hc_mult + 1), hashes * c.engram_head_dim, seed=11
    )
    ids = rng.integers(0, 97, (1, rows, hashes), dtype=np.int32)
    # Distinct tuples per row: a single request's wide rows would otherwise dedupe.
    ids[0, :, 0] = np.arange(rows)
    h = mx.array(rng.normal(0, 1, (1, rows, c.hc_mult, c.dim)).astype(np.float32))

    with monkeypatch.context() as m:
        m.setattr(engram_mod, "_project_rows", lambda p, flat: None)
        expected = model(h, ids)
    actual = model(h, ids)
    np.testing.assert_array_equal(bits(actual), bits(expected))


def bench(rows=6, repeats=5):
    """The Engram path versus per-row timings for the real wkv shape."""
    p = projection(14, *_ENGRAM_SHAPE, seed=3)
    x = rows_input(rows, _ENGRAM_SHAPE[1], seed=4)

    def timeit(call):
        mx.eval(call())
        start = time.perf_counter()
        for _ in range(repeats):
            mx.eval(call())
        return (time.perf_counter() - start) / repeats

    shared = timeit(lambda: engram_mod._project_rows(p, x))
    alone = timeit(lambda: per_row(p, x))
    bytes_read = p.weight.size
    print(
        f"engram wkv {_ENGRAM_SHAPE[0]}x{_ENGRAM_SHAPE[1]} Q6_K, {rows} rows: "
        f"{bytes_read / 1e6:.0f} MB per read; Engram path {shared * 1e3:.2f} ms, "
        f"per row {alone * 1e3:.2f} ms ({rows}x the weight), "
        f"{(1 - shared / alone) * 100:.0f}% less"
    )
    return shared, alone


def exact():
    """The rows kernel against per-row decode at a small shape and the real one."""
    for qtype, output_dims, input_dims in ((8, 64, 512), (14, 6, 2560)):
        p = projection(qtype, output_dims, input_dims)
        x = rows_input(6, input_dims)
        np.testing.assert_array_equal(bits(rows_projection(p, x)), bits(per_row(p, x)))
    p = projection(14, *_ENGRAM_SHAPE, seed=3)
    x = rows_input(6, _ENGRAM_SHAPE[1], seed=4)
    np.testing.assert_array_equal(bits(rows_projection(p, x)), bits(per_row(p, x)))
    print("rows kernel matches per-row decode bit for bit")


if __name__ == "__main__":
    exact()
    for rows in (2, 6, 16):
        bench(rows=rows, repeats=20)
