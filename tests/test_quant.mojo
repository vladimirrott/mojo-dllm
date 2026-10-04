from std.math import abs, sqrt
from std.testing import assert_equal, assert_true, TestSuite

from mojo_dllm.formats.gguf import GGUFFile, TensorInfo
from mojo_dllm.quant.kquants import dequant_row, f16_to_f32
from mojo_dllm.kernels.qgemm import QActs, quantize_acts, qgemm
from mojo_dllm.kernels.packed import PackedMatrix, pgemm
from mojo_dllm.sys.mem import Floats, F32Ptr

comptime FIX = "tests/fixtures/"


def _expect_rows_match(g: GGUFFile, name: String, expected: String) raises:
    var t = g.tensor(name)
    var e = g.tensor(expected)
    var row = Floats(t.ne0)
    var ep = F32Ptr(unsafe_from_address=g.data_addr(e))
    var worst = Float32(0)
    for r in range(t.n_rows()):
        dequant_row(
            t.ggml_type, g.data_addr(t) + r * t.row_bytes(), row.ptr(), t.ne0
        )
        for i in range(t.ne0):
            var want = ep.unsafe_load(r * t.ne0 + i)
            var diff = abs(row[i] - want)
            if diff > worst:
                worst = diff
    assert_true(worst <= 1e-6, name + " max abs diff " + String(worst))


def test_q4k_dequant_matches_reference() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    _expect_rows_match(g, "q4k", "q4k.expected")


def test_q6k_dequant_matches_reference() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    _expect_rows_match(g, "q6k", "q6k.expected")


def test_f16_dequant_matches_reference() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    _expect_rows_match(g, "f16", "f16.expected")


def test_f16_special_values() raises:
    assert_equal(f16_to_f32(0x3C00), 1.0)
    assert_equal(f16_to_f32(0xC000), -2.0)
    assert_equal(f16_to_f32(0x0000), 0.0)


def _fill(mut x: Floats, seed: Int):
    var s = UInt64(seed * 2654435761 + 1)
    for i in range(x.n):
        s = s * 6364136223846793005 + 1442695040888963407
        x[i] = Float32(Int((s >> 33) % 2001) - 1000) / 250.0


def _check_gemm(name: String, n_tokens: Int) raises:
    """int8-activation GEMM against an f32 GEMM over the same dequantized weights.
    """
    var g = GGUFFile(FIX + "kquants.gguf")
    var t = g.tensor(name)
    var K = t.ne0
    var N = t.n_rows()
    var x = Floats(n_tokens * K)
    _fill(x, 7)
    var qa = QActs(n_tokens, K)
    quantize_acts(x.ptr(), n_tokens, K, qa)
    var out = Floats(n_tokens * N)
    qgemm(g.data_addr(t), t.ggml_type, N, K, qa, n_tokens, out.ptr(), N, 4)
    var w = Floats(K)
    var num = Float64(0)
    var den = Float64(0)
    for n in range(N):
        dequant_row(t.ggml_type, g.data_addr(t) + n * t.row_bytes(), w.ptr(), K)
        for m in range(n_tokens):
            var ref_ = Float64(0)
            for k in range(K):
                ref_ += Float64(w[k]) * Float64(x[m * K + k])
            var got = Float64(out[m * N + n])
            num += (got - ref_) * (got - ref_)
            den += ref_ * ref_
    var rel = sqrt(num / den)
    assert_true(rel < 0.01, name + " relative error " + String(rel))


def test_q4k_gemm_one_token() raises:
    _check_gemm("q4k", 1)


def test_q4k_gemm_ragged_tokens() raises:
    _check_gemm("q4k", 7)


def test_q6k_gemm_ragged_tokens() raises:
    _check_gemm("q6k", 6)


def test_q6k_gemm_many_tokens() raises:
    _check_gemm("q6k", 13)


def test_quantize_acts_roundtrip() raises:
    var K = 512
    var x = Floats(2 * K)
    _fill(x, 3)
    var qa = QActs(2, K)
    quantize_acts(x.ptr(), 2, K, qa)
    var worst = Float32(0)
    for m in range(2):
        for k in range(K):
            var d = qa.d.ptr().unsafe_load(m * (K // 256) + k // 256)
            var q = Float32(qa.q.i8().unsafe_load(m * K + k))
            var diff = abs(q * d - x[m * K + k])
            if diff > worst:
                worst = diff
    # amax is 4.0 so one quantization step is 4/127; error is at most half a step.
    assert_true(
        worst <= 4.0 / 127.0 / 2.0 + 1e-6, "roundtrip error " + String(worst)
    )


def _check_packed(name: String, n_tokens: Int) raises:
    """8-row repacked GEMM against an f32 GEMM over the reference dequantization.
    """
    var g = GGUFFile(FIX + "tiny-llada.gguf")
    var t = g.tensor(name)
    var K = t.ne0
    var N = t.n_rows()
    var pm = PackedMatrix(g.data_addr(t), t.ggml_type, N, K)
    var x = Floats(n_tokens * K)
    _fill(x, 11)
    var qa = QActs(n_tokens, K)
    quantize_acts(x.ptr(), n_tokens, K, qa)
    var out = Floats(n_tokens * N)
    pgemm(pm, qa, n_tokens, out.ptr(), N, 4)
    var w = Floats(K)
    var num = Float64(0)
    var den = Float64(0)
    for n in range(N):
        dequant_row(t.ggml_type, g.data_addr(t) + n * t.row_bytes(), w.ptr(), K)
        for m in range(n_tokens):
            var ref_ = Float64(0)
            for k in range(K):
                ref_ += Float64(w[k]) * Float64(x[m * K + k])
            var got = Float64(out[m * N + n])
            num += (got - ref_) * (got - ref_)
            den += ref_ * ref_
    var rel = sqrt(num / den)
    assert_true(rel < 0.01, name + " relative error " + String(rel))


def test_packed_q4k_ragged_tokens() raises:
    _check_packed("blk.0.attn_q.weight", 7)


def test_packed_q4k_one_token() raises:
    _check_packed("blk.1.ffn_up.weight", 1)


def test_packed_q4k_full_tiles_and_remainder() raises:
    # 17 tokens = two full 8-token Q4_K tiles plus one left over.
    _check_packed("blk.0.ffn_gate.weight", 17)


def test_packed_q6k_ragged_tokens() raises:
    _check_packed("blk.0.attn_v.weight", 5)


def test_packed_q6k_two_superblocks() raises:
    _check_packed("blk.1.ffn_down.weight", 9)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
