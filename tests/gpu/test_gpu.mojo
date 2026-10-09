"""GPU kernels and the GPU forward pass, each checked against the CPU path.

Run with scripts/run-gpu-tests.sh, which refuses to report success on a
machine without a GPU. The CPU suite (tests/test_*.mojo) is the oracle here:
every kernel is compared with the CPU function it replaces.
"""

from std.math import sqrt
from std.testing import assert_equal, assert_true, TestSuite

from max.gpu.host import DeviceContext, DeviceBuffer

from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.gpu.kernels import (
    GpuQActs,
    rmsnorm_gpu,
    rope_gpu,
    attention_gpu,
    silu_mul_gpu,
    add_gpu,
    add_bias_gpu,
    quantize_gpu,
    gemm_gpu,
    staged_matrix_bytes,
    dev_f32,
    dev_u8,
    stage_matrix,
    dev_i8,
    dev_i32,
    mma_tile_gpu,
)
from mojo_dllm.gpu.model import GpuDiffusionLM
from mojo_dllm.kernels.ops import (
    rmsnorm,
    rope_table,
    rope_apply,
    attention,
    silu_mul,
    add_inplace,
    add_bias,
)
from mojo_dllm.models.transformer import DiffusionLM
from mojo_dllm.quant.kquants import dequant_row
from mojo_dllm.sys.mem import F32Ptr, I8Ptr, I32Ptr, U8Ptr, Floats, Bytes
from mojo_dllm.sys.mmap import Mapping

comptime FIX = "tests/fixtures/"


def _fill(mut x: Floats, seed: Int):
    var s = UInt64(seed * 2654435761 + 1)
    for i in range(x.n):
        s = s * 6364136223846793005 + 1442695040888963407
        x[i] = Float32(Int((s >> 33) % 2001) - 1000) / 250.0


def _up(ctx: DeviceContext, x: Floats) raises -> DeviceBuffer[DType.float32]:
    var b = ctx.enqueue_create_buffer[DType.float32](x.n)
    ctx.enqueue_copy(b, x.ptr())
    return b


def _down(
    ctx: DeviceContext, b: DeviceBuffer[DType.float32], n: Int
) raises -> Floats:
    var h = Floats(n)
    ctx.enqueue_copy(h.ptr(), b.create_sub_buffer[DType.float32](0, n))
    ctx.synchronize()
    return h^


def _max_abs(a: Floats, b: Floats) -> Float64:
    var worst = Float64(0)
    for i in range(a.n):
        worst = max(worst, abs(Float64(a[i]) - Float64(b[i])))
    return worst


def _rel(got: Floats, want: Floats) -> Float64:
    var num = Float64(0)
    var den = Float64(0)
    for i in range(want.n):
        var d = Float64(got[i]) - Float64(want[i])
        num += d * d
        den += Float64(want[i]) * Float64(want[i])
    return sqrt(num / den)


def test_rmsnorm_matches_cpu() raises:
    var ctx = DeviceContext()
    var rows = 5
    var dim = 384
    var x = Floats(rows * dim)
    var w = Floats(dim)
    _fill(x, 1)
    _fill(w, 2)
    var want = Floats(rows * dim)
    rmsnorm(x.ptr(), w.ptr(), want.ptr(), rows, dim, 1e-5, 1)
    var dx = _up(ctx, x)
    var dw = _up(ctx, w)
    var dy = ctx.enqueue_create_buffer[DType.float32](rows * dim)
    rmsnorm_gpu(ctx, dev_f32(dx), dev_f32(dw), dev_f32(dy), rows, dim, 1e-5)
    var got = _down(ctx, dy, rows * dim)
    assert_true(
        _max_abs(got, want) < 1e-4, "rmsnorm " + String(_max_abs(got, want))
    )


def _check_rope(neox: Bool) raises:
    var ctx = DeviceContext()
    var L = 9
    var heads = 3
    var hd = 64
    var x = Floats(L * heads * hd)
    _fill(x, 3)
    var c = Floats(L * hd // 2)
    var s = Floats(L * hd // 2)
    rope_table(L, hd, 10000.0, 0, c.ptr(), s.ptr())
    var dx = _up(ctx, x)
    var dc = _up(ctx, c)
    var ds = _up(ctx, s)
    rope_apply(x.ptr(), L, heads, hd, c.ptr(), s.ptr(), neox, 1)
    rope_gpu(ctx, dev_f32(dx), L, heads, hd, dev_f32(dc), dev_f32(ds), neox)
    var got = _down(ctx, dx, x.n)
    assert_true(_max_abs(got, x) < 1e-5, "rope " + String(_max_abs(got, x)))


def test_rope_norm_matches_cpu() raises:
    _check_rope(False)


def test_rope_neox_matches_cpu() raises:
    _check_rope(True)


def test_attention_gqa_matches_cpu() raises:
    var ctx = DeviceContext()
    var L = 37
    var heads = 4
    var kvh = 2
    var hd = 128
    var q = Floats(L * heads * hd)
    var k = Floats(L * kvh * hd)
    var v = Floats(L * kvh * hd)
    _fill(q, 4)
    _fill(k, 5)
    _fill(v, 6)
    for i in range(q.n):
        q[i] = q[i] * 0.1
    var want = Floats(L * heads * hd)
    attention(q.ptr(), k.ptr(), v.ptr(), want.ptr(), L, heads, kvh, hd, 2)
    var dq = _up(ctx, q)
    var dk = _up(ctx, k)
    var dv = _up(ctx, v)
    var do_ = ctx.enqueue_create_buffer[DType.float32](want.n)
    attention_gpu(
        ctx,
        dev_f32(dq),
        dev_f32(dk),
        dev_f32(dv),
        dev_f32(do_),
        L,
        heads,
        kvh,
        hd,
    )
    var got = _down(ctx, do_, want.n)
    assert_true(
        _max_abs(got, want) < 1e-4, "attention " + String(_max_abs(got, want))
    )


def test_elementwise_ops_match_cpu() raises:
    var ctx = DeviceContext()
    var rows = 3
    var dim = 1000
    var n = rows * dim
    var g = Floats(n)
    var u = Floats(n)
    var b = Floats(dim)
    _fill(g, 7)
    _fill(u, 8)
    _fill(b, 9)
    var dg = _up(ctx, g)
    var du = _up(ctx, u)
    var db = _up(ctx, b)
    silu_mul(g.ptr(), u.ptr(), n, 1)
    silu_mul_gpu(ctx, dev_f32(dg), dev_f32(du), n)
    add_inplace(g.ptr(), u.ptr(), n)
    add_gpu(ctx, dev_f32(dg), dev_f32(du), n)
    add_bias(g.ptr(), b.ptr(), rows, dim)
    add_bias_gpu(ctx, dev_f32(dg), dev_f32(db), rows, dim)
    var got = _down(ctx, dg, n)
    assert_true(
        _max_abs(got, g) < 1e-4, "elementwise " + String(_max_abs(got, g))
    )


def _check_mma(K: Int) raises:
    var ctx = DeviceContext()
    var ha = Bytes(16 * K)
    var hb = Bytes(8 * K)
    var s = UInt64(12345 + K)
    for i in range(16 * K):
        s = s * 6364136223846793005 + 1442695040888963407
        ha.i8()[unsafe_offset=i] = Int8(Int((s >> 33) % 255) - 127)
    for i in range(8 * K):
        s = s * 6364136223846793005 + 1442695040888963407
        hb.i8()[unsafe_offset=i] = Int8(Int((s >> 33) % 255) - 127)
    var da = ctx.enqueue_create_buffer[DType.int8](16 * K)
    var db = ctx.enqueue_create_buffer[DType.int8](8 * K)
    var dc = ctx.enqueue_create_buffer[DType.int32](128)
    ctx.enqueue_copy(da, ha.i8())
    ctx.enqueue_copy(db, hb.i8())
    mma_tile_gpu(ctx, K, dev_i8(da), dev_i8(db), dev_i32(dc))
    var hc = Bytes(512)
    ctx.enqueue_copy(hc.i32(), dc)
    ctx.synchronize()
    for m in range(16):
        for n in range(8):
            var want = 0
            for k in range(K):
                want += Int(ha.i8()[unsafe_offset=m * K + k]) * Int(
                    hb.i8()[unsafe_offset=n * K + k]
                )
            assert_equal(
                Int(hc.i32()[unsafe_offset=m * 8 + n]),
                want,
                "k" + String(K) + " C[" + String(m) + ", " + String(n) + "]",
            )


def test_mma_k32_fragment_layout() raises:
    _check_mma(32)


def test_mma_k16_fragment_layout() raises:
    _check_mma(16)


def _check_gemm(name: String, n_tokens: Int) raises:
    """GPU int8 GEMM against an f32 GEMM over the reference dequantization."""
    var ctx = DeviceContext()
    var g = GGUFFile(FIX + "tiny-llada.gguf")
    var t = g.tensor(name)
    var K = t.ne0
    var N = t.n_rows()
    var staged = Bytes(staged_matrix_bytes(t.ggml_type, N, K))
    stage_matrix(g.data_addr(t), t.ggml_type, N, K, staged.u8())
    var dw = ctx.enqueue_create_buffer[DType.uint8](staged.size)
    ctx.enqueue_copy(dw, staged.u8())
    var x = Floats(n_tokens * K)
    _fill(x, 11)
    var dx = _up(ctx, x)
    var qa = GpuQActs(ctx, n_tokens, K)
    quantize_gpu(ctx, dev_f32(dx), n_tokens, K, qa)
    var dout = ctx.enqueue_create_buffer[DType.float32](n_tokens * N)
    gemm_gpu(ctx, t.ggml_type, dev_u8(dw), N, K, qa, n_tokens, dev_f32(dout), N)
    var got = _down(ctx, dout, n_tokens * N)
    var want = Floats(n_tokens * N)
    var w = Floats(K)
    for r in range(N):
        dequant_row(t.ggml_type, g.data_addr(t) + r * t.row_bytes(), w.ptr(), K)
        for m in range(n_tokens):
            var acc = Float64(0)
            for k in range(K):
                acc += Float64(w[k]) * Float64(x[m * K + k])
            want[m * N + r] = Float32(acc)
    var rel = _rel(got, want)
    assert_true(rel < 0.01, name + " relative error " + String(rel))


def test_gemm_q4k_ragged_tokens() raises:
    _check_gemm("blk.0.attn_q.weight", 7)


def test_gemm_q4k_many_tokens() raises:
    _check_gemm("blk.0.ffn_gate.weight", 37)


def test_gemm_q6k_ragged_tokens() raises:
    _check_gemm("blk.0.attn_v.weight", 5)


def test_gemm_q6k_two_superblocks() raises:
    _check_gemm("blk.1.ffn_down.weight", 19)


def _check_forward(path: String, vocab: Int) raises:
    # The canvas tests/fixtures/*.expected.f32 was computed for.
    var toks: List[Int] = [5, 17, 300, 511, 511, 42, 511, 7]
    var rows = List[Int]()
    for i in range(len(toks)):
        rows.append(i)
    var cpu = DiffusionLM(FIX + path, threads=2, max_tokens=16)
    var want = Floats(len(toks) * vocab)
    cpu.forward(toks, rows, want.ptr())
    var gpu = GpuDiffusionLM(FIX + path, max_tokens=16)
    var got = Floats(len(toks) * vocab)
    gpu.forward(toks, rows, got.ptr())
    var expected = Mapping(FIX + path.replace(".gguf", ".expected.f32"))
    var ep = F32Ptr(unsafe_from_address=expected.addr)
    var ref_ = Floats(len(toks) * vocab)
    for i in range(ref_.n):
        ref_[i] = ep[unsafe_offset=i]
    var rel_gpu = _rel(got, ref_)
    var rel_cpu = _rel(want, ref_)
    print(
        path,
        "vs f32 reference: gpu",
        rel_gpu,
        "cpu",
        rel_cpu,
        "gpu-vs-cpu",
        _rel(got, want),
    )
    assert_true(
        rel_gpu < 0.02, path + " relative logit error " + String(rel_gpu)
    )
    for r in range(len(toks)):
        var bg = 0
        var bw = 0
        for v in range(vocab):
            if got[r * vocab + v] > got[r * vocab + bg]:
                bg = v
            if want[r * vocab + v] > want[r * vocab + bw]:
                bw = v
        assert_equal(bg, bw, path + " argmax differs at row " + String(r))


def test_forward_llada_matches_cpu() raises:
    _check_forward("tiny-llada.gguf", 512)


def test_forward_dream_matches_cpu() raises:
    _check_forward("tiny-dream.gguf", 512)


def test_forward_subset_rows_match_full() raises:
    var toks: List[Int] = [5, 17, 300, 7, 9, 42, 100, 7]
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var some: List[Int] = [1, 4, 7]
    var m = GpuDiffusionLM(FIX + "tiny-llada.gguf", max_tokens=16)
    var full = Floats(len(toks) * 512)
    m.forward(toks, all_rows, full.ptr())
    var part = Floats(len(some) * 512)
    m.forward(toks, some, part.ptr())
    var worst = Float64(0)
    for j in range(len(some)):
        for v in range(512):
            worst = max(
                worst,
                abs(
                    Float64(part[j * 512 + v])
                    - Float64(full[some[j] * 512 + v])
                ),
            )
    assert_true(worst < 1e-3, "subset rows differ by " + String(worst))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
