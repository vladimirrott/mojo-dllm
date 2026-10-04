from std.math import abs, cos, sin, exp, sqrt
from std.testing import assert_true, assert_equal, TestSuite

from mojo_dllm.kernels.ops import (
    rmsnorm,
    rope_norm,
    attention,
    silu_mul,
    softmax_inplace,
    add_inplace,
    add_bias,
)
from mojo_dllm.sys.mem import Floats


def _close(a: Float32, b: Float32, tol: Float32, what: String) raises:
    assert_true(
        abs(a - b) <= tol, what + ": got " + String(a) + " want " + String(b)
    )


def test_rmsnorm_matches_formula() raises:
    var x = Floats(8)
    var w = Floats(4)
    for i in range(8):
        x[i] = Float32(i + 1)
    for i in range(4):
        w[i] = Float32(0.5) * Float32(i + 1)
    var y = Floats(8)
    rmsnorm(x.ptr(), w.ptr(), y.ptr(), 2, 4, 1e-5, 2)
    for t in range(2):
        var ss = Float32(0)
        for i in range(4):
            ss += x[t * 4 + i] * x[t * 4 + i]
        var inv = 1.0 / sqrt(ss / 4.0 + 1e-5)
        for i in range(4):
            _close(y[t * 4 + i], x[t * 4 + i] * inv * w[i], 1e-5, "rmsnorm")


def test_rope_position_zero_is_identity() raises:
    var x = Floats(2 * 128)
    for i in range(256):
        x[i] = Float32(i % 7) - 3.0
    var y = Floats(256)
    for i in range(256):
        y[i] = x[i]
    rope_norm(y.ptr(), 1, 2, 128, 500000.0, 0)
    for i in range(256):
        _close(y[i], x[i], 1e-6, "rope pos 0")


def test_rope_rotates_adjacent_pairs() raises:
    # Two positions, one head of 128. Position 1, pair i rotates by theta^(-2i/128).
    var x = Floats(2 * 128)
    for i in range(256):
        x[i] = 1.0 if i % 2 == 0 else 0.0
    rope_norm(x.ptr(), 2, 1, 128, 10000.0, 0)
    for i in range(64):
        var ang = 1.0 * (Float64(10000.0) ** (-2.0 * Float64(i) / 128.0))
        _close(x[128 + 2 * i], Float32(cos(ang)), 1e-5, "rope cos")
        _close(x[128 + 2 * i + 1], Float32(sin(ang)), 1e-5, "rope sin")


def test_rope_neox_rotates_half_split_pairs() raises:
    # NEOX pairs element i with element i + d/2.
    var x = Floats(2 * 128)
    for i in range(256):
        x[i] = 1.0 if (i % 128) < 64 else 0.0
    rope_norm(x.ptr(), 2, 1, 128, 10000.0, 0, neox=True)
    for i in range(64):
        var ang = Float64(10000.0) ** (-2.0 * Float64(i) / 128.0)
        _close(x[128 + i], Float32(cos(ang)), 1e-5, "neox cos")
        _close(x[128 + 64 + i], Float32(sin(ang)), 1e-5, "neox sin")


def test_attention_grouped_query_heads_share_kv() raises:
    # 2 query heads read 1 KV head: identical queries give identical outputs.
    var L = 3
    var D = 64
    var q = Floats(L * 2 * D)
    var k = Floats(L * D)
    var v = Floats(L * D)
    for t in range(L):
        for d in range(D):
            v[t * D + d] = Float32(t * 10 + d % 3)
            k[t * D + d] = Float32((t + d) % 5) * 0.1
            q[t * 2 * D + d] = Float32(d % 4) * 0.2
            q[t * 2 * D + D + d] = Float32(d % 4) * 0.2
    var o = Floats(L * 2 * D)
    attention(q.ptr(), k.ptr(), v.ptr(), o.ptr(), L, 2, 1, D, 2)
    for t in range(L):
        for d in range(D):
            _close(
                o[t * 2 * D + d], o[t * 2 * D + D + d], 1e-6, "head 0 == head 1"
            )
    assert_true(o[0] > 0.0, "output is not all zero")


def test_add_bias() raises:
    var x = Floats(6)
    var b = Floats(3)
    for i in range(3):
        b[i] = Float32(i)
    add_bias(x.ptr(), b.ptr(), 2, 3)
    assert_equal(x[4], 1.0)
    assert_equal(x[5], 2.0)


def test_softmax_sums_to_one() raises:
    var x = Floats(5)
    for i in range(5):
        x[i] = Float32(i) * 3.0
    softmax_inplace(x.ptr(), 5)
    var s = Float32(0)
    for i in range(5):
        s += x[i]
    _close(s, 1.0, 1e-6, "softmax sum")
    _close(
        x[4],
        Float32(
            exp(Float64(12.0))
            / (
                1
                + exp(Float64(3))
                + exp(Float64(6))
                + exp(Float64(9))
                + exp(Float64(12))
            )
        ),
        1e-6,
        "softmax max",
    )


def test_attention_is_bidirectional() raises:
    # Query 0 must see key 2 (a causal mask would hide it).
    var L = 3
    var H = 1
    var D = 128
    var q = Floats(L * D)
    var k = Floats(L * D)
    var v = Floats(L * D)
    for t in range(L):
        for d in range(D):
            v[t * D + d] = Float32(t)
    q[0] = 10.0
    k[2 * D + 0] = 10.0
    var o = Floats(L * D)
    attention(q.ptr(), k.ptr(), v.ptr(), o.ptr(), L, H, H, D, 2)
    # score(0,2) = 100/sqrt(128) ~ 8.8, the others 0, so p(2) ~ 0.9997.
    assert_true(o[0] > 1.99, "query 0 attends to future key 2: " + String(o[0]))


def test_attention_uniform_when_scores_equal() raises:
    var L = 4
    var D = 128
    var q = Floats(L * 2 * D)
    var k = Floats(L * 2 * D)
    var v = Floats(L * 2 * D)
    for t in range(L):
        for d in range(2 * D):
            v[t * 2 * D + d] = Float32(t + 1)
    var o = Floats(L * 2 * D)
    attention(q.ptr(), k.ptr(), v.ptr(), o.ptr(), L, 2, 2, D, 2)
    for t in range(L):
        _close(o[t * 2 * D + 5], 2.5, 1e-5, "uniform mean head 0")
        _close(o[t * 2 * D + D + 5], 2.5, 1e-5, "uniform mean head 1")


def test_silu_mul() raises:
    var g = Floats(3)
    var u = Floats(3)
    g[0] = 0.0
    g[1] = 1.0
    g[2] = -2.0
    for i in range(3):
        u[i] = 2.0
    silu_mul(g.ptr(), u.ptr(), 3, 1)
    _close(g[0], 0.0, 1e-6, "silu 0")
    _close(g[1], Float32(2.0 / (1.0 + exp(Float64(-1.0)))), 1e-6, "silu 1")
    _close(
        g[2], Float32(2.0 * -2.0 / (1.0 + exp(Float64(2.0)))), 1e-6, "silu -2"
    )


def test_add_inplace() raises:
    var a = Floats(10)
    var b = Floats(10)
    for i in range(10):
        a[i] = Float32(i)
        b[i] = 1.0
    add_inplace(a.ptr(), b.ptr(), 10)
    assert_equal(a[9], 10.0)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
