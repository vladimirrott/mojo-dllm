from std.math import sqrt
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.models.transformer import DiffusionLM, ModelConfig
from mojo_dllm.sys.mem import F32Ptr, Floats
from mojo_dllm.sys.mmap import Mapping

comptime FIX = "tests/fixtures/"
comptime VOCAB = 512


def _tokens() -> List[Int]:
    return [5, 17, 300, 511, 511, 42, 511, 7]


def test_config_reads_llada_metadata() raises:
    var g = GGUFFile(FIX + "tiny-llada.gguf")
    var c = ModelConfig(g)
    assert_equal(c.n_layers, 2)
    assert_equal(c.hidden, 256)
    assert_equal(c.n_heads, 2)
    assert_equal(c.head_dim, 128)
    assert_equal(c.ffn, 512)
    assert_equal(c.vocab, 512)
    assert_equal(c.mask_id, 511)


def test_forward_matches_numpy_reference() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=4, max_tokens=16)
    var toks = _tokens()
    var rows = List[Int]()
    for i in range(len(toks)):
        rows.append(i)
    var logits = Floats(len(toks) * VOCAB)
    m.forward(toks, rows, logits.ptr())
    var expected = Mapping(FIX + "tiny-llada.expected.f32")
    var ep = F32Ptr(unsafe_from_address=expected.addr)
    var num = Float64(0)
    var den = Float64(0)
    var argmax_agree = 0
    for r in range(len(toks)):
        var best_g = 0
        var best_e = 0
        for v in range(VOCAB):
            var got = Float64(logits[r * VOCAB + v])
            var want = Float64(ep.unsafe_load(r * VOCAB + v))
            num += (got - want) * (got - want)
            den += want * want
            if logits[r * VOCAB + v] > logits[r * VOCAB + best_g]:
                best_g = v
            if ep.unsafe_load(r * VOCAB + v) > ep.unsafe_load(
                r * VOCAB + best_e
            ):
                best_e = v
        if best_g == best_e:
            argmax_agree += 1
    var rel = sqrt(num / den)
    assert_true(rel < 0.02, "relative logit error " + String(rel))
    assert_equal(argmax_agree, len(toks))


def test_forward_subset_rows_equal_full_rows() raises:
    """Asking for logits of a subset of positions returns the same numbers."""
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=4, max_tokens=16)
    var toks = _tokens()
    var all_rows = List[Int]()
    for i in range(len(toks)):
        all_rows.append(i)
    var full = Floats(len(toks) * VOCAB)
    m.forward(toks, all_rows, full.ptr())
    var some: List[Int] = [3, 6]
    var part = Floats(2 * VOCAB)
    m.forward(toks, some, part.ptr())
    for j in range(2):
        for v in range(VOCAB):
            var a = full[some[j] * VOCAB + v]
            var b = part[j * VOCAB + v]
            assert_true(
                abs(a - b) <= 1e-4, "row " + String(some[j]) + " differs"
            )


def test_rejects_too_long_input() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=4)
    var toks = _tokens()
    var rows: List[Int] = [0]
    var out = Floats(VOCAB)
    with assert_raises(contains="exceeds"):
        m.forward(toks, rows, out.ptr())


def test_rejects_wrong_architecture() raises:
    with assert_raises(contains="unsupported model architecture: test"):
        _ = DiffusionLM(FIX + "kquants.gguf", threads=1, max_tokens=4)


def _check_against(
    path: String, expected: String, vocab: Int, tol: Float64
) raises:
    var m = DiffusionLM(FIX + path, threads=4, max_tokens=16)
    var toks = _tokens()
    var rows = List[Int]()
    for i in range(len(toks)):
        rows.append(i)
    var logits = Floats(len(toks) * vocab)
    m.forward(toks, rows, logits.ptr())
    var exp_map = Mapping(FIX + expected)
    var ep = F32Ptr(unsafe_from_address=exp_map.addr)
    var num = Float64(0)
    var den = Float64(0)
    var agree = 0
    for r in range(len(toks)):
        var bg = 0
        var be = 0
        for v in range(vocab):
            var got = Float64(logits[r * vocab + v])
            var want = Float64(ep.unsafe_load(r * vocab + v))
            num += (got - want) * (got - want)
            den += want * want
            if logits[r * vocab + v] > logits[r * vocab + bg]:
                bg = v
            if ep.unsafe_load(r * vocab + v) > ep.unsafe_load(r * vocab + be):
                be = v
        if bg == be:
            agree += 1
    var rel = sqrt(num / den)
    assert_true(rel < tol, path + " relative logit error " + String(rel))
    assert_equal(agree, len(toks))


def test_dream_config_reads_gqa_bias_and_rope_style() raises:
    var g = GGUFFile(FIX + "tiny-dream.gguf")
    var c = ModelConfig(g)
    assert_equal(c.arch, "dream")
    assert_equal(c.n_heads, 4)
    assert_equal(c.n_kv_heads, 2)
    assert_equal(c.head_dim, 64)
    assert_true(c.rope_neox, "dream uses NEOX rope")
    assert_true(c.shift_logits, "dream shifts logits")
    var l = ModelConfig(GGUFFile(FIX + "tiny-llada.gguf"))
    assert_true(not l.rope_neox, "llada uses NORM rope")
    assert_true(not l.shift_logits, "llada does not shift")


def test_dream_forward_matches_numpy_reference() raises:
    # The fixture's Q/K biases are large on purpose (dropping them must fail
    # loudly: 51% error). They also sharpen the attention softmax, which
    # amplifies int8 activation error to about 2.6%, hence 5% here.
    _check_against("tiny-dream.gguf", "tiny-dream.expected.f32", 512, 0.05)


def test_logit_row_follows_the_shift() raises:
    var d = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=8)
    assert_equal(d.logit_row(5), 4)
    var l = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=8)
    assert_equal(l.logit_row(5), 5)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
