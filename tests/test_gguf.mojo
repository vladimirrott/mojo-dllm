from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.formats.gguf import (
    GGUFFile,
    GGML_Q4_K,
    GGML_Q6_K,
    GGML_F16,
    GGML_F32,
)

comptime FIX = "tests/fixtures/"


def test_reads_header_and_scalar_metadata() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    assert_equal(g.version, 3)
    assert_equal(g.architecture(), "test")
    assert_equal(g.get_int("test.u32"), 7)
    assert_equal(g.get_int("test.i32"), -3)
    assert_equal(g.get_f32("test.f32"), 0.25)
    assert_true(g.get_bool("test.bool"))
    assert_equal(g.get_str("test.str"), "héllo")
    assert_true(g.has_key("test.u32"))
    assert_true(not g.has_key("test.nope"))


def test_reads_array_metadata() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    var strs = g.get_str_array("test.strs")
    assert_equal(len(strs), 3)
    assert_equal(strs[2], "déf")
    var ints = g.get_int_array("test.ints")
    assert_equal(len(ints), 3)
    assert_equal(ints[1], -2)


def test_reads_tensor_directory() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    assert_equal(g.n_tensors(), 6)
    var t = g.tensor("q4k")
    assert_equal(t.ggml_type, GGML_Q4_K)
    assert_equal(t.ne0, 512)
    assert_equal(t.ne1, 4)
    assert_equal(t.n_bytes, 4 * 2 * 144)
    assert_equal(g.tensor("q6k").ggml_type, GGML_Q6_K)
    assert_equal(g.tensor("q6k").n_bytes, 4 * 2 * 210)
    assert_equal(g.tensor("f16").ggml_type, GGML_F16)
    assert_equal(g.tensor("q4k.expected").ggml_type, GGML_F32)
    assert_true(g.tensor("q4k").data_offset % 32 == 0)


def test_missing_key_names_the_key() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    with assert_raises(contains="required metadata key missing: test.absent"):
        _ = g.get_int("test.absent")


def test_missing_tensor_names_the_tensor() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    with assert_raises(contains="tensor not found: blk.0.attn_q.weight"):
        _ = g.tensor("blk.0.attn_q.weight")


def test_rejects_bad_magic() raises:
    with assert_raises(contains="not a GGUF file"):
        _ = GGUFFile(FIX + "bad-magic.gguf")


def test_rejects_missing_file() raises:
    with assert_raises(contains="cannot open file"):
        _ = GGUFFile(FIX + "does-not-exist.gguf")


def test_type_names() raises:
    var g = GGUFFile(FIX + "kquants.gguf")
    assert_equal(g.tensor("q4k").type_name(), "Q4_K")
    assert_equal(g.tensor("q6k").type_name(), "Q6_K")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
