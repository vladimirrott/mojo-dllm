from std.math import exp, log
from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from mojo_dllm.diffusion.sampler import (
    GenConfig,
    num_transfer_tokens,
    select_top,
    argmax_and_prob,
    generate,
    Rng,
    _confidence,
)
from mojo_dllm.models.transformer import DiffusionLM
from mojo_dllm.sys.mem import Floats

comptime FIX = "tests/fixtures/"


def test_transfer_schedule_matches_generate_py() raises:
    # LLaDA: base = m // steps, the first (m % steps) steps take one more.
    var s = num_transfer_tokens(32, 8)
    assert_equal(len(s), 8)
    for v in s:
        assert_equal(v, 4)
    var r = num_transfer_tokens(10, 4)
    assert_equal(r[0], 3)
    assert_equal(r[1], 3)
    assert_equal(r[2], 2)
    assert_equal(r[3], 2)
    var total = 0
    for v in num_transfer_tokens(31, 7):
        total += v
    assert_equal(total, 31)


def test_select_top_is_stable_on_ties() raises:
    var conf: List[Float64] = [0.5, 0.9, 0.5, 0.9, 0.1]
    var sel = select_top(conf, 3)
    assert_equal(len(sel), 3)
    assert_equal(sel[0], 1)
    assert_equal(sel[1], 3)
    assert_equal(sel[2], 0)


def test_select_top_caps_at_candidates() raises:
    var conf: List[Float64] = [0.2, 0.3]
    assert_equal(len(select_top(conf, 5)), 2)


def test_argmax_and_prob_in_f64() raises:
    var row = Floats(4)
    row[0] = 1.0
    row[1] = 3.0
    row[2] = 2.0
    row[3] = 3.0
    var r = argmax_and_prob(row.ptr(), 4)
    assert_equal(r[0], 1)  # first maximum wins
    var z = exp(Float64(-2.0)) + 1.0 + exp(Float64(-1.0)) + 1.0
    assert_true(abs(r[1] - 1.0 / z) < 1e-12, "prob " + String(r[1]))


def test_rng_is_deterministic() raises:
    var a = Rng(42)
    var b = Rng(42)
    for _ in range(10):
        assert_equal(a.next_u64(), b.next_u64())
    var r7 = Rng(7)
    var u = r7.uniform()
    assert_true(u > 0.0 and u < 1.0, "uniform in (0,1)")


def _tiny_cfg() -> GenConfig:
    var c = GenConfig()
    c.gen_length = 8
    c.block_length = 4
    c.steps = 4
    c.mask_id = 511
    return c^


def test_generation_resolves_every_mask() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var out = generate(m, prompt, _tiny_cfg())
    assert_equal(len(out.tokens), 11)
    for i in range(3):
        assert_equal(out.tokens[i], prompt[i])
    for i in range(3, 11):
        assert_true(out.tokens[i] != 511, "mask left at position " + String(i))
    assert_equal(out.forward_passes, 4)


def test_generation_is_deterministic_with_temperature() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var c = _tiny_cfg()
    c.temperature = 0.7
    c.seed = 1234
    var a = generate(m, prompt, c)
    var b = generate(m, prompt, c)
    for i in range(len(a.tokens)):
        assert_equal(a.tokens[i], b.tokens[i])


def test_full_logits_ablation_gives_same_tokens() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var a = generate(m, prompt, _tiny_cfg())
    var c = _tiny_cfg()
    c.full_logits = True
    var b = generate(m, prompt, c)
    for i in range(len(a.tokens)):
        assert_equal(a.tokens[i], b.tokens[i])


def test_rejects_bad_block_geometry() raises:
    var m = DiffusionLM(FIX + "tiny-llada.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5]
    var c = _tiny_cfg()
    c.block_length = 3
    with assert_raises(contains="multiple of the block length"):
        _ = generate(m, prompt, c)
    c = _tiny_cfg()
    c.steps = 3
    with assert_raises(contains="multiple of the number of blocks"):
        _ = generate(m, prompt, c)


def test_dream_confidences() raises:
    var row = Floats(3)
    row[0] = 0.0
    row[1] = 0.0
    row[2] = 0.0
    var e = _confidence(row.ptr(), 3, "entropy")
    assert_equal(e[0], 0)
    # uniform over 3: sum p log p = log(1/3)
    assert_true(
        abs(e[1] - log(Float64(1.0) / 3.0)) < 1e-9,
        "neg entropy " + String(e[1]),
    )
    row[2] = 2.0
    var m = _confidence(row.ptr(), 3, "maskgit_plus")
    assert_equal(m[0], 2)
    var z = 2.0 + exp(Float64(2.0))
    assert_true(abs(m[1] - exp(Float64(2.0)) / z) < 1e-12, "p(x0)")
    var k = _confidence(row.ptr(), 3, "topk_margin")
    assert_true(abs(k[1] - (exp(Float64(2.0)) - 1.0) / z) < 1e-12, "margin")


def test_dream_generation_resolves_every_mask() raises:
    var m = DiffusionLM(FIX + "tiny-dream.gguf", threads=2, max_tokens=16)
    var prompt: List[Int] = [5, 17, 300]
    var c = GenConfig()
    c.gen_length = 8
    c.steps = 4
    c.mask_id = 510
    var out = generate(m, prompt, c)
    assert_equal(len(out.tokens), 11)
    for i in range(3, 11):
        assert_true(out.tokens[i] != 510, "mask left at position " + String(i))
    assert_true(out.forward_passes <= 4, "at most one pass per step")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
