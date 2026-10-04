from std.testing import assert_equal, assert_true, assert_raises, TestSuite

from cli.args import parse_csv_ints
from mojo_dllm.formats.gguf import GGUFFile
from mojo_dllm.tokenizer.bpe import (
    Tokenizer,
    UnicodeClasses,
    pretokenize,
    utf8_decode,
    C_LETTER,
    C_NUMBER,
    C_SPACE,
    C_OTHER,
)

comptime FIX = "tests/fixtures/"


def _hex_to_string(h: String) raises -> String:
    var b = h.as_bytes()
    var out = List[UInt8]()
    for i in range(0, len(b), 2):
        var hi = Int(b[i])
        var lo = Int(b[i + 1])
        hi = hi - 48 if hi <= 57 else hi - 87
        lo = lo - 48 if lo <= 57 else lo - 87
        out.append(UInt8(hi * 16 + lo))
    return String(StringSpan(unsafe_from_utf8=Span(out)))


def _cases(prefix: String) raises -> List[Tuple[String, List[Int]]]:
    var out = List[Tuple[String, List[Int]]]()
    with open(FIX + prefix + "-tokenizer-golden.tsv", "r") as f:
        var text = f.read()
        for line in text.split("\n"):
            var l = String(line)
            if l.byte_length() == 0:
                continue
            var parts = l.split("\t")
            out.append(
                (
                    _hex_to_string(String(parts[0])),
                    parse_csv_ints(String(parts[1])),
                )
            )
    return out^


def _check_golden(prefix: String) raises:
    var g = GGUFFile(FIX + prefix + "-tokenizer.gguf")
    var tok = Tokenizer(g)
    var cases = _cases(prefix)
    assert_true(
        len(cases) >= 30,
        "golden corpus is unexpectedly small: " + String(len(cases)),
    )
    var failures = 0
    for c in cases:
        var got = tok.encode(c[0])
        var want = c[1].copy()
        var same = len(got) == len(want)
        if same:
            for i in range(len(got)):
                if got[i] != want[i]:
                    same = False
        if not same:
            failures += 1
            var gs = String("")
            for t in got:
                gs += String(t) + ","
            print("MISMATCH", prefix, repr(c[0]), "got", gs)
        assert_equal(tok.decode(got, skip_special=False), c[0])
    assert_equal(failures, 0)


def test_llada_golden_ids_match_llama_cpp() raises:
    _check_golden("llada")


def test_dream_golden_ids_match_llama_cpp() raises:
    _check_golden("dream")


def test_dream_special_ids() raises:
    var tok = Tokenizer(GGUFFile(FIX + "dream-tokenizer.gguf"))
    assert_equal(tok.eos_id, 151643)
    assert_equal(tok.mask_id, 151666)
    assert_equal(tok.eot_id, tok.encode("<|im_end|>")[0])


def test_decode_skips_special_tokens_by_default() raises:
    var g = GGUFFile(FIX + "llada-tokenizer.gguf")
    var tok = Tokenizer(g)
    var ids = tok.encode("<|eot_id|>Hi")
    assert_equal(ids[0], 126348)
    assert_equal(tok.decode(ids), "Hi")


def test_special_ids() raises:
    var g = GGUFFile(FIX + "llada-tokenizer.gguf")
    var tok = Tokenizer(g)
    assert_equal(tok.bos_id, 126080)
    assert_equal(tok.eos_id, 126081)
    assert_equal(tok.eot_id, 126348)
    assert_equal(tok.mask_id, 126336)
    assert_equal(tok.vocab_size(), 126464)


def test_classify() raises:
    var uc = UnicodeClasses()
    assert_equal(uc.classify(65), C_LETTER)
    assert_equal(uc.classify(0x4E2D), C_LETTER)  # 中
    assert_equal(uc.classify(0x0661), C_NUMBER)  # Arabic-Indic one
    assert_equal(uc.classify(0x00A0), C_SPACE)
    assert_equal(
        uc.classify(0x0301), C_OTHER
    )  # combining acute: not a letter in llama.cpp's tables


def test_pretokenize_newline_runs() raises:
    # "\s*[\r\n]" ends at the last newline of a whitespace run.
    var cps = utf8_decode("x  \n  y")
    var p = pretokenize(cps, UnicodeClasses())
    assert_equal(len(p), 4)
    assert_equal(p[1][0], 1)
    assert_equal(p[1][1], 4)


def test_rejects_other_tokenizers() raises:
    var g = GGUFFile(FIX + "tiny-llada.gguf")
    with assert_raises(contains="tokenizer.ggml.model"):
        _ = Tokenizer(g)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
