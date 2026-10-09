"""Command line for mojo-dllm.

    mojo-dllm run      --model M --prompt TEXT [--max-tokens N] [--steps N] [--block-length N]
                       [--seed N] [--temperature T] [--threads N] [--device cpu|gpu]
                       [--remasking low_confidence|random]
                       [--no-chat] [--full-logits] [--visual] [--verbose] [--dump-step-stats] [--json]
    mojo-dllm tokenize --model M --prompt TEXT [--no-chat]
    mojo-dllm inspect  MODEL.gguf
    mojo-dllm logits   --model M --tokens 1,2,3 --rows 0,2 --out logits.f32 [--threads N] [--device cpu|gpu]
"""

from std.ffi import external_call, c_int
from std.sys import argv, exit, num_logical_cores
from std.time import perf_counter_ns

from cli.args import Args, parse_csv_ints
from mojo_dllm.formats.gguf import GGUFFile, GGUF_TYPE_ARRAY
from mojo_dllm.diffusion.sampler import (
    GenConfig,
    StepObserver,
    generate_observed,
)
from mojo_dllm.gpu.model import GpuDiffusionLM
from mojo_dllm.models.transformer import DenoisingModel, DiffusionLM
from mojo_dllm.sys.mem import Bytes, Floats
from mojo_dllm.tokenizer.bpe import Tokenizer

comptime VERSION = "0.1.0"


def usage() -> String:
    return (
        "mojo-dllm "
        + VERSION
        + ": diffusion language model inference in Mojo\n\n"
        + "usage:\n"
        + "  mojo-dllm run --model M --prompt TEXT [--max-tokens 128] [--steps"
        " 128] [--block-length 32]\n"
        + "                [--seed 42] [--temperature 0] [--threads N]"
        " [--device cpu|gpu]\n"
        + "                [--remasking low_confidence|random]\n"
        + "                [--no-chat] [--full-logits] [--visual] [--verbose]"
        " [--dump-step-stats] [--json]\n"
        + "  mojo-dllm tokenize --model M --prompt TEXT [--no-chat]\n"
        + "  mojo-dllm inspect MODEL.gguf\n"
        + "  mojo-dllm logits --model M --tokens IDS --rows POS --out FILE"
        " [--threads N] [--device cpu|gpu]\n"
    )


def peak_rss_bytes() raises -> Int:
    """Peak resident set size in bytes (getrusage ru_maxrss, which Linux reports in KiB).
    """
    var buf = Bytes(256)
    _ = external_call["getrusage", c_int](c_int(0), buf.addr)
    # struct rusage: two struct timeval (16 bytes each), then long ru_maxrss.
    return Int(buf.u8().unsafe_bitcast[Int64]().unsafe_load(4)) * 1024


def chat_prompt(arch: String, text: String) -> String:
    """One user turn in the model's chat template (its tokenizer.chat_template).

    LLaDA-8B-Instruct: Llama-3 style headers after <|startoftext|>.
    Dream-v0-Instruct: Qwen2 ChatML with the default system message.
    """
    if arch == "dream":
        return (
            "<|im_start|>system\nYou are a helpful"
            " assistant.<|im_end|>\n<|im_start|>user\n"
            + String(text.strip())
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
    return (
        "<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n"
        + String(text.strip())
        + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def json_str(s: String) -> String:
    """A JSON string literal. Multi-byte UTF-8 passes through unchanged."""
    var out = List[UInt8]()
    out.append(34)
    var hexd = "0123456789abcdef".as_bytes()
    for b in s.as_bytes():
        var c = Int(b)
        if c == 34 or c == 92:
            out.append(92)
            out.append(b)
        elif c == 10:
            out.append(92)
            out.append(110)
        elif c == 13:
            out.append(92)
            out.append(114)
        elif c == 9:
            out.append(92)
            out.append(116)
        elif c < 32:
            for q in "\\u00".as_bytes():
                out.append(q)
            out.append(hexd[c // 16])
            out.append(hexd[c % 16])
        else:
            out.append(b)
    out.append(34)
    return String(StringSpan(unsafe_from_utf8=Span(out)))


def encode_prompt(
    mut tok: Tokenizer, arch: String, a: Args
) raises -> List[Int]:
    var text = a.get("prompt")
    if a.has("no-chat"):
        return tok.encode(text, add_bos=tok.add_bos)
    return tok.encode(chat_prompt(arch, text))


struct CanvasPrinter(StepObserver):
    """Redraws the canvas after every step when --visual is on.

    Masked positions are teal blocks, tokens committed in this step are bold
    ember, earlier tokens plain. It owns the tokenizer so it can decode as it
    draws; `cmd_run` decodes the final text through it too.
    """

    var tok: Tokenizer
    var start: Int
    var mask_id: Int
    var enabled: Bool

    def __init__(
        out self, var tok: Tokenizer, start: Int, mask_id: Int, enabled: Bool
    ):
        self.tok = tok^
        self.start = start
        self.mask_id = mask_id
        self.enabled = enabled

    def on_step(
        mut self, tokens: List[Int], committed: List[Int], step: Int, total: Int
    ) raises:
        if not self.enabled:
            return
        var esc = chr(27)
        var out = esc + "[2J" + esc + "[H"
        out += (
            esc
            + "[38;5;80mmojo-dllm"
            + esc
            + "[0m  denoising step "
            + String(step)
            + " / "
            + String(total)
            + "\n\n"
        )
        for p in range(self.start, len(tokens)):
            var t = tokens[p]
            if t == self.mask_id:
                out += esc + "[38;5;30m\u2591" + esc + "[0m"
                continue
            if t == self.tok.eos_id or t == self.tok.eot_id:
                out += esc + "[38;5;240m\u00b7" + esc + "[0m"
                continue
            var piece = self.tok.decode([t])
            var fresh = False
            for c in committed:
                if c == p:
                    fresh = True
            if fresh:
                out += esc + "[1;38;5;208m" + piece + esc + "[0m"
            else:
                out += piece
        print(out + "\n", flush=True)


def cmd_tokenize(a: Args) raises:
    var g = GGUFFile(a.get("model"))
    var tok = Tokenizer(g)
    var ids = encode_prompt(tok, g.architecture(), a)
    var s = String("")
    for i in range(len(ids)):
        s += (", " if i else "") + String(ids[i])
    print("[" + s + "]")


def cmd_run(a: Args) raises:
    var threads = a.int_or("threads", num_logical_cores())
    var cfg = GenConfig()
    cfg.gen_length = a.int_or("max-tokens", 128)
    cfg.steps = a.int_or("steps", cfg.gen_length)
    cfg.block_length = a.int_or("block-length", min(32, cfg.gen_length))
    cfg.seed = a.int_or("seed", 42)
    cfg.temperature = a.float_or("temperature", 0.0)
    cfg.remasking = a.get_or("remasking", "low_confidence")
    cfg.full_logits = a.has("full-logits")
    cfg.threads = threads
    var t0 = perf_counter_ns()
    var g = GGUFFile(a.get("model"))
    var tok = Tokenizer(g)
    var prompt = encode_prompt(tok, g.architecture(), a)
    _ = g^
    var tok_s = Float64(perf_counter_ns() - t0) / 1e9
    var device = a.get_or("device", "cpu")
    var max_tokens = len(prompt) + cfg.gen_length
    if device == "gpu":
        var gm = GpuDiffusionLM(a.get("model"), max_tokens=max_tokens)
        _run(
            gm,
            a,
            cfg^,
            prompt,
            tok^,
            tok_s,
            device,
            "gpu (" + gm.ctx.name() + ")",
            threads,
        )
    elif device == "cpu":
        var cm = DiffusionLM(
            a.get("model"), threads=threads, max_tokens=max_tokens
        )
        _run(
            cm,
            a,
            cfg^,
            prompt,
            tok^,
            tok_s,
            device,
            "cpu   threads: " + String(threads),
            threads,
        )
    else:
        raise Error("unknown device: " + device + " (expected cpu or gpu)")


def _run[
    M: DenoisingModel
](
    mut model: M,
    a: Args,
    var cfg: GenConfig,
    prompt: List[Int],
    var tok: Tokenizer,
    tok_s: Float64,
    device: String,
    device_line: String,
    threads: Int,
) raises:
    var verbose = a.has("verbose")
    cfg.mask_id = model.config().mask_id
    if verbose:
        print(
            "model:",
            model.model_name(),
            "(" + model.config().arch + ")",
        )
        print("device:", device_line)
        print(
            "prompt tokens:",
            len(prompt),
            "  generation tokens:",
            cfg.gen_length,
        )
        if model.config().arch == "dream":
            print(
                "diffusion steps:",
                cfg.steps,
                "  schedule: Dream timesteps, eps",
                cfg.eps,
            )
        else:
            print(
                "diffusion steps:",
                cfg.steps,
                "  block length:",
                cfg.block_length,
                "  remasking:",
                cfg.remasking,
            )
        print("load time:", model.load_time(), "s (tokenizer", tok_s, "s)")
    var printer = CanvasPrinter(
        tok^, len(prompt), model.config().mask_id, a.has("visual")
    )
    var res = generate_observed(model, prompt, cfg, printer)
    var gen = List[Int]()
    for i in range(len(prompt), len(res.tokens)):
        var t = res.tokens[i]
        if t == printer.tok.eos_id or t == printer.tok.eot_id:
            break
        gen.append(t)
    var text = printer.tok.decode(gen)
    var tm = model.stats()
    var fw = Float64(max(tm.forwards, 1))
    if a.has("dump-step-stats"):
        for i in range(len(res.step_ms)):
            print("step", i + 1, ":", res.step_ms[i], "ms")
    if a.has("json"):
        var steps_json = String("[")
        for i in range(len(res.step_ms)):
            steps_json += (", " if i else "") + String(res.step_ms[i])
        steps_json += "]"
        print(
            "{"
            + '"runtime": "mojo-dllm", "version": "'
            + VERSION
            + '", "arch": "'
            + model.config().arch
            + '", "device": "'
            + device
            + '"'
            + ', "threads": '
            + String(threads)
            + ', "prompt_tokens": '
            + String(len(prompt))
            + ', "gen_length": '
            + String(cfg.gen_length)
            + ', "steps": '
            + String(cfg.steps)
            + ', "block_length": '
            + String(cfg.block_length)
            + ', "full_logits": '
            + ("true" if cfg.full_logits else "false")
            + ', "forward_passes": '
            + String(res.forward_passes)
            + ', "load_s": '
            + String(model.load_time())
            + ', "tokenizer_s": '
            + String(tok_s)
            + ', "generate_s": '
            + String(res.seconds)
            + ', "ms_per_step": '
            + String(res.seconds * 1000.0 / Float64(max(res.forward_passes, 1)))
            + ', "forward_ms": '
            + String(Float64(tm.total_ns) / 1e6 / fw)
            + ', "gemm_ms_per_forward": '
            + String(Float64(tm.gemm_ns) / 1e6 / fw)
            + ', "attention_ms_per_forward": '
            + String(Float64(tm.attn_ns) / 1e6 / fw)
            + ', "lm_head_ms_per_forward": '
            + String(Float64(tm.head_ns) / 1e6 / fw)
            + ', "peak_rss_bytes": '
            + String(peak_rss_bytes())
            + ', "step_ms": '
            + steps_json
            + ', "tokens": ['
            + _csv(res.tokens)
            + "]"
            + ', "text": '
            + json_str(text)
            + "}"
        )
        return
    print(text)
    if verbose:
        print()
        print(
            "generation:",
            res.seconds,
            "s for",
            res.forward_passes,
            "denoising steps",
        )
        print(
            "per step:",
            res.seconds * 1000.0 / Float64(max(res.forward_passes, 1)),
            "ms   (forward pass",
            Float64(tm.total_ns) / 1e6 / fw,
            "ms)",
        )
        print(
            "  gemm",
            Float64(tm.gemm_ns) / 1e6 / fw,
            "ms   attention",
            Float64(tm.attn_ns) / 1e6 / fw,
            "ms   lm head",
            Float64(tm.head_ns) / 1e6 / fw,
            "ms",
        )
        print("tokens/s:", Float64(cfg.gen_length) / res.seconds)
        print("peak RSS:", Float64(peak_rss_bytes()) / 1e9, "GB")


def _csv(xs: List[Int]) -> String:
    var s = String("")
    for i in range(len(xs)):
        s += ("," if i else "") + String(xs[i])
    return s


def cmd_inspect(a: Args) raises:
    if len(a.positional) < 1:
        raise Error("inspect needs a model path")
    var g = GGUFFile(a.positional[0])
    print("file:", a.positional[0])
    print("gguf version:", g.version)
    print("architecture:", g.get_str_or("general.architecture", "?"))
    print("name:", g.get_str_or("general.name", "?"))
    print("tensors:", g.n_tensors())
    print("metadata keys:", len(g.kvs))
    print()
    print("metadata:")
    for i in range(len(g.kvs)):
        print("  ", g.kvs[i].key, " = ", g.value_str(i), sep="")
    var counts = Dict[String, Int]()
    for i in range(g.n_tensors()):
        var tn = g.tensors[i].type_name()
        counts[tn] = counts.get(tn, 0) + 1
    print()
    print("tensor types:")
    for e in counts.items():
        print("  ", e.key, ": ", e.value, sep="")
    print()
    print("tensors:")
    for i in range(g.n_tensors()):
        var t = g.tensors[i]
        print("  ", t.name, "  ", t.type_name(), "  ", t.shape_str(), sep="")


def cmd_logits(a: Args) raises:
    var threads = a.int_or("threads", num_logical_cores())
    var tokens = parse_csv_ints(a.get("tokens"))
    var rows = parse_csv_ints(a.get("rows"))
    var device = a.get_or("device", "cpu")
    if device == "gpu":
        var gm = GpuDiffusionLM(a.get("model"), max_tokens=len(tokens))
        _logits(gm, tokens, rows, a.get("out"))
    elif device == "cpu":
        var cm = DiffusionLM(
            a.get("model"), threads=threads, max_tokens=len(tokens)
        )
        _logits(cm, tokens, rows, a.get("out"))
    else:
        raise Error("unknown device: " + device + " (expected cpu or gpu)")


def _logits[
    M: DenoisingModel
](mut m: M, tokens: List[Int], rows: List[Int], path: String) raises:
    var vocab = m.config().vocab
    var out = Floats(len(rows) * vocab)
    m.forward(tokens, rows, out.ptr())
    with open(path, "w") as f:
        var p = out.ptr().unsafe_bitcast[UInt8]()
        f.write_bytes(Span(unsafe_ptr=p, length=len(rows) * vocab * 4))
    print(
        "wrote",
        len(rows),
        "rows x",
        vocab,
        "logits; load",
        m.load_time(),
        "s",
    )


def main():
    var raw = List[String]()
    for s in argv():
        raw.append(String(s))
    try:
        var a = Args(
            raw,
            [
                "verbose",
                "full-logits",
                "no-chat",
                "dump-step-stats",
                "json",
                "visual",
            ],
        )
        if a.command == "run":
            cmd_run(a)
        elif a.command == "tokenize":
            cmd_tokenize(a)
        elif a.command == "inspect":
            cmd_inspect(a)
        elif a.command == "logits":
            cmd_logits(a)
        elif a.command == "" or a.command == "help" or a.command == "--help":
            print(usage())
        else:
            raise Error("unknown command: " + a.command)
    except e:
        print("ERROR:", e)
        exit(1)
