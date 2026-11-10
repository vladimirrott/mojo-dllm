#!/usr/bin/env python3
"""Logit and generation parity between mojo-dllm and llama.cpp on a real model.

usage: python3 bench/parity.py --model MODEL.gguf [--ref build/llada_ref] [--threads 12] [--device cpu|gpu]
                              [--ref2 build/llada_ref_cuda --ref2-label "llama.cpp CUDA"]

--ref2 adds a second reference for the logits only: the result then also
records mojo-dllm against it and the two references against each other, which
shows how far two builds of llama.cpp disagree on the same model.

Writes bench/parity/<date>-<arch>-<model-stem>[-gpu].json. scripts/bench_table.py renders it into
docs/correctness.md. Requires build/mojo-dllm and build/llada_ref
(see docs/correctness.md for how to build the latter).

DiffusionGemma (--arch diffusion-gemma) runs in two phases that never overlap, because the 16.81 GB file
is mapped by whichever process runs, and mojo-dllm keeps its own repacked copy beside it:

  --phase ref      the llama.cpp #24423 branch (scripts/build_dg_ref.sh) writes the canvas logits of a
                   fixed prompt and canvas, four ways, into build/dg-parity/<model>/caseN/.
  --phase compare  reads those files and mojo-dllm's, written by mojo-dllm itself as <variant>.mojo.f32
                   beside them, and writes bench/parity/ evidence only when every variant is inside the
                   fixed-input bound measured from DiffusionGemma's own reference controls. A missing output
                   or execution receipt stops the run: there is
                   nothing to compare until the forward pass (#64) exists, and the reference's own output
                   is never used in its place.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import pathlib
import random
import re
import subprocess
import tempfile
from typing import TYPE_CHECKING, Callable, NamedTuple

if TYPE_CHECKING:
    import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
SPECS = {"llada": dict(vocab=126464, mask=126336), "dream": dict(vocab=152064, mask=151666)}

class LogitCase(NamedTuple):
    prompt: str
    masks: int


class GenCase(NamedTuple):
    prompt: str
    gen: int
    block: int
    steps: int


LOGIT_CASES = [
    LogitCase("Explain speculative decoding in two sentences.", 44),
    LogitCase("List three differences between diffusion language models and autoregressive language models.", 100),
]
GEN_CASES = [
    GenCase("Explain speculative decoding in two sentences.", 32, 32, 32),
    GenCase("Write a short Python function that checks whether a number is prime.", 64, 32, 32),
]


def run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"command failed ({p.returncode}): {' '.join(cmd[:3])} ...\n{p.stderr[-800:]}")
    return p.stdout


def tokenize(model: str, prompt: str) -> list[int]:
    out = run([str(ROOT / "build/mojo-dllm"), "tokenize", "--model", model, "--prompt", prompt])
    return [int(x) for x in out.strip().splitlines()[-1].strip("[]").split(",")]


def result_path(model: str, arch: str, device: str, day: dt.date) -> pathlib.Path:
    """Different checkpoints of one architecture must not overwrite each other."""
    suffix = "-gpu" if device == "gpu" else ""
    return ROOT / "bench" / "parity" / f"{day.isoformat()}-{arch}-{pathlib.Path(model).stem}{suffix}.json"


def compare(a: np.ndarray, b: np.ndarray, first_mask: int) -> dict:
    import numpy as np

    cos = np.array([float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))) for x, y in zip(a, b)])
    rel = np.array([float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(y**2))) for x, y in zip(a, b)])
    top5 = [len(set(np.argsort(-x)[:5]) & set(np.argsort(-y)[:5])) for x, y in zip(a, b)]
    agree = int(sum(int(np.argmax(x) == np.argmax(y)) for x, y in zip(a, b)))
    worst = np.argsort(cos)[:3]
    return dict(rows=len(cos), min_cosine=float(cos.min()), median_cosine=float(np.median(cos)),
                masked_median_cosine=float(np.median(cos[first_mask:])),
                max_rel_rms=float(rel.max()), median_rel_rms=float(np.median(rel)), argmax_agree=agree,
                mean_top5_overlap=sum(top5) / len(top5), worst_rows=[int(i) for i in worst])


# DiffusionGemma ---------------------------------------------------------------------------------------

DG_ARCH = "diffusion-gemma"
# The canvas is random token ids (the model starts from noise, not masks), fixed here so that both
# runtimes see the same one. Seeds are the issue number plus the case index.
DG_CANVAS_SEED = 66
# The four forward passes of one canvas. The cached ones run the prompt through a prefill first and then
# the canvas against the stored K/V; the -sc ones add self-conditioning from the previous step's logits.
DG_VARIANTS = ("unified", "cached", "unified-sc", "cached-sc")
DG_PROMPTS = (
    "Explain speculative decoding in two sentences.",
    "List three differences between diffusion language models and autoregressive language models.",
    "Write a short Python function that checks whether a number is prime.",
    "Translate into English: 春天来了，河边的树开花了。Then explain the image in one sentence.",
)
DG_SENSITIVITY_EVIDENCE = "bench/checks/2026-11-05-dg-sensitivity.json"
# The -sc variants condition on the reference's own unified logits (the "previous step"), which both
# runtimes read, so neither side's own earlier step enters the comparison.
DG_PREV_FILE = "unified.ref.f32"
# The first denoising step runs at temperature 0.8 (issue #60), so the second step's self-conditioning
# divides the previous logits by it.
DG_PREV_TEMP_INV = 1.0 / 0.8
LLADA_EVIDENCE = "bench/parity/2026-10-03-llada.json"
# A changed-input sensitivity diagnostic. Its spread is recorded beside parity;
# only same-input cache-path controls determine the implementation tolerance.
DG_NOISE_REL = 1e-6
DG_NOISE_SEED = 66
DG_NOISY_FILE = "unified-sc-noisy.ref.f32"


def splitmix64_stream(seed: int):
    """Steele et al.'s SplitMix64: a generator small enough to port to Mojo unchanged."""
    mask = (1 << 64) - 1
    state = seed & mask
    while True:
        state = (state + 0x9E3779B97F4A7C15) & mask
        z = state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
        yield z ^ (z >> 31)


def dg_canvas(vocab: int, length: int, seed: int) -> list[int]:
    stream = splitmix64_stream(seed)
    return [next(stream) % vocab for _ in range(length)]


def parse_token_ids(out: str) -> list[int]:
    """The ids on the last line of `llama-tokenize --ids`, which looks like `[2, 1234, 5]`."""
    lines = [line.strip() for line in out.strip().splitlines() if line.strip()]
    if not lines or not re.fullmatch(r"\[\s*\d+(\s*,\s*\d+)*\s*\]", lines[-1]):
        raise SystemExit(f"refusing: llama-tokenize printed no token ids; last line: {lines[-1:] or 'none'}")
    return [int(x) for x in re.findall(r"\d+", lines[-1])]


def dg_pinned_commit(root: pathlib.Path) -> str:
    """The llama.cpp commit scripts/build_dg_ref.sh builds: the one place the pin is written."""
    text = (root / "scripts" / "build_dg_ref.sh").read_text()
    found = re.findall(r'^PINNED_COMMIT="([0-9a-f]{40})"$', text, flags=re.M)
    if len(found) != 1:
        raise SystemExit("refusing: scripts/build_dg_ref.sh does not hold exactly one full PINNED_COMMIT")
    return found[0]


def dg_require_commit(src: pathlib.Path, pinned: str) -> str:
    head = run(["git", "-C", str(src), "rev-parse", "HEAD"]).strip()
    if head != pinned:
        raise SystemExit(f"refusing: {src} is at {head}, the pinned llama.cpp commit is {pinned}")
    return head


def llada_bound(root: pathlib.Path) -> dict:
    """No worse than the worst case of the committed LLaDA comparison, on each metric.

    The numbers come from LLADA_EVIDENCE, never typed here: if that file is replaced the bound moves
    with it, and if it is missing there is no bound, so nothing passes.
    """
    path = root / LLADA_EVIDENCE
    if not path.is_file():
        raise SystemExit(f"refusing: {LLADA_EVIDENCE} is missing, so there is no bound to hold to")
    entries = json.loads(path.read_text())["logits"]
    if not entries:
        raise SystemExit(f"refusing: {LLADA_EVIDENCE} holds no logit comparisons")
    return dict(
        min_cosine=min(e["min_cosine"] for e in entries),
        argmax_fraction=min(e["argmax_agree"] / e["rows"] for e in entries),
        mean_top5_overlap=min(e["mean_top5_overlap"] for e in entries),
        max_rel_rms=max(e["max_rel_rms"] for e in entries),
        # The soft-cap at 30 pins a row's largest logit near 30 for any correct runtime, so whole-row metrics
        # cannot see it removed (cosine 0.997, same argmax and top-5). The largest logit gets the relative
        # error LLaDA's rows reach.
        max_top1_rel_err=max(e["max_rel_rms"] for e in entries),
        source=LLADA_EVIDENCE,
    )


def dg_violations(m: dict, bound: dict) -> list[str]:
    out = []
    if not m["min_cosine"] >= bound["min_cosine"]:
        out.append(f"min_cosine {m['min_cosine']:.5f} < {bound['min_cosine']:.5f}")
    frac = m["argmax_agree"] / m["rows"]
    if not frac >= bound["argmax_fraction"]:
        out.append(f"argmax agreement {m['argmax_agree']}/{m['rows']} = {frac:.4f} < {bound['argmax_fraction']:.4f}")
    if not m["mean_top5_overlap"] >= bound["mean_top5_overlap"]:
        out.append(f"top-5 overlap {m['mean_top5_overlap']:.3f} < {bound['mean_top5_overlap']:.3f}")
    if not m["max_rel_rms"] <= bound["max_rel_rms"]:
        out.append(f"max relative RMS {m['max_rel_rms']:.4f} > {bound['max_rel_rms']:.4f}")
    if not m["top1_max_rel_err"] <= bound["max_top1_rel_err"]:
        out.append(f"top logit relative error {m['top1_max_rel_err']:.4f} > {bound['max_top1_rel_err']:.4f}")
    return out


def dg_reference_bound(records: list[dict]) -> dict:
    """Per-prompt, per-state extrema from the two directions of the cache control.

    Fault injections and changed-input experiments never determine the bound.
    Repeating the same path supplies an independent positive control.
    """
    controls = [r for r in records if r["same_inputs"] and not r["expected_rejected"]
                and r["kind"] in ("cache-path", "cache-path-reverse")]
    if not controls:
        raise SystemExit("refusing: no same-input cache-path reference controls")
    out = {}
    for prompt in dict.fromkeys(r["prompt"] for r in controls):
        out[prompt] = {}
        for state in ("no-sc", "sc"):
            rows = [r for r in controls if r["prompt"] == prompt and r["state"] == state]
            for direction in ("cache-path", "cache-path-reverse"):
                if not any(r["kind"] == direction for r in rows):
                    raise SystemExit(f"refusing: {prompt!r} {state} lacks the {direction} control")
            out[prompt][state] = dict(
                min_cosine=min(r["min_cosine"] for r in rows),
                argmax_fraction=min(r["argmax_agree"] / r["rows"] for r in rows),
                mean_top5_overlap=min(r["mean_top5_overlap"] for r in rows),
                max_rel_rms=max(r["max_rel_rms"] for r in rows),
                max_top1_rel_err=max(r["top1_max_rel_err"] for r in rows))
    return out


def dg_softcap_violations(max_abs_logit: float, cap: float) -> list[str]:
    """A direct model invariant, independent of any cross-runtime distance."""
    if not math.isfinite(max_abs_logit) or not math.isfinite(cap) or cap <= 0:
        return ["soft-cap check needs finite logits and a positive finite cap"]
    if max_abs_logit > cap:
        return [f"absolute logit {max_abs_logit:.6g} exceeds model soft-cap {cap:.6g}"]
    return []


def dg_sensitivity_bound(root: pathlib.Path) -> dict:
    """Load the fixed-input regression protocol; derive its numbers from controls."""
    path = root / DG_SENSITIVITY_EVIDENCE
    if not path.is_file():
        raise SystemExit(f"refusing: {DG_SENSITIVITY_EVIDENCE} is missing")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence["llama_cpp_commit"] != dg_pinned_commit(root):
        raise SystemExit(f"refusing: {path} was collected with a different reference commit")
    profiles = dg_reference_bound(evidence["records"])
    inputs = list(evidence["inputs"].values())
    if len(inputs) != len(DG_PROMPTS) or {c["prompt"] for c in inputs} != set(DG_PROMPTS):
        raise SystemExit(f"refusing: {path} does not cover the complete fixed DiffusionGemma prompt suite")
    check = {prompt: False for prompt in ("soft-cap removed", "output scale dropped")}
    for record in evidence["records"]:
        found = dg_violations(record, profiles[record["prompt"]][record["state"]])
        found += dg_softcap_violations(record["max_abs_logit"], record["softcap"])
        if record["same_inputs"] and not record["expected_rejected"] and found:
            raise SystemExit(f"refusing: {path} rejects its {record['kind']} reference control")
        if record["expected_rejected"]:
            for label in check:
                if label in record["kind"] and found:
                    check[label] = True
    if not all(check.values()):
        raise SystemExit(f"refusing: {path} does not reject both required mutants")
    return dict(protocol="dg-fixed-input-v1", source=DG_SENSITIVITY_EVIDENCE, model=evidence["model"],
                model_identity=evidence["model_identity"],
                llama_cpp_commit=evidence["llama_cpp_commit"],
                cases={c["prompt"]: dict(inputs=c, states=profiles[c["prompt"]]) for c in inputs})


def dg_case_bound(meta: dict, variant: str, bound: dict) -> dict:
    if bound.get("protocol") != "dg-fixed-input-v1":
        return bound  # callers of the historical gross-bug detector
    prompt = meta["prompt"]
    if prompt not in bound["cases"]:
        raise SystemExit(f"refusing: no measured DiffusionGemma bound for prompt {prompt!r}")
    expected = bound["cases"][prompt]["inputs"]
    for key in ("prompt_tokens", "canvas_tokens", "canvas_length", "n_vocab", "prev_temp_inv"):
        if meta.get(key) != expected[key]:
            raise SystemExit(f"refusing: {prompt!r} {key} differs from the measured DiffusionGemma inputs")
    for key in ("model", "llama_cpp_commit"):
        if meta.get(key) != bound[key]:
            raise SystemExit(f"refusing: {prompt!r} {key} differs from the measured DiffusionGemma reference")
    if "model_identity" in bound and dg_model_identity(meta) != bound["model_identity"]:
        raise SystemExit(f"refusing: {prompt!r} model_identity differs from the measured DiffusionGemma artifact")
    if float(meta["final_logit_softcapping"]) != float(expected["final_logit_softcapping"]):
        raise SystemExit(f"refusing: {prompt!r} final_logit_softcapping differs from the measured model")
    return bound["cases"][prompt]["states"]["sc" if variant.endswith("-sc") else "no-sc"]


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def dg_model_identity(meta: dict) -> dict:
    """Reuse supplied model provenance; never hash the 16.8 GB model for a receipt."""
    identity = dict(name=meta["model"], bytes=meta["model_bytes"])
    if meta.get("model_provenance"):
        source = meta["model_provenance"]
        if source.get("verified_bytes") != meta["model_bytes"] or source.get("filename") != meta["model"]:
            raise SystemExit("refusing: model provenance filename or verified byte count differs from the model")
        identity["provenance"] = {k: source[k] for k in ("repository", "revision", "filename", "declared_hf_lfs_oid")}
    elif meta.get("model_sha256"):
        identity["sha256"] = meta["model_sha256"]
    else:
        identity["reference_file"] = dict(path=meta["model_path"], mtime_ns=meta["model_mtime_ns"])
    return identity


def dg_receipt_inputs(case_dir: pathlib.Path, meta: dict, variant: str) -> dict:
    previous = None
    if variant.endswith("-sc"):
        path = case_dir / DG_PREV_FILE
        if not path.is_file():
            raise SystemExit(f"refusing: shared previous-logit file {path} is missing")
        previous = dict(file=DG_PREV_FILE, bytes=path.stat().st_size, sha256=_sha256(path),
                        temperature_inverse=meta["prev_temp_inv"])
    return dict(variant=variant, model=dg_model_identity(meta), llama_cpp_commit=meta["llama_cpp_commit"],
                prompt_tokens=meta["prompt_tokens"], canvas_tokens=meta["canvas_tokens"],
                n_vocab=meta["n_vocab"], canvas_length=meta["canvas_length"],
                final_logit_softcapping=float(meta["final_logit_softcapping"]), previous=previous)


def dg_validate_receipt(case_dir: pathlib.Path, meta: dict, variant: str) -> dict:
    path = case_dir / f"{variant}.mojo.receipt.json"
    if not path.is_file():
        raise SystemExit(f"refusing: candidate execution receipt {path} is missing; use bench/dg_candidate.py")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    execution = receipt.get("execution", {})
    if (receipt.get("producer") != "bench/dg_candidate.py/v1" or execution.get("exit_status") != 0
            or not execution.get("argv") or not execution.get("executable")
            or not execution.get("candidate_revision")):
        raise SystemExit(f"refusing: {path} has no completed candidate execution")
    if receipt.get("inputs") != dg_receipt_inputs(case_dir, meta, variant):
        raise SystemExit(f"refusing: {path} inputs differ from the required model, variant, canvas or shared previous logits")
    output = case_dir / f"{variant}.mojo.f32"
    expected_bytes = meta["canvas_length"] * meta["n_vocab"] * 4
    identity = dict(file=output.name, bytes=output.stat().st_size, sha256=_sha256(output))
    if identity["bytes"] != expected_bytes or receipt.get("output") != identity:
        raise SystemExit(f"refusing: {path} output differs from the candidate produced by its execution")
    return receipt


def _load_f32(path: pathlib.Path, vocab: int):
    import numpy as np

    return np.fromfile(path, dtype=np.float32).reshape(-1, vocab).astype(np.float64)


def dg_top1_error(top_a: list[float], top_b: list[float]) -> float:
    """The worst |a - b| / |b| over rows, from each row's largest logit (b is the reference)."""
    if not top_a or len(top_a) != len(top_b):
        raise SystemExit("refusing: the row maxima are empty or differ in length")
    return max(abs(x - y) / abs(y) for x, y in zip(top_a, top_b))


def dg_metrics(a: np.ndarray, b: np.ndarray, first_mask: int) -> dict:
    """`compare`, plus the worst relative error of a row's largest logit (b is the reference)."""
    m = compare(a, b, first_mask)
    m["top1_max_rel_err"] = dg_top1_error(a.max(axis=1).tolist(), b.max(axis=1).tolist())
    return m


def dg_compare(
    workdir: pathlib.Path,
    bound: dict,
    load: Callable[[pathlib.Path], object] | None = None,
    compare_fn: Callable[[object, object, int], dict] | None = None,
) -> tuple[list[dict], list[str]]:
    """Compare mojo-dllm's saved canvas logits with the reference's, case by case and variant by variant.

    Every file is checked before the first number is computed, so a half-finished set never produces a
    partial table. `load` and `compare_fn` default to numpy; tests pass plain stand-ins.
    """
    cases = sorted(workdir.glob("case*/case.json"))
    if not cases:
        raise SystemExit(f"refusing: no reference cases under {workdir}; run --phase ref first")
    plan = []
    seen_prompts = set()
    for case_json in cases:
        d = case_json.parent
        meta = json.loads(case_json.read_text(encoding="utf-8"))
        seen_prompts.add(meta["prompt"])
        for v in DG_VARIANTS:
            receipt = None
            ref, mojo = d / f"{v}.ref.f32", d / f"{v}.mojo.f32"
            if not ref.is_file():
                raise SystemExit(f"refusing: the reference output {ref} is missing; run --phase ref again")
            if not mojo.is_file():
                raise SystemExit(
                    f"refusing: mojo-dllm's output {mojo} is missing. The DiffusionGemma forward pass is #64 "
                    "and has to write this file; the reference's own output is never compared with itself."
                )
            case_bound = dg_case_bound(meta, v, bound)
            fixed_input = bound.get("protocol") == "dg-fixed-input-v1"
            if fixed_input:
                receipt = dg_validate_receipt(d, meta, v)
            if (not fixed_input or case_bound["max_rel_rms"] > 0) and mojo.read_bytes() == ref.read_bytes():
                raise SystemExit(
                    f"refusing: {mojo} is byte-identical to {ref}; this comparison requires a distinct "
                    "candidate output when the measured reference spread is nonzero."
                )
            if "ref_spread" not in meta:
                raise SystemExit(f"refusing: {case_json} has no ref_spread; run --phase ref again")
            plan.append((meta, v, ref, mojo, receipt))
    if bound.get("protocol") == "dg-fixed-input-v1" and seen_prompts != set(bound["cases"]):
        raise SystemExit("refusing: the saved cases do not cover the complete measured DiffusionGemma suite")
    if compare_fn is None:
        compare_fn = dg_metrics
    entries, violations = [], []
    for case, v, ref, mojo, receipt in plan:
        if load is None:
            vocab = case["n_vocab"]
            a, b = _load_f32(mojo, vocab), _load_f32(ref, vocab)
            if a.size == 0 or a.shape != b.shape:
                raise SystemExit(f"refusing: {mojo} and {ref} are empty or differ in shape")
        else:
            a, b = load(mojo), load(ref)
        m = compare_fn(a, b, 0)
        entries.append(
            dict(prompt=case["prompt"], variant=v, seq_len=len(case["prompt_tokens"]) + m["rows"],
                 reference_spread=case["ref_spread"], **m)
        )
        if receipt is not None:
            entries[-1]["candidate_receipt"] = receipt
        found = dg_violations(m, dg_case_bound(case, v, bound))
        if bound.get("protocol") == "dg-fixed-input-v1":
            maximum = float(abs(a).max()) if load is None else m["max_abs_logit"]
            found += dg_softcap_violations(maximum, float(case["final_logit_softcapping"]))
            entries[-1]["max_abs_logit"] = maximum
        violations += [f"{case['prompt'][:40]!r} {v}: {x}" for x in found]
    return entries, violations


def dg_write_noisy(src: pathlib.Path, dst: pathlib.Path, vocab: int) -> float:
    """dst = src with each value scaled by 1 + DG_NOISE_REL * N(0, 1); returns the relative L2 change."""
    import numpy as np

    prev = _load_f32(src, vocab).astype(np.float32)
    rng = np.random.default_rng(DG_NOISE_SEED)
    noisy = (prev * (1.0 + DG_NOISE_REL * rng.standard_normal(prev.shape).astype(np.float32))).astype(np.float32)
    rel = float(np.linalg.norm(noisy.astype(np.float64) - prev) / np.linalg.norm(prev))
    if not 0.5 * DG_NOISE_REL <= rel <= 2 * DG_NOISE_REL:
        raise SystemExit(f"refusing: the perturbation is {rel:.3g} relative, not about {DG_NOISE_REL:g}")
    noisy.tofile(dst)
    return rel


def dg_spread(case_dir: pathlib.Path, vocab: int) -> dict:
    """How far the reference lies from itself: another path to the same math, and a rounding-level nudge."""
    load = lambda name: _load_f32(case_dir / name, vocab)  # noqa: E731
    pairs = {
        "cached vs unified": ("cached.ref.f32", "unified.ref.f32"),
        "cached-sc vs unified-sc": ("cached-sc.ref.f32", "unified-sc.ref.f32"),
        f"unified-sc, previous logits changed by {DG_NOISE_REL:g} relative, vs unified-sc": (
            DG_NOISY_FILE,
            "unified-sc.ref.f32",
        ),
    }
    return {label: dg_metrics(load(a), load(b), 0) for label, (a, b) in pairs.items()}


def dg_spread_lines(spread: dict) -> list[str]:
    return [
        f"{label}: min_cosine {m['min_cosine']:.4f}, argmax {m['argmax_agree']}/{m['rows']}, "
        f"top-5 {m['mean_top5_overlap']:.2f}, max relative RMS {m['max_rel_rms']:.3f}, "
        f"top logit {m['top1_max_rel_err']:.4f}"
        for label, m in spread.items()
    ]


def dg_tokenize(llama_src: pathlib.Path, model: str, prompt: str) -> list[int]:
    # With the BOS the branch adds on its own (llama.cpp forces add_bos for Gemma 4), as the model reads it.
    out = run([str(llama_src / "build/bin/llama-tokenize"), "-m", model, "-p", prompt, "--ids", "--log-disable"])
    return parse_token_ids(out)


def dg_reference(args: argparse.Namespace, workdir: pathlib.Path, llama_src: pathlib.Path) -> int:
    """Run the reference side: one process at a time, each one mapping the model file alone."""
    pinned = dg_pinned_commit(ROOT)
    head = dg_require_commit(llama_src, pinned)
    info = dict(line.split("=", 1) for line in run([args.ref, "info", args.model]).splitlines() if "=" in line)
    vocab, canvas_len = int(info["n_vocab"]), int(info["canvas_length"])
    provenance = json.loads(pathlib.Path(args.model_provenance).read_text(encoding="utf-8")) if args.model_provenance else None
    sha = None if provenance else _sha256(pathlib.Path(args.model))
    model_stat = pathlib.Path(args.model).stat()
    for i, prompt_text in enumerate(DG_PROMPTS):
        d = workdir / f"case{i}"
        d.mkdir(parents=True, exist_ok=True)
        prompt = dg_tokenize(llama_src, args.model, prompt_text)
        canvas = dg_canvas(vocab, canvas_len, DG_CANVAS_SEED + i)
        meta = dict(
            prompt=prompt_text, prompt_tokens=prompt, canvas_tokens=canvas, canvas_seed=DG_CANVAS_SEED + i,
            n_vocab=vocab, canvas_length=canvas_len, final_logit_softcapping=info["final_logit_softcapping"],
            prev_file=DG_PREV_FILE, prev_temp_inv=DG_PREV_TEMP_INV, model=pathlib.Path(args.model).name,
            model_sha256=sha, llama_cpp_commit=head,
            model_bytes=model_stat.st_size, model_mtime_ns=model_stat.st_mtime_ns,
            model_path=str(pathlib.Path(args.model).resolve()),
            model_provenance=provenance,
        )
        (d / "case.json").write_text(json.dumps(meta, indent=2) + "\n")
        tcsv, ccsv = ",".join(map(str, prompt)), ",".join(map(str, canvas))
        for v in DG_VARIANTS:
            cmd = [args.ref, "logits", args.model, tcsv, ccsv, str(d / f"{v}.ref.f32"), args.threads]
            if v.startswith("cached"):
                cmd.append("--cached")
            if v.endswith("-sc"):
                cmd += ["--prev", str(d / DG_PREV_FILE), "--prev-temp-inv", repr(DG_PREV_TEMP_INV)]
            run(cmd)
            print(f"case{i} {v}: wrote {d / (v + '.ref.f32')}")
        rel = dg_write_noisy(d / DG_PREV_FILE, d / "prev-noisy.f32", vocab)
        run([args.ref, "logits", args.model, tcsv, ccsv, str(d / DG_NOISY_FILE), args.threads,
             "--prev", str(d / "prev-noisy.f32"), "--prev-temp-inv", repr(DG_PREV_TEMP_INV)])
        print(f"case{i} noise floor: previous logits changed by {rel:.3g} relative, wrote {d / DG_NOISY_FILE}")
        meta["ref_spread"] = dg_spread(d, vocab)
        (d / "case.json").write_text(json.dumps(meta, indent=2) + "\n")
        for line in dg_spread_lines(meta["ref_spread"]):
            print(f"case{i} reference spread, {line}")
    return 0


def dgemma_main(args: argparse.Namespace) -> int:
    stem = pathlib.Path(args.model).stem
    workdir = ROOT / "build" / "dg-parity" / stem
    llama_src = pathlib.Path(args.llama_src).expanduser()
    if args.phase == "ref":
        return dg_reference(args, workdir, llama_src)
    bound = dg_sensitivity_bound(ROOT)
    entries, violations = dg_compare(workdir, bound)
    for e in entries:
        print("logits", e)
    if violations:
        seen = {e["prompt"]: e["reference_spread"] for e in entries}
        context = [f"{prompt[:40]!r}: {line}" for prompt, sp in seen.items() for line in dg_spread_lines(sp)]
        raise SystemExit(
            "outside the measured DiffusionGemma fixed-input bound:\n  " + "\n  ".join(violations)
            + "\nthe reference against itself, for reading these numbers:\n  " + "\n  ".join(context)
        )
    meta = json.loads(next(iter(sorted(workdir.glob("case*/case.json")))).read_text())
    doc = dict(
        date=dt.datetime.now().isoformat(timespec="seconds"), arch=DG_ARCH, device="cpu", model=meta["model"],
        versions={"mojo-dllm": ", ".join(sorted({e["candidate_receipt"]["execution"]["candidate_revision"] for e in entries})),
                  "llama.cpp": meta["llama_cpp_commit"]},
        reference="llama.cpp libllama (ggml-org/llama.cpp#24423 branch) via tools/ref/dg_ref.cpp",
        model_sha256=meta["model_sha256"], bound=bound, logits=entries, generation=[],
    )
    out = result_path(args.model, DG_ARCH, "cpu", dt.date.today())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-provenance", help="DiffusionGemma: existing HF revision/LFS identity and size provenance")
    ap.add_argument("--ref", default=None, help="the reference binary (build/llada_ref, or build/dg_ref for DiffusionGemma)")
    ap.add_argument("--threads", default="12")
    ap.add_argument("--arch", choices=[*sorted(SPECS), DG_ARCH], default="llada")
    ap.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    ap.add_argument("--ref2", default=None, help="a second reference binary, logits only")
    ap.add_argument("--ref2-label", default="llama.cpp CUDA")
    ap.add_argument("--phase", choices=["ref", "compare"], default=None, help="DiffusionGemma only; see the docstring")
    ap.add_argument("--llama-src", default="~/src/llama.cpp-diffusion-gemma",
                    help="DiffusionGemma only: the checkout scripts/build_dg_ref.sh built")
    args = ap.parse_args()
    if args.arch == DG_ARCH:
        if args.phase is None:
            ap.error("--arch diffusion-gemma needs --phase ref or --phase compare")
        if args.ref is None:
            args.ref = str(ROOT / "build/dg_ref")
        return dgemma_main(args)
    if args.phase is not None:
        ap.error("--phase applies to --arch diffusion-gemma only")
    if args.ref is None:
        args.ref = str(ROOT / "build/llada_ref")
    import numpy as np

    VOCAB = SPECS[args.arch]["vocab"]
    MASK = SPECS[args.arch]["mask"]
    logits = []
    with tempfile.TemporaryDirectory() as tmp:
        for case in LOGIT_CASES:
            toks = tokenize(args.model, case.prompt) + [MASK] * case.masks
            L = len(toks)
            # Every row: a sample of 8 let the published worst case depend on
            # which positions happened to be picked.
            rows = list(range(L))
            tcsv = ",".join(map(str, toks))
            rcsv = ",".join(map(str, rows))
            a, b, b2 = f"{tmp}/mojo.f32", f"{tmp}/ref.f32", f"{tmp}/ref2.f32"
            run([str(ROOT / "build/mojo-dllm"), "logits", "--model", args.model, "--tokens", tcsv,
                 "--rows", rcsv, "--out", a, "--threads", args.threads, "--device", args.device])
            run([args.ref, "logits", args.model, tcsv, rcsv, b, args.threads])
            x = np.fromfile(a, dtype=np.float32).reshape(-1, VOCAB).astype(np.float64)
            y = np.fromfile(b, dtype=np.float32).reshape(-1, VOCAB).astype(np.float64)
            if x.size == 0 or x.shape != y.shape:
                raise SystemExit("refusing: logit dumps are empty or differ in shape")
            first = len(toks) - case.masks
            entry = dict(prompt=case.prompt, seq_len=L, **compare(x, y, first))
            if args.ref2:
                run([args.ref2, "logits", args.model, tcsv, rcsv, b2, args.threads])
                z = np.fromfile(b2, dtype=np.float32).reshape(-1, VOCAB).astype(np.float64)
                if z.shape != y.shape:
                    raise SystemExit("refusing: the second reference's dump differs in shape")
                entry["vs_ref2"] = compare(x, z, first)
                entry["ref_vs_ref2"] = compare(y, z, first)
            logits.append(entry)
            print("logits", logits[-1])
    gens = []
    for case in GEN_CASES:
        toks = tokenize(args.model, case.prompt)
        out = run([str(ROOT / "build/mojo-dllm"), "run", "--model", args.model, "--prompt", case.prompt,
                   "--max-tokens", str(case.gen), "--block-length", str(case.block),
                   "--steps", str(case.steps), "--threads", args.threads, "--device", args.device, "--json",
                   "--no-early-stop"])
        j = json.loads(next(l for l in out.splitlines() if l.startswith("{")))
        mine = j["tokens"][len(toks):]
        if args.arch == "dream":
            ref_out = run([args.ref, "generate-dream", args.model, ",".join(map(str, toks)), str(case.gen),
                           str(case.steps), args.threads])
        else:
            ref_out = run([args.ref, "generate", args.model, ",".join(map(str, toks)), str(case.gen),
                           str(case.block), str(case.steps), args.threads])
        ref = [int(x) for x in ref_out.strip().splitlines()[-1].split(",")][len(toks):]
        if len(mine) != case.gen or len(ref) != case.gen:
            raise SystemExit("refusing: a generation came back with the wrong length")
        same = sum(int(p == q) for p, q in zip(mine, ref))
        first = next((i for i, (p, q) in enumerate(zip(mine, ref)) if p != q), None)
        gens.append(dict(prompt=case.prompt, gen_length=case.gen, block_length=case.block,
                         steps=case.steps, matching_tokens=same, first_divergence=first,
                         mojo_text=j["text"]))
        print("generate", gens[-1])
    doc = dict(date=dt.datetime.now().isoformat(timespec="seconds"), arch=args.arch, device=args.device,
               model=pathlib.Path(args.model).name,
               versions={"mojo-dllm": run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"]).strip()},
               reference="llama.cpp libllama via tools/ref/llada_ref.cpp", logits=logits, generation=gens)
    if args.ref2:
        doc["reference2"] = args.ref2_label
    out = result_path(args.model, args.arch, args.device, dt.date.today())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
