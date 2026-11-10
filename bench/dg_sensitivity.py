#!/usr/bin/env python3
"""Collect DiffusionGemma reference controls, one full-model process at a time.

The repeated and cached passes have identical inputs. The noisy previous-step
buffer deliberately changes an input and is recorded separately as sensitivity,
not as an implementation tolerance. A mutant always reads the ORIGINAL model's
previous-step buffer, so a single-step fault is not mixed with accumulated error.

  python bench/dg_sensitivity.py --model MODEL --ref build/dg_ref \
      --llama-src LLAMA --out build/dg-sensitivity/original
  python bench/dg_sensitivity.py --model MUTANT --ref build/dg_ref \
      --llama-src LLAMA --out build/dg-sensitivity/dropped \
      --original build/dg-sensitivity/original

--resume reuses only completed outputs with matching recorded commands and
model metadata. Delete an output and its .run.json sidecar to rerun one pass.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import time
from dataclasses import dataclass

import parity

PROMPTS = parity.DG_PROMPTS


@dataclass(frozen=True)
class Forward:
    name: str
    output: pathlib.Path
    previous: pathlib.Path | None
    cached: bool
    threads: str


def forward_plan(case_dir: pathlib.Path, original_dir: pathlib.Path, threads: str, *, mutant: bool) -> list[Forward]:
    names = list(parity.DG_VARIANTS)
    if not mutant:
        names += ["unified-repeat", "unified-sc-repeat", "unified-sc-noisy"]
    return [
        Forward(name, case_dir / f"{name}.ref.f32",
                original_dir / ("prev-noisy.f32" if name.endswith("-noisy") else parity.DG_PREV_FILE)
                if "-sc" in name else None,
                name.startswith("cached"), threads)
        for name in names
    ]


def require_same_inputs(original: dict, other: dict, case: str) -> None:
    for key in ("prompt", "prompt_tokens", "canvas_tokens", "n_vocab", "prev_temp_inv"):
        if original.get(key) != other.get(key):
            raise SystemExit(f"refusing: {case} {key} differs from the original reference")


def require_mutant_inputs(original: dict, mutant: dict, case: str, previous: pathlib.Path) -> None:
    """A mutant changes weights; its SC input must still come from the original."""
    require_same_inputs(original, mutant, case)
    if pathlib.Path(mutant.get("prev_file", "")).resolve() != previous.resolve():
        raise SystemExit(f"refusing: {case} mutant must read the original reference's previous logits")
    for variant in ("unified-sc", "cached-sc"):
        # The collector also records the actual command. Validate it when an
        # execution directory is supplied by the caller, beyond case metadata.
        receipt = pathlib.Path(mutant.get("case_directory", ".")) / f"{variant}.ref.run.json"
        if "case_directory" in mutant:
            if not receipt.is_file():
                raise SystemExit(f"refusing: {case} mutant execution receipt {receipt} is missing")
            command = json.loads(receipt.read_text(encoding="utf-8"))["command"]
            if ("--prev" not in command or pathlib.Path(command[command.index("--prev") + 1]).resolve() != previous.resolve()
                    or "--prev-temp-inv" not in command
                    or float(command[command.index("--prev-temp-inv") + 1]) != original["prev_temp_inv"]):
                raise SystemExit(f"refusing: {case} {variant} executed with different previous logits or temperature")


def write_json(path: pathlib.Path, value: dict) -> None:
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    part.replace(path)


def reference_source(llama_src: pathlib.Path, manifest: pathlib.Path | None) -> dict:
    pin = parity.dg_pinned_commit(parity.ROOT)
    if manifest is None:
        return dict(commit=parity.dg_require_commit(llama_src, pin), mechanism="git HEAD at pinned commit")
    source = json.loads(manifest.read_text(encoding="utf-8"))
    if source.get("pin") != pin or not source.get("mechanism") or not source.get("files"):
        raise SystemExit(f"refusing: {manifest} lacks the pinned commit, source mechanism, or file manifest")
    return dict(commit=pin, mechanism="reconstructed pinned CPU sources; see manifest", manifest=source)


def controls(case_dir: pathlib.Path, vocab: int) -> list[dict]:
    pairs = (
        ("repeat", "no-sc", "unified-repeat", "unified"),
        ("cache-path", "no-sc", "cached", "unified"),
        ("repeat", "sc", "unified-sc-repeat", "unified-sc"),
        ("cache-path", "sc", "cached-sc", "unified-sc"),
        ("previous-logit-noise", "sc", "unified-sc-noisy", "unified-sc"),
    )
    entries = []
    for kind, state, candidate, reference in pairs:
        a = parity._load_f32(case_dir / f"{candidate}.ref.f32", vocab)
        b = parity._load_f32(case_dir / f"{reference}.ref.f32", vocab)
        entries.append(dict(kind=kind, state=state, same_inputs=kind != "previous-logit-noise",
                            candidate=candidate, reference=reference, **parity.dg_metrics(a, b, 0)))
    return entries


def same_input_envelopes(records: list[dict]) -> dict:
    """Diagnostic candidate bounds, excluding experiments that change the input.

    These extrema alone are not acceptance evidence: a held-out control and the
    fault injections still have to be evaluated against the proposed rule.
    """
    out = {}
    for state in ("no-sc", "sc"):
        values = [m for m in records if m["state"] == state and m["same_inputs"]]
        if not values:
            raise SystemExit(f"refusing: no same-input reference controls for {state}")
        out[state] = dict(min_cosine=min(m["min_cosine"] for m in values),
                          argmax_fraction=min(m["argmax_agree"] / m["rows"] for m in values),
                          mean_top5_overlap=min(m["mean_top5_overlap"] for m in values),
                          max_rel_rms=max(m["max_rel_rms"] for m in values),
                          max_top1_rel_err=max(m["top1_max_rel_err"] for m in values))
    return out


def sampling_metrics(candidate, reference, *, temperature: float) -> dict:
    """Distances between each row's sampled distribution, in nats for JS.

    Work row by row: a full DiffusionGemma canvas has 67 million logits. A
    constant shift changes raw-logit RMS but leaves these distributions equal.
    """
    import numpy as np

    if not np.isfinite(temperature) or temperature <= 0:
        raise SystemExit("refusing: sampling temperature must be positive and finite")
    if candidate.shape != reference.shape or candidate.ndim != 2 or not candidate.size:
        raise SystemExit("refusing: sampling metrics need two non-empty logit matrices of the same shape")
    tv, js = [], []
    for x, y in zip(candidate, reference):
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise SystemExit("refusing: sampling metrics received non-finite logits")
        p = np.exp((x.astype(np.float64) - x.max()) / temperature)
        q = np.exp((y.astype(np.float64) - y.max()) / temperature)
        p /= p.sum()
        q /= q.sum()
        middle = (p + q) * 0.5
        pmask, qmask = p > 0, q > 0
        divergence = 0.5 * (np.sum(p[pmask] * np.log(p[pmask] / middle[pmask]))
                            + np.sum(q[qmask] * np.log(q[qmask] / middle[qmask])))
        tv.append(float(np.sum(np.abs(p - q)) * 0.5))
        js.append(max(0.0, float(divergence)))
    return dict(temperature=temperature, max_total_variation=max(tv), mean_total_variation=float(np.mean(tv)),
                max_js_divergence=max(js), mean_js_divergence=float(np.mean(js)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--llama-src", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--original", type=pathlib.Path, help="original case directory when collecting a mutant")
    ap.add_argument("--threads", default="8")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--model-sha256", help="optional existing model provenance; not recomputed")
    ap.add_argument("--model-provenance", type=pathlib.Path, help="existing clean HF model provenance JSON")
    ap.add_argument("--source-manifest", type=pathlib.Path,
                    help="record a reconstructed pinned source manifest instead of requiring git HEAD")
    args = ap.parse_args()
    if int(args.threads) <= 0:
        ap.error("--threads must be positive")
    source = reference_source(args.llama_src, args.source_manifest)
    model = pathlib.Path(args.model).resolve()
    stat = model.stat()
    identity = dict(model=model.name, model_bytes=stat.st_size, model_mtime_ns=stat.st_mtime_ns,
                    model_path=str(model), model_sha256=args.model_sha256,
                    llama_cpp_commit=source["commit"], reference_source=source)
    if args.model_provenance:
        identity["model_provenance"] = json.loads(args.model_provenance.read_text(encoding="utf-8"))
        parity.dg_model_identity(identity)
    info = dict(line.split("=", 1) for line in parity.run([args.ref, "info", str(model)]).splitlines() if "=" in line)
    vocab, canvas_len = int(info["n_vocab"]), int(info["canvas_length"])
    args.out.mkdir(parents=True, exist_ok=True)
    all_cases = []
    for i, prompt in enumerate(PROMPTS):
        d = args.out / f"case{i}"
        d.mkdir(parents=True, exist_ok=True)
        original_dir = args.original / d.name if args.original else d
        if args.original:
            original = json.loads((original_dir / "case.json").read_text(encoding="utf-8"))
            if original["prompt"] != prompt or original["n_vocab"] != vocab:
                raise SystemExit(f"refusing: {original_dir} is a different prompt or vocabulary")
            tokens, canvas = original["prompt_tokens"], original["canvas_tokens"]
        else:
            tokens = parity.dg_tokenize(args.llama_src, str(model), prompt)
            canvas = parity.dg_canvas(vocab, canvas_len, parity.DG_CANVAS_SEED + i)
        meta = dict(**identity, prompt=prompt, prompt_tokens=tokens, canvas_tokens=canvas,
                    canvas_seed=parity.DG_CANVAS_SEED + i, canvas_length=canvas_len, n_vocab=vocab,
                    final_logit_softcapping=info["final_logit_softcapping"],
                    prev_file=str(original_dir / parity.DG_PREV_FILE), prev_temp_inv=parity.DG_PREV_TEMP_INV,
                    threads=int(args.threads))
        if args.original:
            require_same_inputs(original, meta, d.name)
        meta_path = d / "case.json"
        if meta_path.exists():
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
            if not args.resume or any(existing.get(k) != v for k, v in meta.items()):
                raise SystemExit(f"refusing: {meta_path} exists; --resume requires matching inputs and model metadata")
        write_json(meta_path, meta)
        tcsv, ccsv = ",".join(map(str, tokens)), ",".join(map(str, canvas))
        for job in forward_plan(d, original_dir, args.threads, mutant=bool(args.original)):
            if job.name.endswith("-noisy"):
                meta["previous_noise_relative_l2"] = parity.dg_write_noisy(
                    d / parity.DG_PREV_FILE, d / "prev-noisy.f32", vocab)
            command = [args.ref, "logits", str(model), tcsv, ccsv, str(job.output), job.threads]
            if job.cached:
                command.append("--cached")
            if job.previous is not None:
                command += ["--prev", str(job.previous), "--prev-temp-inv", repr(parity.DG_PREV_TEMP_INV)]
            receipt = job.output.with_suffix(".run.json")
            expected_bytes = canvas_len * vocab * 4
            if job.output.exists() or receipt.exists():
                if (not args.resume or not job.output.is_file() or job.output.stat().st_size != expected_bytes
                        or not receipt.is_file()
                        or json.loads(receipt.read_text(encoding="utf-8")).get("command") != command):
                    raise SystemExit(f"refusing: incomplete or mismatched existing output {job.output}")
                print(f"{d.name} {job.name}: reused complete output", flush=True)
                continue
            started = time.monotonic()
            parity.run(command)
            if not job.output.is_file() or job.output.stat().st_size != expected_bytes:
                raise SystemExit(f"refusing: {job.output} does not contain {expected_bytes} bytes")
            elapsed = time.monotonic() - started
            write_json(receipt, dict(command=command, elapsed_seconds=elapsed,
                                    date=dt.datetime.now(dt.timezone.utc).isoformat()))
            print(f"{d.name} {job.name}: {elapsed:.2f}s", flush=True)
        if not args.original:
            meta["controls"] = controls(d, vocab)
            meta["ref_spread"] = parity.dg_spread(d, vocab)
        write_json(meta_path, meta)
        all_cases.append(meta)
    write_json(args.out / "sensitivity.json", dict(schema_version=1, experiment="reference-self-sensitivity",
               reference_identity=identity, original=str(args.original) if args.original else None,
               cases=all_cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
