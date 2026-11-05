#!/usr/bin/env python3
"""Measure DiffusionGemma sensitivity and verify the fixed-input parity protocol.

usage: uv run --only-group fixtures python bench/check_dg_bound.py check CASES_DIR
           [--dropped DIR --dropped-layer N] [--out bench/checks/<date>-dg-bound.json]
       uv run --only-group fixtures python bench/check_dg_bound.py make-mutant MODEL.gguf --layer N --out MUTANT.gguf

CASES_DIR is the original output of bench/dg_sensitivity.py. The check uses the
same metric code and case/state bound as the real runtime comparison:

  controls       independent same-path repeats and both directions of cached/unified;
                 every same-input reference control must pass.
  soft-cap       the logits with the final soft-cap removed, cap * atanh(logit / cap): a runtime that
                 forgot the tanh at the end of the output head.
  dropped scale  the logits of a run where one layer's decoder output scale was set to 1.0, which is
                 what a runtime that forgot to multiply that layer's canvas rows by it computes. The run
                 uses a model made by `make-mutant`, collected with --original pointing to CASES_DIR
                 so self-conditioning reads the unmodified model's previous-step logits.

The exit status is 1 when a control is rejected or a mutant passes the entire suite.
make-mutant copies the GGUF and overwrites the four bytes
of `blk.N.layer_output_scale.weight`; the original file is never written.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import parity  # noqa: E402

ROOT = HERE.parent


def attach_provenance(evidence: dict, cases: dict, provenance: pathlib.Path, verification: pathlib.Path | None) -> dict:
    """Attach existing download/build records without repeating model inference."""
    source = json.loads(provenance.read_text(encoding="utf-8"))
    for name, (meta, path) in cases.items():
        meta["model_provenance"] = source
        identity = parity.dg_model_identity(meta)
        case_json = path.parent / "case.json"
        case_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        evidence["inputs"][name]["previous_noise_relative_l2"] = meta["previous_noise_relative_l2"]
        evidence["inputs"][name]["previous_noise_seed"] = parity.DG_NOISE_SEED
    evidence["model_identity"] = identity
    evidence["model_provenance"] = source
    if verification is not None:
        evidence["source"]["compilation_input_verification"] = json.loads(verification.read_text(encoding="utf-8"))
    evidence["protocol"] = "dg-fixed-input-v1"
    evidence["reference_configuration"] = dict(threads=next(iter(cases.values()))[0]["threads"],
                                               flash_attention=False, cached_kv_type="f32", cpu_repacking=True)
    evidence["candidate_rule"] = "per-prompt, per-SC-state same-input cache-path extrema plus the model soft-cap invariant"
    evidence["candidate_status"] = "fixed-input regression protocol; previous-logit perturbations are sensitivity diagnostics"
    return evidence


def render_sensitivity(evidence: dict) -> str:
    """Markdown values come from the recorded measurements, never from prose edits."""
    lines = ["| Case | Prompt tokens | no-SC min cosine / max rel. RMS | SC min cosine / max rel. RMS | noisy-SC median rel. RMS |",
             "|---|---:|---:|---:|---:|"]
    for name, inputs in evidence["inputs"].items():
        records = [r for r in evidence["records"] if r["case"] == name]
        no_sc = next(r for r in records if r["kind"] == "cache-path" and r["state"] == "no-sc")
        sc = next(r for r in records if r["kind"] == "cache-path" and r["state"] == "sc")
        noisy = next(r for r in records if r["kind"] == "previous-logit-noise")
        lines.append(f"| {name} | {len(inputs['prompt_tokens'])} | {no_sc['min_cosine']:.5f} / {no_sc['max_rel_rms']:.5f} "
                     f"| {sc['min_cosine']:.5f} / {sc['max_rel_rms']:.5f} | {noisy['median_rel_rms']:.5f} |")
    lines += ["", "| Fault injection | Rejected case/variant comparisons |", "|---|---:|"]
    for label in evidence["mutants_rejected"]:
        rows = [r for r in evidence["records"] if r["kind"] == label]
        lines.append(f"| {label} | {sum(r['case_envelope_rejected'] for r in rows)} / {len(rows)} |")
    return "\n".join(lines) + "\n"


def remove_softcap(logits, cap: float):
    """Inverse tanh in f64, including f32 values that rounded to exactly +/-cap."""
    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(cap) or cap <= 0 or not np.isfinite(values).all():
        raise SystemExit("refusing: soft-cap inversion needs finite logits and a positive finite cap")
    # 1 - 1e-9 rounds to 1 in f32. Promote before clipping and use a representable
    # f64 interior point, so saturation is a finite mutant rather than an infinity.
    inside = np.nextafter(np.float64(1.0), np.float64(0.0))
    return cap * np.arctanh(np.clip(values / cap, -inside, inside))


def sensitivity_report(args: argparse.Namespace) -> int:
    """Assess the empirical same-input envelope before adopting a parity rule."""
    import numpy as np

    import dg_sensitivity as sensitivity

    cases = load_cases(pathlib.Path(args.cases))
    dropped = load_cases(pathlib.Path(args.dropped))
    if set(dropped) != set(cases):
        raise SystemExit("refusing: --dropped holds different cases from CASES_DIR")
    records = []
    for name, (meta, unified) in cases.items():
        sensitivity.require_mutant_inputs(meta, dict(dropped[name][0], case_directory=str(dropped[name][1].parent)),
                                          name, unified)
        if "controls" not in meta:
            raise SystemExit(f"refusing: {name} has no controls; collect with bench/dg_sensitivity.py")
        vocab, cap = meta["n_vocab"], float(meta["final_logit_softcapping"])
        folder = unified.parent
        for control in meta["controls"]:
            a = parity._load_f32(folder / f"{control['candidate']}.ref.f32", vocab)
            b = parity._load_f32(folder / f"{control['reference']}.ref.f32", vocab)
            records.append(dict(case=name, prompt=meta["prompt"], **control,
                                sampling=sensitivity.sampling_metrics(a, b, temperature=args.temperature),
                                max_abs_logit=float(np.max(np.abs(a))), softcap=cap, expected_rejected=False))
            if control["kind"] == "cache-path":
                records.append(dict(case=name, prompt=meta["prompt"], kind="cache-path-reverse",
                                    state=control["state"], same_inputs=True,
                                    candidate=control["reference"], reference=control["candidate"],
                                    **parity.dg_metrics(b, a, 0), sampling=records[-1]["sampling"],
                                    max_abs_logit=float(np.max(np.abs(b))), softcap=cap, expected_rejected=False))
        for variant in parity.DG_VARIANTS:
            reference = parity._load_f32(folder / f"{variant}.ref.f32", vocab)
            for kind, candidate in (
                ("soft-cap removed", remove_softcap(reference, cap)),
                (f"layer {args.dropped_layer} output scale dropped",
                 parity._load_f32(dropped[name][1].parent / f"{variant}.ref.f32", vocab)),
            ):
                records.append(dict(case=name, prompt=meta["prompt"], kind=kind,
                                    state="sc" if variant.endswith("-sc") else "no-sc",
                                    same_inputs=True, candidate=kind, reference=variant,
                                    **parity.dg_metrics(candidate, reference, 0),
                                    sampling=sensitivity.sampling_metrics(candidate, reference,
                                                                          temperature=args.temperature),
                                    max_abs_logit=float(np.max(np.abs(candidate))), softcap=cap,
                                    expected_rejected=True))
    control_records = [r for r in records if not r["expected_rejected"]]
    envelopes = sensitivity.same_input_envelopes(control_records)
    prompt_bounds = parity.dg_reference_bound(records)
    case_envelopes = {name: prompt_bounds[meta["prompt"]] for name, (meta, _) in cases.items()}
    for record in records:
        record["envelope_violations"] = parity.dg_violations(record, envelopes[record["state"]])
        record["softcap_violated"] = record["max_abs_logit"] > record["softcap"]
        record["envelope_rejected"] = bool(record["envelope_violations"] or record["softcap_violated"])
        record["case_envelope_violations"] = parity.dg_violations(
            record, case_envelopes[record["case"]][record["state"]])
        record["case_envelope_rejected"] = bool(record["case_envelope_violations"] or record["softcap_violated"])
    mutant_labels = sorted({r["kind"] for r in records if r["expected_rejected"]})
    mutant_rejected = {label: any(r["case_envelope_rejected"] for r in records if r["kind"] == label)
                       for label in mutant_labels}
    controls_pass = all(not r["case_envelope_rejected"] for r in control_records if r["same_inputs"])
    input_keys = ("prompt", "prompt_tokens", "canvas_tokens", "canvas_seed", "canvas_length", "n_vocab",
                  "prev_temp_inv", "final_logit_softcapping", "threads")
    doc = dict(schema_version=1, experiment="reference-sensitivity-rule-assessment",
               date=dt.datetime.now(dt.timezone.utc).isoformat(),
               model=next(iter(cases.values()))[0]["model"],
               llama_cpp_commit=next(iter(cases.values()))[0]["llama_cpp_commit"],
               source=next(iter(cases.values()))[0].get("reference_source"),
               inputs={name: {k: meta[k] for k in input_keys} for name, (meta, _) in cases.items()},
               candidate_rule="same-input per-state metric extrema plus the model soft-cap invariant",
               candidate_status="diagnostic; input-noise experiments do not set implementation tolerance",
               same_input_envelopes=envelopes, case_envelopes=case_envelopes,
               same_input_controls_pass=controls_pass, mutants_rejected=mutant_rejected,
               all_as_expected=controls_pass and all(mutant_rejected.values()), records=records)
    sensitivity.write_json(pathlib.Path(args.out), doc)
    for record in records:
        print(f"{record['case']} {record['reference']:12} {record['kind']:35} "
              f"cos={record['min_cosine']:.5f} rms={record['max_rel_rms']:.4f} "
              f"TV={record['sampling']['max_total_variation']:.4f} "
              f"JS={record['sampling']['max_js_divergence']:.4f} "
              f"envelope_rejected={record['envelope_rejected']} case_rejected={record['case_envelope_rejected']}")
    print(f"wrote {args.out}")
    return 0


def load_cases(folder: pathlib.Path) -> dict[str, tuple[dict, pathlib.Path]]:
    cases = {}
    for case_json in sorted(folder.glob("case*/case.json")):
        ref = case_json.parent / "unified.ref.f32"
        if not ref.is_file():
            raise SystemExit(f"refusing: {ref} is missing; run bench/parity.py --phase ref first")
        cases[case_json.parent.name] = (json.loads(case_json.read_text(encoding="utf-8")), ref)
    if not cases:
        raise SystemExit(f"refusing: no cases under {folder}")
    return cases


def check(args: argparse.Namespace) -> int:
    import numpy as np

    import dg_sensitivity as sensitivity

    bound = parity.dg_sensitivity_bound(ROOT)
    cases = load_cases(pathlib.Path(args.cases))
    dropped = load_cases(pathlib.Path(args.dropped)) if args.dropped else {}
    if args.dropped and set(dropped) != set(cases):
        raise SystemExit("refusing: --dropped holds different cases from CASES_DIR")
    if {meta["prompt"] for meta, _ in cases.values()} != set(bound["cases"]):
        raise SystemExit("refusing: CASES_DIR does not contain the complete measured prompt suite")
    results, failures = [], []
    detected = {"soft-cap removed": False}
    if dropped:
        detected[f"layer {args.dropped_layer} output scale dropped"] = False
    for name, (meta, ref_path) in cases.items():
        vocab = meta["n_vocab"]
        cap = float(meta["final_logit_softcapping"])
        if dropped:
            sensitivity.require_mutant_inputs(meta, dict(dropped[name][0], case_directory=str(dropped[name][1].parent)),
                                              name, ref_path)
        if "controls" not in meta:
            raise SystemExit(f"refusing: {name} lacks independent controls; use bench/dg_sensitivity.py")
        plan = [(r["kind"], r["candidate"], r["reference"], False) for r in meta["controls"] if r["same_inputs"]]
        plan += [("cache-path-reverse", "unified" + suffix, "cached" + suffix, False) for suffix in ("", "-sc")]
        for variant in parity.DG_VARIANTS:
            plan.append(("soft-cap removed", variant, variant, True))
            if dropped:
                plan.append((f"layer {args.dropped_layer} output scale dropped", variant, variant, True))
        for label, candidate_name, variant, is_mutant in plan:
            ref = parity._load_f32(ref_path.parent / f"{variant}.ref.f32", vocab)
            source = dropped[name][1].parent if label.endswith("output scale dropped") else ref_path.parent
            candidate = parity._load_f32(source / f"{candidate_name}.ref.f32", vocab)
            if label == "soft-cap removed":
                candidate = remove_softcap(candidate, cap)
            if not candidate.size or candidate.shape != ref.shape:
                raise SystemExit(f"refusing: {name} {label} is empty or differs in shape")
            m = parity.dg_metrics(candidate, ref, 0)
            violations = parity.dg_violations(m, parity.dg_case_bound(meta, variant, bound))
            maximum = float(np.max(np.abs(candidate)))
            violations += parity.dg_softcap_violations(maximum, cap)
            rejected = bool(violations)
            if is_mutant:
                detected[label] |= rejected
            elif rejected:
                failures.append(f"{name} {variant} {label}: the unmodified reference was rejected")
            results.append(dict(case=name, prompt=meta["prompt"], variant=variant, mutant=label, rejected=rejected,
                                is_mutant=is_mutant, max_abs_logit=maximum, violations=violations, **m))
            print(f"{name} {variant:12} {label:35} rejected={rejected!s:5} min_cos={m['min_cosine']:.5f} "
                  f"argmax={m['argmax_agree']}/{m['rows']} max_rel_rms={m['max_rel_rms']:.4f} {violations}")
    failures += [f"{label}: accepted on every case and variant" for label, rejected in detected.items() if not rejected]
    if args.out:
        first = next(iter(cases.values()))[0]
        doc = dict(date=dt.datetime.now().isoformat(timespec="seconds"), bound=bound, model=first["model"],
                   model_sha256=first["model_sha256"], llama_cpp_commit=first["llama_cpp_commit"],
                   dropped_layer=args.dropped_layer if args.dropped else None,
                   dropped_model=(
                       {k: next(iter(dropped.values()))[0][k] for k in ("model", "model_sha256")} if dropped else None
                   ),
                   results=results, expectation_scope="every same-input control passes; each mutant fails somewhere",
                   mutants_rejected=detected,
                   all_as_expected=not failures)
        pathlib.Path(args.out).write_text(json.dumps(doc, indent=2) + "\n")
        print(f"wrote {args.out}")
    if failures:
        print("the bound does not separate good from bad here:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


def make_mutant(args: argparse.Namespace) -> int:
    from gguf import GGUFReader

    src, out = pathlib.Path(args.model), pathlib.Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing: {out} exists")
    name = f"blk.{args.layer}.layer_output_scale.weight"
    reader = GGUFReader(str(src))
    found = [t for t in reader.tensors if t.name == name]
    if len(found) != 1:
        raise SystemExit(f"refusing: {src} has {len(found)} tensors named {name}")
    tensor = found[0]
    if tensor.n_elements != 1 or tensor.tensor_type.name != "F32":
        raise SystemExit(f"refusing: {name} is not a single f32 ({tensor.n_elements} x {tensor.tensor_type.name})")
    before = float(tensor.data[0])
    if before == 1.0:
        raise SystemExit(f"refusing: {name} is already 1.0, so dropping it changes nothing")
    shutil.copyfile(src, out)
    with out.open("r+b") as f:
        f.seek(int(tensor.data_offset))
        if struct.unpack("<f", f.read(4))[0] != before:
            out.unlink()
            raise SystemExit(f"refusing: the bytes at the offset of {name} are not the value gguf read")
        f.seek(int(tensor.data_offset))
        f.write(struct.pack("<f", 1.0))
    after = float([t for t in GGUFReader(str(out)).tensors if t.name == name][0].data[0])
    if after != 1.0:
        out.unlink()
        raise SystemExit(f"refusing: {name} reads {after} after the patch, not 1.0")
    print(f"{name}: {before} -> {after} in {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("cases")
    c.add_argument("--dropped", default=None)
    c.add_argument("--dropped-layer", type=int, default=None)
    c.add_argument("--out", default=None)
    mk = sub.add_parser("make-mutant")
    mk.add_argument("model")
    mk.add_argument("--layer", type=int, required=True)
    mk.add_argument("--out", required=True)
    report = sub.add_parser("analyze-sensitivity", help="assess controls and mutants without changing the parity rule")
    report.add_argument("cases")
    report.add_argument("--dropped", required=True)
    report.add_argument("--dropped-layer", type=int, required=True)
    report.add_argument("--temperature", type=float, default=0.8)
    report.add_argument("--out", required=True)
    annotate = sub.add_parser("attach-provenance", help="attach existing clean download/build provenance to a measurement")
    annotate.add_argument("evidence")
    annotate.add_argument("--cases", required=True)
    annotate.add_argument("--dropped", required=True)
    annotate.add_argument("--model-provenance", type=pathlib.Path, required=True)
    annotate.add_argument("--reference-verification", type=pathlib.Path)
    render = sub.add_parser("render-sensitivity", help="render the recorded spread and fault counts as Markdown")
    render.add_argument("evidence")
    args = ap.parse_args()
    if args.cmd == "attach-provenance":
        import dg_sensitivity as sensitivity

        original = load_cases(pathlib.Path(args.cases))
        dropped = load_cases(pathlib.Path(args.dropped))
        if set(original) != set(dropped):
            ap.error("original and mutant case sets differ")
        for name, (meta, previous) in original.items():
            sensitivity.require_mutant_inputs(meta, dict(dropped[name][0], case_directory=str(dropped[name][1].parent)),
                                              name, previous)
        path = pathlib.Path(args.evidence)
        document = attach_provenance(json.loads(path.read_text(encoding="utf-8")), original,
                                     args.model_provenance, args.reference_verification)
        document["mutant_shared_previous_execution_verified"] = True
        sensitivity.write_json(path, document)
        return 0
    if args.cmd == "render-sensitivity":
        print(render_sensitivity(json.loads(pathlib.Path(args.evidence).read_text(encoding="utf-8"))), end="")
        return 0
    if args.cmd == "analyze-sensitivity":
        return sensitivity_report(args)
    if args.cmd == "check":
        if (args.dropped is None) != (args.dropped_layer is None):
            ap.error("--dropped and --dropped-layer go together")
        return check(args)
    return make_mutant(args)


if __name__ == "__main__":
    raise SystemExit(main())
