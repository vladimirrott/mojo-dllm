#!/usr/bin/env python3
"""Execute one candidate forward pass and bind its output to the parity inputs.

  python bench/dg_candidate.py CASE_DIR --model MODEL --variant unified --candidate-revision COMMIT -- \
      CANDIDATE --case {case_json} --variant {variant} --model {model} --out {out}

The command is an argv template, executed without a shell. It must consume the
case JSON (or both {prompt_csv} and {canvas_csv}), {variant}, {model}, and {out}.
Self-conditioned variants also require {prev} and {prev_temp_inv}. The runtime
forward pass is supplied by the caller; this wrapper records its real execution.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import shutil
import subprocess
import time

import parity


def export_candidate(case_dir: pathlib.Path, model: pathlib.Path, variant: str, template: list[str],
                     *, candidate_revision: str) -> None:
    case_dir, model = case_dir.resolve(), model.resolve()
    meta = json.loads((case_dir / "case.json").read_text(encoding="utf-8"))
    stat = model.stat()
    if (str(model) != meta["model_path"] or stat.st_size != meta["model_bytes"]
            or stat.st_mtime_ns != meta["model_mtime_ns"]):
        raise SystemExit("refusing: candidate model path/size/mtime differs from the reference case's source model")
    if variant not in parity.DG_VARIANTS:
        raise SystemExit(f"refusing: unknown DiffusionGemma variant {variant}")
    joined = " ".join(template)
    required = ["{model}", "{out}", "{variant}"]
    required += ["{case_json}"] if "{case_json}" in joined else ["{prompt_csv}", "{canvas_csv}"]
    if variant.endswith("-sc"):
        required += ["{prev}", "{prev_temp_inv}"]
    if any(key not in joined for key in required):
        raise SystemExit(f"refusing: command must consume these input placeholders: {', '.join(required)}")
    output = case_dir / f"{variant}.mojo.f32"
    receipt = case_dir / f"{variant}.mojo.receipt.json"
    if output.exists() or receipt.exists():
        raise SystemExit(f"refusing: {output} or its receipt already exists")
    inputs = parity.dg_receipt_inputs(case_dir, meta, variant)
    values = dict(model=str(model), out=str(output), variant=variant, case_json=str(case_dir / "case.json"),
                  prompt_csv=",".join(map(str, meta["prompt_tokens"])),
                  canvas_csv=",".join(map(str, meta["canvas_tokens"])),
                  prev=str(case_dir / parity.DG_PREV_FILE), prev_temp_inv=str(meta["prev_temp_inv"]))
    argv = [part.format_map(values) for part in template]
    executable = shutil.which(argv[0])
    if not executable:
        raise SystemExit(f"refusing: candidate executable {argv[0]} was not found")
    if not candidate_revision.strip():
        raise SystemExit("refusing: candidate revision must identify the build being executed")
    executable_path = pathlib.Path(executable).resolve()
    executable_identity = dict(path=str(executable_path), bytes=executable_path.stat().st_size,
                               sha256=parity._sha256(executable_path))
    started = time.monotonic()
    result = subprocess.run(argv, check=False)
    if result.returncode:
        raise SystemExit(f"candidate command failed with exit status {result.returncode}; no receipt written")
    expected_bytes = meta["canvas_length"] * meta["n_vocab"] * 4
    if not output.is_file() or output.stat().st_size != expected_bytes:
        raise SystemExit(f"refusing: candidate output {output} must contain {expected_bytes} bytes")
    after = model.stat()
    if after.st_size != stat.st_size or after.st_mtime_ns != stat.st_mtime_ns:
        raise SystemExit("refusing: the model changed during candidate execution")
    if inputs != parity.dg_receipt_inputs(case_dir, meta, variant):
        raise SystemExit("refusing: shared previous logits changed during candidate execution")
    doc = dict(producer="bench/dg_candidate.py/v1", inputs=inputs,
               execution=dict(argv=argv, executable=executable_identity, candidate_revision=candidate_revision,
                              exit_status=0,
                              elapsed_seconds=time.monotonic() - started,
                              date=dt.datetime.now(dt.timezone.utc).isoformat()),
               output=dict(file=output.name, bytes=output.stat().st_size, sha256=parity._sha256(output)))
    receipt.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", type=pathlib.Path)
    ap.add_argument("--model", type=pathlib.Path, required=True)
    ap.add_argument("--variant", choices=parity.DG_VARIANTS, required=True)
    ap.add_argument("--candidate-revision", required=True, help="revision of the candidate executable being run")
    args, command = ap.parse_known_args()
    if command and command[0] == "--":
        command.pop(0)
    if not command:
        ap.error("supply the candidate command after --")
    export_candidate(args.case, args.model, args.variant, command, candidate_revision=args.candidate_revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
