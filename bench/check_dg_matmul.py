#!/usr/bin/env python3
"""Replay a traced first-layer Q projection at several prompt/canvas splits.

    python bench/check_dg_matmul.py MODEL.gguf UNIFIED_TRACE --matmul build/dg_matmul --out-dir NEW_DIR

The matrix operator sees fixed f32 input rows. The canvas has the trace's
256 rows, while prompt lengths 8, 9, 12, 14 and 16 test alignment. This tests
the matrix product alone, with CPU repacking enabled and disabled.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess

import numpy as np

import compare_dg_traces


def main() -> int:
    from gguf import GGUFReader

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("unified", type=pathlib.Path)
    ap.add_argument("--prompt", type=int, default=9)
    ap.add_argument("--matmul", type=pathlib.Path, required=True)
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir()
    binary = str(args.matmul.resolve())
    tensors = compare_dg_traces.load(args.unified)
    x = tensors[("unified", "attn_norm-0")][1].squeeze()
    original = tensors[("unified", "Qcur-0")][1].squeeze()
    reader = GGUFReader(args.model)
    weight = next(t for t in reader.tensors if t.name == "blk.0.attn_q.weight")
    k, n = map(int, weight.shape)
    raw = args.out_dir / "attn_q.bin"
    weight.data.tofile(raw)

    def product(inputs, name, no_repack):
        input_path = args.out_dir / f"{name}.input.f32"
        output_path = args.out_dir / f"{name}.output.f32"
        inputs.astype("<f4").tofile(input_path)
        command = [binary, str(weight.tensor_type.value), str(k), str(n), str(raw),
                   str(input_path), str(len(inputs)), str(output_path), "1"]
        if no_repack:
            command.append("--no-repack")
        done = subprocess.run(command, check=True, capture_output=True, text=True)
        values = np.fromfile(output_path, dtype="<f4").reshape(len(inputs), n)
        return values, done.stderr.strip()

    replay, backend = product(x, "original-replay", False)
    results = dict(weight=weight.name, weight_type=weight.tensor_type.name, backend=backend,
                   original_replay_bit_equal=bool(np.array_equal(replay.view(np.uint32), original.view(np.uint32))),
                   original_replay_max_abs=float(np.max(np.abs(replay - original))), tests=[])
    # This is a shape test, so the prompt pool may include canvas-origin rows.
    # Reuse fixed rows to keep every canvas input identical across the splits.
    prompt_pool, canvas = x[:16], x[args.prompt:]
    for no_repack in (False, True):
        for p in (8, 9, 12, 14, 16):
            stem = f"{'plain' if no_repack else 'repacked'}-p{p}"
            both, backend = product(np.concatenate([prompt_pool[:p], canvas]), stem + "-unified", no_repack)
            prefill, _ = product(prompt_pool[:p], stem + "-prefill", no_repack)
            decode, _ = product(canvas, stem + "-decode", no_repack)
            cached = np.concatenate([prefill, decode])
            different = np.flatnonzero(np.any(both.view(np.uint32) != cached.view(np.uint32), axis=1)).tolist()
            result = dict(prompt_length=p, no_repack=no_repack, backend=backend, differing_rows=different,
                          max_abs_diff=float(np.max(np.abs(both.astype(np.float64) - cached))))
            results["tests"].append(result)
            print(json.dumps(result), flush=True)
    (args.out_dir / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
