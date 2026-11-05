#!/usr/bin/env python3
"""Compare dg_ref graph traces from unified and prompt-K/V forwards.

    python bench/compare_dg_traces.py UNIFIED_DIR CACHED_DIR --prompt 9 --out report.json

The manifests describe logical ggml dimension order. Each matching prompt or
canvas tensor is aligned along the dimension whose length changes across the
two paths. The metrics are the ones used by check_dg_bound.py, plus exact
equality and maximum absolute error. This is a diagnostic, with no pass bound.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import parity


def load(folder: pathlib.Path) -> dict[tuple[str, str], tuple[dict, np.ndarray]]:
    tensors = {}
    for line in (folder / "manifest.jsonl").read_text().splitlines():
        meta = json.loads(line)
        key = (meta["phase"], meta["name"])
        if key in tensors:
            raise ValueError(f"{folder}: repeated tensor {key}")
        ne = meta["ne"]
        a = np.fromfile(folder / meta["file"], dtype="<f4")
        if a.size != int(np.prod(ne)):
            raise ValueError(f"{folder / meta['file']}: wrong size for {ne}")
        if not np.isfinite(a).all():
            raise ValueError(f"{folder / meta['file']}: non-finite tensor values")
        tensors[key] = (meta, a.reshape(ne[::-1]))
    if not tensors:
        raise ValueError(f"{folder}: empty trace")
    return tensors


def align(a: np.ndarray, b: np.ndarray, phase: str, prompt: int) -> tuple[np.ndarray, np.ndarray]:
    changed = [i for i, (na, nb) in enumerate(zip(a.shape, b.shape)) if na != nb]
    if len(changed) == 2 and phase == "prefill":
        # Attention scores are [batch, heads, queries, keys]. Prefill removes
        # the canvas on both axes, while decode changes only the query axis.
        if any(b.shape[i] != prompt for i in changed) or a.shape[changed[0]] != a.shape[changed[1]]:
            raise ValueError(f"cannot align prefill attention: {a.shape}, {b.shape}, prompt {prompt}")
        axis = changed[0]
        selection = [slice(0, prompt) if i in changed else slice(None) for i in range(a.ndim)]
        a = a[tuple(selection)]
    elif changed:
        if len(changed) != 1:
            raise ValueError(f"multiple dimensions differ: {a.shape}, {b.shape}")
        axis = changed[0]
        start = 0 if phase == "prefill" else prompt
        if phase == "prefill" and b.shape[axis] != prompt:
            raise ValueError(f"prefill dimension {b.shape[axis]} does not match prompt {prompt}")
        if ((phase == "decode" and a.shape[axis] != prompt + b.shape[axis])
                or start + b.shape[axis] > a.shape[axis]):
            raise ValueError(f"cannot align {phase}: {a.shape}, {b.shape}, prompt {prompt}")
        selection = [slice(None)] * a.ndim
        selection[axis] = slice(start, start + b.shape[axis])
        a = a[tuple(selection)]
    else:
        # Output row selection can remove prompt rows before result_norm.
        # Graph traces retain four dimensions, with unit outer dimensions.
        axes = [i for i, n in enumerate(a.shape[:-1]) if n > 1]
        axis = axes[0] if axes else a.ndim - 2
    if a.shape != b.shape:
        raise ValueError(f"aligned shapes differ: {a.shape}, {b.shape}")
    return np.moveaxis(a, axis, 0).reshape(a.shape[axis], -1), np.moveaxis(b, axis, 0).reshape(b.shape[axis], -1)


def compare(unified: pathlib.Path, cached: pathlib.Path, prompt: int) -> list[dict]:
    u, c = load(unified), load(cached)
    rows = []
    for (phase, name), (meta, a) in u.items():
        if phase != "unified":
            raise ValueError(f"{unified}: expected unified phase, found {phase}")
        for target in ("prefill", "decode"):
            if target == "prefill" and name.startswith(("result_norm", "result_output")):
                continue  # Prefill requests only its final prompt row, unified requests the canvas.
            key = (target, name)
            if key not in c:
                continue
            cm, b = c[key]
            aa, bb = align(a, b, target, prompt)
            metric = parity.dg_metrics(bb.astype(np.float64), aa.astype(np.float64), 0)
            rows.append(dict(name=name, phase=target, unified_ne=meta["ne"], cached_ne=cm["ne"],
                             bit_equal=bool(np.array_equal(aa.view(np.uint32), bb.view(np.uint32))),
                             max_abs_diff=float(np.max(np.abs(aa.astype(np.float64) - bb))), **metric))
    if not rows:
        raise ValueError("the traces have no matching tensor names")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("unified", type=pathlib.Path)
    ap.add_argument("cached", type=pathlib.Path)
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()
    if args.prompt <= 0:
        ap.error("--prompt must be positive")
    rows = compare(args.unified, args.cached, args.prompt)
    for r in rows:
        print(f"{r['phase']:7} {r['name']:30} equal={r['bit_equal']!s:5} "
              f"max_abs={r['max_abs_diff']:.7g} min_cos={r['min_cosine']:.8f} "
              f"argmax={r['argmax_agree']}/{r['rows']}")
    if args.out:
        args.out.write_text(json.dumps(dict(prompt_length=args.prompt, results=rows), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
