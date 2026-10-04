#!/usr/bin/env python3
"""Compare two f32 logit dumps of shape [rows, vocab].

usage: compare_logits.py A.f32 B.f32 VOCAB [--json OUT.json]

Prints, per row and overall: cosine similarity, max |diff|, RMS diff relative
to the RMS of B, argmax agreement and top-5 overlap. Exits non-zero when the
files are empty, have different sizes, or are not a whole number of rows, so
a missing dump can never read as agreement.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    a = np.fromfile(sys.argv[1], dtype=np.float32)
    b = np.fromfile(sys.argv[2], dtype=np.float32)
    vocab = int(sys.argv[3])
    if a.size == 0 or b.size == 0:
        print("refusing: an input is empty")
        return 1
    if a.size != b.size or a.size % vocab != 0:
        print(f"refusing: sizes {a.size} and {b.size} do not match vocab {vocab}")
        return 1
    a = a.reshape(-1, vocab).astype(np.float64)
    b = b.reshape(-1, vocab).astype(np.float64)
    rows = []
    for i in range(a.shape[0]):
        x, y = a[i], b[i]
        cos = float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
        top_a = set(np.argsort(-x)[:5].tolist())
        top_b = set(np.argsort(-y)[:5].tolist())
        rows.append(
            dict(
                row=i,
                cosine=cos,
                max_abs_diff=float(np.max(np.abs(x - y))),
                rel_rms=float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(y**2))),
                argmax_a=int(np.argmax(x)),
                argmax_b=int(np.argmax(y)),
                top5_overlap=len(top_a & top_b),
            )
        )
    summary = dict(
        rows=len(rows),
        min_cosine=min(r["cosine"] for r in rows),
        max_rel_rms=max(r["rel_rms"] for r in rows),
        argmax_agree=sum(r["argmax_a"] == r["argmax_b"] for r in rows),
        mean_top5_overlap=sum(r["top5_overlap"] for r in rows) / len(rows),
    )
    for r in rows:
        print(
            f"row {r['row']:3d}  cos {r['cosine']:.6f}  max|d| {r['max_abs_diff']:.4f}  "
            f"relrms {r['rel_rms']:.4f}  argmax {r['argmax_a']}/{r['argmax_b']}  top5 {r['top5_overlap']}/5"
        )
    print("summary", json.dumps(summary))
    if "--json" in sys.argv:
        out = sys.argv[sys.argv.index("--json") + 1]
        with open(out, "w") as f:
            json.dump(dict(summary=summary, rows=rows), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
