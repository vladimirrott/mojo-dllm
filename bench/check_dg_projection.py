#!/usr/bin/env python3
"""Independently recompute the first differing DiffusionGemma Q/K/V projections.

    python bench/check_dg_projection.py MODEL.gguf UNIFIED_TRACE CACHED_TRACE --prompt 9 --out projection.json

Only one weight is dequantized at a time. Five token rows cover the prompt
end, canvas start and canvas end. Products accumulate in NumPy float64 over
the GGUF package's dequantized weights, both with plain f32 activations and
with the pinned reference's Q8_K activation rounding.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

import compare_dg_traces
import parity


def q8_k_reference(x: np.ndarray) -> np.ndarray:
    """ggml-quants.c quantize_row_q8_K_ref, followed by dequantization.

    The first signed maximum fixes the scale. nearest_int rounds to nearest
    even under the default floating-point mode. Every operation before the
    final float64 matrix product is performed at f32 precision.
    """
    blocks = np.asarray(x, dtype=np.float32).reshape(len(x), -1, 256)
    index = np.abs(blocks).argmax(axis=-1, keepdims=True)
    maximum = np.take_along_axis(blocks, index, axis=-1)
    nonzero = maximum != 0
    inverse = np.divide(np.float32(-127), maximum, out=np.zeros_like(maximum), where=nonzero)
    scale = np.divide(np.float32(1), inverse, out=np.zeros_like(inverse), where=nonzero)
    rounded = np.minimum(np.rint(blocks * inverse), np.float32(127))
    return (rounded * scale).reshape(x.shape)


def main() -> int:
    from gguf import GGUFReader
    from gguf.quants import dequantize

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("unified", type=pathlib.Path)
    ap.add_argument("cached", type=pathlib.Path)
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    unified, cached = compare_dg_traces.load(args.unified), compare_dg_traces.load(args.cached)
    inputs = unified[("unified", "attn_norm-0")][1].squeeze()
    p, t = args.prompt, len(inputs)
    selected = sorted(set([0, p - 1, p, p + 1, t - 1]))
    x = inputs[selected]
    quantized = q8_k_reference(x)
    reader = GGUFReader(args.model)
    results = []
    for projection in ("q", "k", "v"):
        name = f"blk.0.attn_{projection}.weight"
        tensor = next(t for t in reader.tensors if t.name == name)
        weight = dequantize(tensor.data, tensor.tensor_type).astype(np.float64)
        plain = x.astype(np.float64) @ weight.T
        rounded = quantized.astype(np.float64) @ weight.T
        trace_name = projection.upper() + "cur-0"
        u = unified[("unified", trace_name)][1].squeeze()
        c = np.concatenate([cached[("prefill", trace_name)][1].squeeze(),
                            cached[("decode", trace_name)][1].squeeze()])
        differing_rows = np.flatnonzero(np.any(u.view(np.uint32) != c.view(np.uint32), axis=1)).tolist()
        entry = dict(weight=name, weight_type=tensor.tensor_type.name, selected_rows=selected,
                     differing_rows=differing_rows, comparisons=[])
        for path, candidate in (("unified", u[selected]), ("cached", c[selected])):
            for reference, oracle in (("plain_f32_inputs", plain), ("q8_k_inputs", rounded)):
                error = candidate.astype(np.float64) - oracle
                entry["comparisons"].append(dict(path=path, reference=reference,
                    max_abs_diff=float(np.abs(error).max()), rms_diff=float(np.sqrt(np.mean(error ** 2))),
                    **parity.dg_metrics(candidate.astype(np.float64), oracle, 0)))
        results.append(entry)
        print(json.dumps(entry), flush=True)
        del weight
    args.out.write_text(json.dumps(dict(prompt_length=p, results=results), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
