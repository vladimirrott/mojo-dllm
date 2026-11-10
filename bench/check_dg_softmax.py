#!/usr/bin/env python3
"""Replay the first-layer prompt softmax using only scores and a causal mask.

    python bench/check_dg_softmax.py UNIFIED_TRACE CACHED_TRACE --prompt 8 --llama-bin LLAMA/build/bin --out softmax.json

The CPU libraries are the same pinned Linux shared libraries as dg_ref.
Only the key dimension changes. Added keys are masked to negative infinity.
NumPy float64 supplies an independent softmax over the original prompt keys.
"""

from __future__ import annotations

import argparse
import ctypes as ct
import json
import pathlib

import numpy as np

import compare_dg_traces


class InitParams(ct.Structure):
    _fields_ = [("mem_size", ct.c_size_t), ("mem_buffer", ct.c_void_p), ("no_alloc", ct.c_bool)]


class Softmax:
    def __init__(self, folder: pathlib.Path):
        for name in ("libggml-base.so", "libggml-cpu.so", "libggml.so"):
            ct.CDLL(str(folder / name), mode=ct.RTLD_GLOBAL)
        api = ct.CDLL(None)
        pointer, i64, size = ct.c_void_p, ct.c_int64, ct.c_size_t
        signatures = {
            "ggml_init": (pointer, [InitParams]),
            "ggml_new_tensor_4d": (pointer, [pointer, ct.c_int, i64, i64, i64, i64]),
            "ggml_soft_max_ext": (pointer, [pointer, pointer, pointer, ct.c_float, ct.c_float]),
            "ggml_new_graph": (pointer, [pointer]),
            "ggml_build_forward_expand": (None, [pointer, pointer]),
            "ggml_backend_cpu_init": (pointer, []),
            "ggml_backend_cpu_set_n_threads": (None, [pointer, ct.c_int]),
            "ggml_backend_alloc_ctx_tensors": (pointer, [pointer, pointer]),
            "ggml_backend_tensor_set": (None, [pointer, pointer, size, size]),
            "ggml_backend_tensor_get": (None, [pointer, pointer, size, size]),
            "ggml_backend_graph_compute": (ct.c_int, [pointer, pointer]),
            "ggml_backend_buffer_free": (None, [pointer]),
            "ggml_backend_free": (None, [pointer]),
            "ggml_free": (None, [pointer]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(api, name)
            function.restype, function.argtypes = result, arguments
        self.api = api

    def run(self, scores: np.ndarray, keys: int) -> np.ndarray:
        api = self.api
        heads, queries, prompt = scores.shape
        values = np.zeros((heads, queries, keys), dtype=np.float32)
        values[:, :, :prompt] = scores
        mask = np.full((queries, keys), -np.inf, dtype=np.float32)
        for q in range(queries):
            mask[q, :q + 1] = 0
        context = api.ggml_init(InitParams(8 * 1024 * 1024, None, True))
        source = api.ggml_new_tensor_4d(context, 0, keys, queries, heads, 1)
        masking = api.ggml_new_tensor_4d(context, 0, keys, queries, 1, 1)
        output = api.ggml_soft_max_ext(context, source, masking, 1, 0)
        graph = api.ggml_new_graph(context)
        api.ggml_build_forward_expand(graph, output)
        backend = api.ggml_backend_cpu_init()
        api.ggml_backend_cpu_set_n_threads(backend, 1)
        buffer = api.ggml_backend_alloc_ctx_tensors(context, backend)
        api.ggml_backend_tensor_set(source, values.ctypes.data, 0, values.nbytes)
        api.ggml_backend_tensor_set(masking, mask.ctypes.data, 0, mask.nbytes)
        if api.ggml_backend_graph_compute(backend, graph) != 0:
            raise RuntimeError("isolated CPU softmax failed")
        result = np.empty_like(values)
        api.ggml_backend_tensor_get(output, result.ctypes.data, 0, result.nbytes)
        api.ggml_backend_buffer_free(buffer)
        api.ggml_backend_free(backend)
        api.ggml_free(context)
        return result[:, :, :prompt].copy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("unified", type=pathlib.Path)
    ap.add_argument("cached", type=pathlib.Path)
    ap.add_argument("--prompt", type=int, required=True)
    ap.add_argument("--llama-bin", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    p = args.prompt
    unified, cached = compare_dg_traces.load(args.unified), compare_dg_traces.load(args.cached)
    scores = unified[("unified", "kq-0")][1].squeeze()[:, :p, :p]
    total = unified[("unified", "kq-0")][1].shape[-1]
    actual = {
        p: cached[("prefill", "kq_soft_max-0")][1].squeeze(),
        total: unified[("unified", "kq_soft_max-0")][1].squeeze()[:, :p, :p],
    }
    causal = np.arange(p)[None, :] <= np.arange(p)[:, None]
    masked = np.where(causal[None], scores.astype(np.float64), -np.inf)
    exponential = np.exp(masked - masked.max(-1, keepdims=True))
    oracle = exponential / exponential.sum(-1, keepdims=True)
    kernel = Softmax(args.llama_bin.resolve())
    results = []
    short = None
    for keys in sorted(set([p, 16, 32, total])):
        output = kernel.run(scores, keys)
        if short is None:
            short = output
        result = dict(key_length=keys, max_abs_from_short=float(np.max(np.abs(output - short))),
                      max_abs_from_f64=float(np.max(np.abs(output.astype(np.float64) - oracle))),
                      rms_from_f64=float(np.sqrt(np.mean((output.astype(np.float64) - oracle) ** 2))))
        if keys in actual:
            result["replay_bit_equal"] = bool(np.array_equal(output.view(np.uint32), actual[keys].copy().view(np.uint32)))
        results.append(result)
        print(json.dumps(result), flush=True)
    args.out.write_text(json.dumps(dict(prompt_length=p, results=results), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
