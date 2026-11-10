#!/usr/bin/env python3
"""Compare both reference paths with the independent f32 tiny-model forward.

    python bench/check_dg_tiny_paths.py --ref build/dg_ref --out-dir NEW_DIR

The existing deterministic tiny GGUF supplies all weights. Its tensors are
dequantized with the GGUF Python package into a temporary f32 model. A small
vocabulary is added so libllama can validate the already-chosen token IDs.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import subprocess
import sys

import numpy as np

import parity

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    from gguf import GGUFReader, GGUFWriter, GGUFValueType
    from gguf.quants import dequantize

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=pathlib.Path, required=True)
    ap.add_argument("--fixture", type=pathlib.Path, default=ROOT / "tests/fixtures/tiny-diffusion-gemma.gguf")
    ap.add_argument("--numpy", type=pathlib.Path, default=ROOT / "scripts/dg_numpy.py")
    ap.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir()
    model_path = args.out_dir / "tiny-f32.gguf"
    reader = GGUFReader(str(args.fixture))
    writer = GGUFWriter(str(model_path), "diffusion-gemma")
    for name, field in reader.fields.items():
        if name.startswith("GGUF.") or name == "general.architecture":
            continue
        writer.add_key_value(name, field.contents(), field.types[0],
                             sub_type=field.types[-1] if field.types[0] == GGUFValueType.ARRAY else None)
    vocab = next(t for t in reader.tensors if t.name == "token_embd.weight").shape[1]
    writer.add_tokenizer_model("llama")
    writer.add_token_list([f"token{i}" for i in range(vocab)])
    writer.add_token_types([1] * vocab)
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(1)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, dequantize(tensor.data, tensor.tensor_type).astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    spec = importlib.util.spec_from_file_location("dg_numpy", args.numpy)
    independent = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = independent
    spec.loader.exec_module(independent)
    model = independent.Model(model_path)
    results = []
    for prompt in (8, 9, 14, 16):
        tokens = independent.tiny_tokens(model, prompt, 7)
        expected = independent.forward(model, tokens, prompt, quant=False)
        expected.astype("<f4").tofile(args.out_dir / f"p{prompt}.numpy.f32")
        outputs = {}
        for path in ("unified", "cached"):
            output = args.out_dir / f"p{prompt}.{path}.f32"
            cmd = [str(args.ref.resolve()), "logits", str(model_path), ",".join(map(str, tokens[:prompt])),
                   ",".join(map(str, tokens[prompt:])), str(output), "1"]
            if path == "cached":
                cmd.append("--cached")
            with (args.out_dir / f"p{prompt}.{path}.log").open("w") as log:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            outputs[path] = np.fromfile(output, dtype="<f4").reshape(expected.shape)
            result = dict(prompt_length=prompt, path=path, reference="independent_f32",
                          max_abs_diff=float(np.max(np.abs(outputs[path].astype(np.float64) - expected))),
                          **parity.dg_metrics(outputs[path].astype(np.float64), expected.astype(np.float64), 0))
            results.append(result)
            print(json.dumps(result), flush=True)
        results.append(dict(prompt_length=prompt, path="cached", reference="unified",
                            bit_equal=bool(np.array_equal(outputs["cached"].view(np.uint32),
                                                          outputs["unified"].view(np.uint32))),
                            **parity.dg_metrics(outputs["cached"].astype(np.float64),
                                                outputs["unified"].astype(np.float64), 0)))
    (args.out_dir / "comparison.json").write_text(json.dumps(dict(seed=7, results=results), indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
