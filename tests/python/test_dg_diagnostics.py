"""Diagnostic reports must expose wrong results and preserve operator inputs.

Small literal tensors exercise the command entry points. The child-process
and shared-library boundaries are controlled, so these tests run with the
fixtures group's NumPy and do not require a model or native reference build.
"""

import ctypes
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "bench" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


projection = load_module("check_dg_projection")
matmul = load_module("check_dg_matmul")
softmax = load_module("check_dg_softmax")
tiny_paths = load_module("check_dg_tiny_paths")


def write_trace(folder, phases):
    folder.mkdir()
    records = []
    for phase, tensors in phases.items():
        for name, values in tensors.items():
            filename = f"{phase}-{len(records)}.f32"
            values.astype("<f4").tofile(folder / filename)
            records.append(dict(phase=phase, name=name, file=filename, ne=list(values.shape[::-1])))
    (folder / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def install_gguf_double(monkeypatch, tensors):
    gguf = ModuleType("gguf")
    gguf.GGUFReader = lambda path: SimpleNamespace(tensors=tensors)
    quants = ModuleType("gguf.quants")
    quants.dequantize = lambda data, kind: data.copy()
    monkeypatch.setitem(sys.modules, "gguf", gguf)
    monkeypatch.setitem(sys.modules, "gguf.quants", quants)


def test_projection_reports_quantized_oracle_and_boundary_differences(tmp_path, monkeypatch):
    prompt, tokens, hidden = 9, 25, 256
    inputs = np.zeros((tokens, hidden), dtype=np.float32)
    inputs[:, :3] = [127, 0.5, -1.5]
    inputs[:, 3] = np.arange(1, tokens + 1)
    weight = np.eye(hidden, dtype=np.float32)
    weight[0, 1], weight[1, 0] = 3, 2
    # Literal results for nearest-even Q8_K rounding with scale 1.
    expected = inputs.copy()
    expected[:, :3] = [127, 254, -2]
    cached = expected.copy()
    cached[prompt - 1, 0] += 0.25
    names = ["Qcur-0", "Kcur-0", "Vcur-0"]
    unified_tensors = {name: expected[None, None] for name in names}
    unified_tensors["attn_norm-0"] = inputs[None, None]
    write_trace(tmp_path / "unified", {"unified": unified_tensors})
    write_trace(tmp_path / "cached", {
        "prefill": {name: cached[None, None, :prompt] for name in names},
        "decode": {name: cached[None, None, prompt:] for name in names},
    })
    install_gguf_double(monkeypatch, [
        SimpleNamespace(name=f"blk.0.attn_{name}.weight", data=weight,
                        tensor_type=SimpleNamespace(name="Q4_K")) for name in "qkv"
    ])
    output = tmp_path / "projection.json"
    monkeypatch.setattr(sys, "argv", ["check_dg_projection.py", "model.gguf",
        str(tmp_path / "unified"), str(tmp_path / "cached"), "--prompt", "9", "--out", str(output)])
    assert projection.main() == 0
    results = json.loads(output.read_text())["results"]
    assert [r["weight"] for r in results] == [f"blk.0.attn_{name}.weight" for name in "qkv"]
    for result in results:
        assert result["selected_rows"] == [0, 8, 9, 10, 24]
        assert result["differing_rows"] == [8]
        comparisons = {(r["path"], r["reference"]): r for r in result["comparisons"]}
        assert comparisons["unified", "q8_k_inputs"]["max_abs_diff"] == 0
        assert comparisons["cached", "q8_k_inputs"]["max_abs_diff"] == 0.25
        assert comparisons["unified", "plain_f32_inputs"]["max_abs_diff"] == 1.5


def test_matmul_reports_replay_mismatch_and_honors_no_repack(tmp_path, monkeypatch):
    inputs = np.tile(np.array([1, 2, 3, 4], dtype=np.float32), (25, 1))
    inputs[:, 0] = np.arange(1, 26)
    weight = np.vstack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32) * 2])
    original = inputs @ weight.T
    write_trace(tmp_path / "unified", {"unified": {
        "attn_norm-0": inputs[None, None], "Qcur-0": original[None, None],
    }})
    install_gguf_double(monkeypatch, [SimpleNamespace(name="blk.0.attn_q.weight", shape=[4, 8],
        data=weight, tensor_type=SimpleNamespace(value=0, name="F32"))])
    calls = []

    def run_product(command, **kwargs):
        calls.append(command)
        k, n, rows = int(command[2]), int(command[3]), int(command[6])
        w = np.fromfile(command[4], dtype="<f4").reshape(n, k)
        x = np.fromfile(command[5], dtype="<f4").reshape(rows, k)
        result = x @ w.T
        if "--no-repack" not in command and rows % 4:
            result[-1, 0] += 0.125
        result.astype("<f4").tofile(command[7])
        return subprocess.CompletedProcess(command, 0, "", "controlled CPU matrix output")

    monkeypatch.setattr(matmul.subprocess, "run", run_product)
    output = tmp_path / "replay"
    monkeypatch.setattr(sys, "argv", ["check_dg_matmul.py", "model.gguf", str(tmp_path / "unified"),
        "--prompt", "9", "--matmul", str(tmp_path / "dg_matmul"), "--out-dir", str(output)])
    assert matmul.main() == 0
    report = json.loads((output / "comparison.json").read_text())
    assert report["original_replay_bit_equal"] is False
    assert report["original_replay_max_abs"] == 0.125
    assert len(calls) == 31
    assert sum("--no-repack" in command for command in calls) == 15
    results = {(r["prompt_length"], r["no_repack"]): r for r in report["tests"]}
    assert results[9, False]["differing_rows"] == [8, 24]
    assert results[14, False]["differing_rows"] == [13, 29]
    for prompt in [8, 9, 12, 14, 16]:
        assert results[prompt, True]["differing_rows"] == []
        assert results[prompt, True]["max_abs_diff"] == 0


def causal_softmax(scores):
    heads, queries, keys = scores.shape
    result = np.zeros_like(scores)
    for head in range(heads):
        for query in range(queries):
            allowed = [float(scores[head, query, key]) for key in range(min(query + 1, keys))]
            maximum = max(allowed)
            numerators = [math.exp(value - maximum) for value in allowed]
            result[head, query, :len(allowed)] = np.array(numerators) / math.fsum(numerators)
    return result


def test_softmax_reports_f64_error_and_replay_mismatch(tmp_path, monkeypatch):
    prompt, total = 8, 24
    head, query, key = np.indices((2, total, total))
    scores = ((head + 1) * (query + 1) * (key + 1) / 32).astype(np.float32)
    expected = causal_softmax(scores[:, :prompt, :prompt])
    unified = np.zeros_like(scores)
    unified[:, :prompt, :prompt] = expected
    cached = expected.copy()
    cached[0, 0, 0] += 0.125
    write_trace(tmp_path / "unified", {"unified": {
        "kq-0": scores[None], "kq_soft_max-0": unified[None],
    }})
    write_trace(tmp_path / "cached", {"prefill": {"kq_soft_max-0": cached[None]}})
    kernel = SimpleNamespace(run=lambda values, keys: causal_softmax(values))
    monkeypatch.setattr(softmax, "Softmax", lambda folder: kernel)
    output = tmp_path / "softmax.json"
    monkeypatch.setattr(sys, "argv", ["check_dg_softmax.py", str(tmp_path / "unified"),
        str(tmp_path / "cached"), "--prompt", "8", "--llama-bin", str(tmp_path), "--out", str(output)])
    assert softmax.main() == 0
    results = json.loads(output.read_text())["results"]
    assert [r["key_length"] for r in results] == [8, 16, 24, 32]
    assert all(r["max_abs_from_f64"] < 1e-7 for r in results)
    assert results[0]["replay_bit_equal"] is False
    assert results[2]["replay_bit_equal"] is True


def test_softmax_reports_one_ulp_shape_difference(tmp_path, monkeypatch):
    prompt, total = 8, 24
    scores = np.zeros((2, total, total), dtype="<f4")
    short = causal_softmax(scores[:, :prompt, :prompt])
    longer = short.copy()
    longer.view(np.uint32)[0, 0, 0] += 1
    full = np.zeros_like(scores)
    full[:, :prompt, :prompt] = longer
    write_trace(tmp_path / "unified", {"unified": {"kq-0": scores[None], "kq_soft_max-0": full[None]}})
    write_trace(tmp_path / "cached", {"prefill": {"kq_soft_max-0": short[None]}})
    monkeypatch.setattr(softmax, "Softmax", lambda folder: SimpleNamespace(
        run=lambda values, keys: short.copy() if keys == prompt else longer.copy()))
    output = tmp_path / "softmax.json"
    monkeypatch.setattr(sys, "argv", ["check_dg_softmax.py", str(tmp_path / "unified"),
        str(tmp_path / "cached"), "--prompt", "8", "--llama-bin", str(tmp_path), "--out", str(output)])
    assert softmax.main() == 0
    results = json.loads(output.read_text())["results"]
    one_ulp = float(np.spacing(np.float32(1)))
    assert [r["max_abs_from_short"] for r in results] == [0, one_ulp, one_ulp, one_ulp]
    assert [r["max_abs_from_f64"] for r in results][1:] == [one_ulp] * 3
    assert results[0]["replay_bit_equal"] is True
    assert results[2]["replay_bit_equal"] is True


def test_tiny_paths_executes_both_paths_and_reports_one_ulp_difference(tmp_path, monkeypatch):
    tensor = SimpleNamespace(name="token_embd.weight", shape=[2, 8],
                             data=np.ones((8, 2), dtype=np.float32), tensor_type=0)
    install_gguf_double(monkeypatch, [tensor])
    sys.modules["gguf"].GGUFReader = lambda path: SimpleNamespace(fields={}, tensors=[tensor])
    sys.modules["gguf"].GGUFValueType = SimpleNamespace(ARRAY=9)
    writes = []

    class Writer:
        def __init__(self, path, architecture):
            assert architecture == "diffusion-gemma"
            self.path = Path(path)

        def add_tensor(self, name, values):
            writes.append((name, values.copy()))

        def close(self):
            self.path.write_bytes(b"controlled f32 fixture")

        def __getattr__(self, name):
            assert name.startswith(("add_", "write_")), name
            return lambda *args, **kwargs: None

    sys.modules["gguf"].GGUFWriter = Writer
    independent = tmp_path / "independent.py"
    independent.write_text(
        "import numpy as np\n"
        "class Model:\n"
        "    def __init__(self, path):\n"
        "        assert path.read_bytes() == b'controlled f32 fixture'\n"
        "def tiny_tokens(model, prompt, seed):\n"
        "    assert seed == 7\n"
        "    return np.arange(prompt + 4) % 8\n"
        "def forward(model, tokens, prompt, quant):\n"
        "    assert quant is False\n"
        "    return np.tile(np.arange(1, 9, dtype=np.float32), (4, 1))\n"
    )
    commands = []

    def run_reference(command, **kwargs):
        commands.append(command)
        values = np.tile(np.arange(1, 9, dtype="<f4"), (4, 1))
        if "--cached" in command:
            values.view(np.uint32)[0, 0] += 1
        values.tofile(command[5])
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(tiny_paths.subprocess, "run", run_reference)
    monkeypatch.setitem(sys.modules, "dg_numpy", ModuleType("dg_numpy"))
    output = tmp_path / "tiny-paths"
    monkeypatch.setattr(sys, "argv", ["check_dg_tiny_paths.py", "--ref", str(tmp_path / "dg_ref"),
        "--numpy", str(independent), "--out-dir", str(output)])
    assert tiny_paths.main() == 0
    results = json.loads((output / "comparison.json").read_text())["results"]
    assert len(commands) == 8
    assert ["--cached" in command for command in commands] == [False, True] * 4
    assert [(r["prompt_length"], r["path"], r["reference"]) for r in results] == [
        (prompt, path, reference) for prompt in (8, 9, 14, 16)
        for path, reference in (("unified", "independent_f32"), ("cached", "independent_f32"), ("cached", "unified"))
    ]
    for start in range(0, 12, 3):
        assert results[start]["max_abs_diff"] == 0
        assert results[start + 1]["max_abs_diff"] == float(np.spacing(np.float32(1)))
        assert results[start + 1]["argmax_agree"] == 4
        assert results[start + 2]["bit_equal"] is False
    assert writes[0][0] == "token_embd.weight"
    assert writes[0][1].dtype == np.float32


def softmax_with_recorded_api(compute_status=0):
    uploads, tensor_shapes = {}, {}
    tensor_ids = iter([10, 11])

    def new_tensor(context, kind, keys, queries, heads, batch):
        tensor = next(tensor_ids)
        tensor_shapes[tensor] = (batch, heads, queries, keys)
        return tensor

    def upload(tensor, address, offset, size):
        uploads[tensor] = np.frombuffer(ctypes.string_at(address, size), dtype="<f4").copy()

    def download(tensor, address, offset, size):
        values = np.full(size // 4, 0.25, dtype=np.float32)
        ctypes.memmove(address, values.ctypes.data, size)

    api = SimpleNamespace(
        ggml_init=lambda params: 1, ggml_new_tensor_4d=new_tensor,
        ggml_soft_max_ext=lambda *args: 12, ggml_new_graph=lambda context: 2,
        ggml_build_forward_expand=lambda *args: None, ggml_backend_cpu_init=lambda: 3,
        ggml_backend_cpu_set_n_threads=lambda *args: None,
        ggml_backend_alloc_ctx_tensors=lambda *args: 4,
        ggml_backend_tensor_set=upload, ggml_backend_tensor_get=download,
        ggml_backend_graph_compute=lambda *args: compute_status,
        ggml_backend_buffer_free=lambda buffer: None, ggml_backend_free=lambda backend: None,
        ggml_free=lambda context: None,
    )
    kernel = softmax.Softmax.__new__(softmax.Softmax)
    kernel.api = api
    return kernel, uploads, tensor_shapes


def test_softmax_uploads_causal_mask_and_masked_padding():
    kernel, uploads, shapes = softmax_with_recorded_api()
    scores = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    output = kernel.run(scores, 16)
    assert shapes == {10: (1, 2, 3, 16), 11: (1, 1, 3, 16)}
    values = uploads[10].reshape(2, 3, 16)
    np.testing.assert_array_equal(values[:, :, :3], scores)
    np.testing.assert_array_equal(values[:, :, 3:], np.zeros((2, 3, 13)))
    mask = uploads[11].reshape(3, 16)
    expected = np.full((3, 16), -np.inf, dtype=np.float32)
    expected[0, 0] = 0
    expected[1, :2] = 0
    expected[2, :3] = 0
    np.testing.assert_array_equal(mask, expected)
    assert output.shape == (2, 3, 3)


def test_softmax_propagates_backend_compute_failure():
    kernel, _, _ = softmax_with_recorded_api(compute_status=7)
    with pytest.raises(RuntimeError, match="isolated CPU softmax failed"):
        kernel.run(np.ones((2, 3, 3), dtype=np.float32), 16)
