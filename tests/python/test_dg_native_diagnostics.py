"""Small CPU reference-tool checks, enabled by DG_REF and DG_MATMUL paths."""

import os
from pathlib import Path
import subprocess

import pytest

np = pytest.importorskip("numpy")


def native_tool(variable):
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"set {variable} to the pinned CPU reference executable")
    executable = Path(value)
    assert executable.is_file(), f"{variable} does not name a file: {executable}"
    return str(executable)


@pytest.fixture
def matrix_case(tmp_path):
    inputs = np.arange(1, 33, dtype=np.float32).reshape(8, 4)
    weight = np.vstack([np.eye(4, dtype=np.float32), np.eye(4, dtype=np.float32) * 2])
    weight_path, input_path, output = tmp_path / "weight.bin", tmp_path / "input.f32", tmp_path / "output.f32"
    weight.tofile(weight_path)
    inputs.tofile(input_path)
    command = [native_tool("DG_MATMUL"), "0", "4", "8", str(weight_path), str(input_path),
               "8", str(output), "1", "--no-repack"]
    return command, weight_path, input_path, output, inputs


def test_native_matmul_small_f32_roundtrip(matrix_case):
    command, _, _, output, inputs = matrix_case
    done = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert done.returncode == 0, done.stderr
    actual = np.fromfile(output, dtype="<f4").reshape(8, 8)
    np.testing.assert_array_equal(actual, np.concatenate([inputs, inputs * 2], axis=1))


@pytest.mark.parametrize("type_id", [4, 5])
def test_native_matmul_rejects_deprecated_type_without_dividing_by_zero(matrix_case, type_id):
    command, _, _, output, _ = matrix_case
    command[1] = str(type_id)
    done = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert done.returncode == 1, done.stderr
    assert f"unsupported type id {type_id}" in done.stderr
    assert not output.exists()


@pytest.mark.parametrize("which", ["weight", "input"])
@pytest.mark.parametrize("damage", ["short", "extra"])
def test_native_matmul_rejects_inexact_tensor_file(matrix_case, which, damage):
    command, weight, inputs, _, _ = matrix_case
    path = weight if which == "weight" else inputs
    contents = path.read_bytes()
    path.write_bytes(contents[:-1] if damage == "short" else contents + b"\0")
    done = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert done.returncode == 2, done.stderr
    assert str(path) in done.stderr
    assert "must hold exactly" in done.stderr


def trace_command(tmp_path):
    return [native_tool("DG_REF"), "logits", str(tmp_path / "missing.gguf"), "1", "1",
            str(tmp_path / "output.f32"), "1"]


def test_native_trace_only_requires_trace_dir(tmp_path):
    done = subprocess.run(trace_command(tmp_path) + ["--trace-only", "Qcur-0"],
                          capture_output=True, text=True, timeout=20)
    assert done.returncode == 1, done.stderr
    assert "--trace-only requires --trace-dir" in done.stderr


def test_native_trace_directory_must_be_new(tmp_path):
    trace = tmp_path / "trace"
    trace.mkdir()
    sentinel = trace / "manifest.jsonl"
    sentinel.write_text("existing evidence\n")
    done = subprocess.run(trace_command(tmp_path) + ["--trace-dir", str(trace)],
                          capture_output=True, text=True, timeout=20)
    assert done.returncode == 2, done.stderr
    assert "trace directory must be new" in done.stderr
    assert sentinel.read_text() == "existing evidence\n"


def test_native_trace_rejects_empty_prefixes(tmp_path):
    done = subprocess.run(trace_command(tmp_path) + ["--trace-dir", str(tmp_path / "trace"),
                          "--trace-only", ",,,"], capture_output=True, text=True, timeout=20)
    assert done.returncode == 2, done.stderr
    assert "--trace-only needs a tensor-name prefix" in done.stderr
