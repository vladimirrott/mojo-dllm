"""Shape alignment and damaged-input checks for the optional graph diagnostic."""

import importlib.util
import json
import pathlib
import sys

import pytest

np = pytest.importorskip("numpy")
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
spec = importlib.util.spec_from_file_location("compare_dg_traces", ROOT / "bench/compare_dg_traces.py")
traces = importlib.util.module_from_spec(spec)
spec.loader.exec_module(traces)
spec = importlib.util.spec_from_file_location("check_dg_projection", ROOT / "bench/check_dg_projection.py")
projection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(projection)


@pytest.mark.parametrize("prompt", [8, 9, 14, 16])
@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_aligns_the_token_axis_of_a_head_tensor(phase, prompt):
    unified = np.arange((prompt + 256) * 4 * 8, dtype=np.float32).reshape(1, prompt + 256, 4, 8)
    expected = slice(0, prompt) if phase == "prefill" else slice(prompt, prompt + 256)
    cached = unified[:, expected].copy()
    a, b = traces.align(unified, cached, phase, prompt)
    assert a.shape == (cached.shape[1], 32)
    np.testing.assert_array_equal(a, b)


def test_rejects_a_wrong_canvas_length():
    with pytest.raises(ValueError, match="cannot align decode"):
        traces.align(np.ones((1, 1, 265, 8)), np.ones((1, 1, 255, 8)), "decode", 9)


def test_aligns_both_query_and_key_axes_of_prefill_attention():
    unified = np.arange(4 * 265 * 265, dtype=np.float32).reshape(1, 4, 265, 265)
    cached = unified[:, :, :9, :9].copy()
    a, b = traces.align(unified, cached, "prefill", 9)
    assert a.shape == (9, 4 * 9)
    np.testing.assert_array_equal(a, b)


def write_trace(folder, values, entries=1):
    values.astype("<f4").tofile(folder / "l_out-0.f32")
    record = dict(phase="unified", name="l_out-0", file="l_out-0.f32", ne=[8, 2, 1, 1])
    (folder / "manifest.jsonl").write_text((json.dumps(record) + "\n") * entries)


def test_rejects_duplicate_names_in_one_phase(tmp_path):
    write_trace(tmp_path, np.ones(16), entries=2)
    with pytest.raises(ValueError, match="repeated tensor"):
        traces.load(tmp_path)


def test_rejects_a_short_tensor(tmp_path):
    write_trace(tmp_path, np.ones(15))
    with pytest.raises(ValueError, match="l_out-0.f32.*wrong size"):
        traces.load(tmp_path)


def test_rejects_non_finite_tensor_values(tmp_path):
    values = np.ones(16)
    values[3] = np.nan
    write_trace(tmp_path, values)
    with pytest.raises(ValueError, match="l_out-0.f32.*non-finite"):
        traces.load(tmp_path)


def test_rejects_an_empty_trace(tmp_path):
    (tmp_path / "manifest.jsonl").write_text("")
    with pytest.raises(ValueError, match="empty trace"):
        traces.load(tmp_path)


def test_load_preserves_logical_ggml_dimension_order(tmp_path):
    write_trace(tmp_path, np.arange(16))
    _, values = traces.load(tmp_path)[("unified", "l_out-0")]
    assert values.shape == (1, 1, 2, 8)
    np.testing.assert_array_equal(values[0, 0, 1], np.arange(8, 16))


def test_compare_preserves_order_alignment_and_one_ulp_differences(tmp_path):
    unified, cached = tmp_path / "unified", tmp_path / "cached"
    unified.mkdir()
    cached.mkdir()
    records = {unified: [], cached: []}

    def add(folder, phase, name, values):
        filename = f"{phase}-{name}.f32"
        values.tofile(folder / filename)
        records[folder].append(dict(phase=phase, name=name, file=filename, ne=list(values.shape[::-1])))

    values = np.ones((1, 1, 3, 8), dtype="<f4")
    for name in ("attn_norm-0", "Qcur-0"):
        add(unified, "unified", name, values)
        prefill = values[:, :, :1].copy()
        decode = values[:, :, 1:].copy()
        if name == "Qcur-0":
            prefill.view(np.uint32)[0, 0, 0, 3] += 1
        add(cached, "prefill", name, prefill)
        add(cached, "decode", name, decode)
    for folder, entries in records.items():
        (folder / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in entries))

    results = traces.compare(unified, cached, prompt=1)
    assert [(r["name"], r["phase"]) for r in results] == [
        ("attn_norm-0", "prefill"), ("attn_norm-0", "decode"),
        ("Qcur-0", "prefill"), ("Qcur-0", "decode"),
    ]
    assert [r["bit_equal"] for r in results] == [True, True, False, True]
    assert [r["rows"] for r in results] == [1, 2, 1, 2]
    assert results[2]["max_abs_diff"] == float(np.spacing(np.float32(1)))
    assert results[2]["unified_ne"] == [8, 3, 1, 1]
    assert results[2]["cached_ne"] == [8, 1, 1, 1]


def test_compare_distinguishes_signed_zero_bits(tmp_path, monkeypatch):
    positive = np.ones((1, 1, 2, 8), dtype="<f4")
    negative = positive.copy()
    positive[0, 0, 0, 0] = 0
    negative[0, 0, 0, 0] = -0.0
    unified, cached = tmp_path / "unified", tmp_path / "cached"
    monkeypatch.setattr(traces, "load", lambda folder: {
        ("unified" if folder == unified else "decode", "result_output"):
            (dict(ne=[8, 2, 1, 1]), positive if folder == unified else negative)
    })
    result, = traces.compare(unified, cached, prompt=1)
    assert result["bit_equal"] is False
    assert result["max_abs_diff"] == 0


def test_rejects_a_prefill_dimension_that_is_not_the_prompt():
    with pytest.raises(ValueError, match="prefill dimension 8 does not match prompt 9"):
        traces.align(np.ones((1, 1, 265, 8)), np.ones((1, 1, 8, 8)), "prefill", 9)


@pytest.mark.parametrize("unified_shape,cached_shape", [
    ((1, 2, 265, 265), (1, 2, 9, 8)),
    ((1, 2, 265, 266), (1, 2, 9, 9)),
])
def test_rejects_invalid_prefill_attention_axes(unified_shape, cached_shape):
    with pytest.raises(ValueError, match="cannot align prefill attention"):
        traces.align(np.ones(unified_shape), np.ones(cached_shape), "prefill", 9)


def test_q8_k_reference_rounds_half_to_even_and_preserves_zero_rows():
    values = np.zeros((2, 256), dtype=np.float32)
    values[0, :4] = [127, 0.5, -1.5, -2.5]
    rounded = projection.q8_k_reference(values)
    np.testing.assert_array_equal(rounded[0, :4], [127, 0, -2, -2])
    np.testing.assert_array_equal(rounded[1], np.zeros(256))


def test_q8_k_reference_scales_each_block_independently():
    values = np.zeros((1, 512), dtype=np.float32)
    values[0, :2] = [127, 0.5]
    values[0, 256:258] = [63.5, 1.25]
    rounded = projection.q8_k_reference(values)
    np.testing.assert_array_equal(rounded[0, :2], [127, 0])
    np.testing.assert_array_equal(rounded[0, 256:258], [63.5, 1])


def test_reference_source_record_identifies_the_overlay_and_verified_pin():
    evidence = json.loads((ROOT / "bench/checks/2026-11-05-dg-reference-paths.json").read_text())
    source = json.loads((ROOT / "bench/checks" / evidence["reference_source_manifest"]).read_text())
    pin = evidence["llama_cpp_commit"]
    assert source["commit"] == pin
    assert source["manifest"]["pin"] == pin
    assert source["manifest"]["base"] == "a34de8618b25dfc32ee43749db47d767a9f0a25d"
    assert source["manifest"]["files"]
    verification = source["compilation_input_verification"]
    assert verification["source_pin"] == pin
    assert verification["compilation_inputs_checked"] == verification["matching"] == 403
    assert verification["differences"] == []
