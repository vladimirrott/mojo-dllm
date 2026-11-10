"""The sensitivity experiment must compare like inputs and retain its controls."""

from __future__ import annotations

import importlib.util
import json
import re
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench"))
spec = importlib.util.spec_from_file_location("dg_sensitivity", ROOT / "bench/dg_sensitivity.py")
assert spec is not None and spec.loader is not None
sensitivity = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sensitivity
spec.loader.exec_module(sensitivity)


def test_suite_has_more_than_two_distinct_prompts() -> None:
    assert len(sensitivity.PROMPTS) >= 4
    assert len(set(sensitivity.PROMPTS)) == len(sensitivity.PROMPTS)


def test_original_plan_has_path_and_repeat_controls_in_both_states(tmp_path: Path) -> None:
    jobs = sensitivity.forward_plan(tmp_path, tmp_path, "12", mutant=False)
    by_name = {job.name: job for job in jobs}
    assert set(by_name) == {
        "unified", "cached", "unified-repeat", "unified-sc", "cached-sc", "unified-sc-repeat", "unified-sc-noisy"
    }
    assert by_name["unified"].previous is None
    assert by_name["unified-repeat"].previous is None
    assert by_name["cached"].cached
    for name in ("unified-sc", "cached-sc", "unified-sc-repeat"):
        assert by_name[name].previous == tmp_path / "unified.ref.f32"
    assert by_name["unified-sc-noisy"].previous == tmp_path / "prev-noisy.f32"


def test_mutant_self_conditioning_uses_original_previous_step(tmp_path: Path) -> None:
    original, mutant = tmp_path / "original", tmp_path / "mutant"
    jobs = sensitivity.forward_plan(mutant, original, "12", mutant=True)
    assert {job.name for job in jobs} == {"unified", "cached", "unified-sc", "cached-sc"}
    assert all(job.previous == original / "unified.ref.f32" for job in jobs if job.name.endswith("-sc"))
    assert all(job.output.parent == mutant for job in jobs)


def test_case_identity_includes_canvas_and_previous_temperature() -> None:
    meta = dict(prompt="same", prompt_tokens=[2, 3], canvas_tokens=[4, 5], n_vocab=7, prev_temp_inv=1.25)
    sensitivity.require_same_inputs(meta, dict(meta), "case0")
    for key, wrong in (("prompt_tokens", [2, 4]), ("canvas_tokens", [5, 4]), ("prev_temp_inv", 1.0)):
        try:
            sensitivity.require_same_inputs(meta, dict(meta, **{key: wrong}), "case0")
        except SystemExit as exc:
            assert key in str(exc) and "case0" in str(exc)
        else:
            raise AssertionError(f"{key} mismatch was accepted")


def test_reconstructed_source_is_recorded_as_a_manifest_not_git_head(tmp_path: Path) -> None:
    manifest = tmp_path / "source.json"
    source = dict(pin=sensitivity.parity.dg_pinned_commit(ROOT), mechanism="copy and overlay pinned files",
                  files=[dict(path="src/model.cpp", github_blob="source-id")])
    manifest.write_text(json.dumps(source))
    identity = sensitivity.reference_source(tmp_path, manifest)
    assert identity["manifest"] == source
    assert "reconstructed" in identity["mechanism"]
    manifest.write_text(json.dumps(dict(source, pin="0" * 40)))
    try:
        sensitivity.reference_source(tmp_path, manifest)
    except SystemExit as exc:
        assert str(manifest) in str(exc)
    else:
        raise AssertionError("a manifest for another commit was accepted")


def test_input_perturbations_do_not_set_the_implementation_bound() -> None:
    small = dict(min_cosine=0.99, argmax_agree=9, rows=10, mean_top5_overlap=4.5,
                 max_rel_rms=0.1, top1_max_rel_err=0.02)
    noisy = dict(small, min_cosine=0.1, max_rel_rms=2.0)
    records = [dict(small, state="no-sc", kind="repeat", same_inputs=True),
               dict(small, state="sc", kind="cache-path", same_inputs=True),
               dict(noisy, state="sc", kind="previous-logit-noise", same_inputs=False)]
    envelopes = sensitivity.same_input_envelopes(records)
    assert envelopes["sc"]["min_cosine"] == 0.99
    assert envelopes["sc"]["max_rel_rms"] == 0.1
    assert envelopes["no-sc"]["max_top1_rel_err"] == 0.02


def test_missing_controls_are_an_error_not_an_unbounded_state() -> None:
    try:
        sensitivity.same_input_envelopes([])
    except SystemExit as exc:
        assert "no-sc" in str(exc)
    else:
        raise AssertionError("missing non-self-conditioned controls were accepted")


def test_mutant_with_its_own_previous_logits_is_refused(tmp_path: Path) -> None:
    meta = dict(prompt="same", prompt_tokens=[2], canvas_tokens=[3], n_vocab=4, prev_temp_inv=1.25)
    previous = tmp_path / "original" / "unified.ref.f32"
    sensitivity.require_mutant_inputs(meta, dict(meta, prev_file=str(previous)), "case0", previous)
    try:
        sensitivity.require_mutant_inputs(meta, dict(meta, prev_file=str(tmp_path / "mutant" / previous.name)),
                                          "case0", previous)
    except SystemExit as exc:
        assert "original reference's previous" in str(exc)
    else:
        raise AssertionError("the mutant's own previous-step logits were accepted")


def test_documented_spread_is_rendered_from_the_measurement_record() -> None:
    import check_dg_bound

    evidence = json.loads((ROOT / sensitivity.parity.DG_SENSITIVITY_EVIDENCE).read_text(encoding="utf-8"))
    docs = (ROOT / "docs/correctness.md").read_text(encoding="utf-8")
    section = docs.split("<!-- dg-sensitivity:start -->\n", 1)[1].split("<!-- dg-sensitivity:end -->", 1)[0]
    assert section == check_dg_bound.render_sensitivity(evidence)


def test_sensitivity_table_includes_the_reverse_direction_in_each_bound() -> None:
    import check_dg_bound

    evidence = json.loads((ROOT / sensitivity.parity.DG_SENSITIVITY_EVIDENCE).read_text(encoding="utf-8"))
    reverse = next(row for row in evidence["records"]
                   if row["case"] == "case0" and row["state"] == "sc" and row["kind"] == "cache-path-reverse")
    reverse["max_rel_rms"] = 1.23456
    row = next(line for line in check_dg_bound.render_sensitivity(evidence).splitlines() if line.startswith("| case0 |"))
    assert " / 1.23456 |" in row


def test_documented_layer15_scale_matches_the_recorded_model_mutation() -> None:
    record = json.loads((ROOT / "bench/checks/2026-11-10-dg-mutant.json").read_text(encoding="utf-8"))
    calibration = json.loads((ROOT / sensitivity.parity.DG_SENSITIVITY_EVIDENCE).read_text(encoding="utf-8"))
    docs = (ROOT / "docs/correctness.md").read_text(encoding="utf-8")
    values = re.search(r"layer-15 mutant.*?\bfrom\s+`?([0-9]+(?:\.[0-9]+)?)`?\s+to\s+`?([0-9]+(?:\.[0-9]+)?)`?",
                       docs, flags=re.S)
    assert values is not None, "the mutant's original and replacement scales must be documented"
    assert tuple(map(float, values.groups())) == (record["tensor"]["before"], record["tensor"]["after"])
    assert record["tensor"]["name"] == "blk.15.layer_output_scale.weight"
    assert record["tensor"]["type"] == "F32"
    assert 0 <= record["tensor"]["data_offset"] <= record["source"]["bytes"] - 4
    assert record["source"]["file"] == calibration["model"]
    assert record["source"]["bytes"] == calibration["model_identity"]["bytes"]
    assert record["source"]["sha256"] == calibration["model_provenance"]["declared_hf_lfs_oid"]
    for key in ("file", "bytes", "sha256"):
        assert record["mutant"][key] == record["measured_mutant"][key]
    assert record["mutant"]["bytes"] == record["source"]["bytes"]
    assert re.fullmatch(r"[0-9a-f]{64}", record["mutant"]["sha256"])
    assert record["measured_mutant"]["matches_regenerated"] is True


def test_heldout_verdicts_and_documented_rms_use_the_frozen_calibration() -> None:
    parity = sensitivity.parity
    heldout = json.loads((ROOT / "bench/checks/2026-11-10-dg-heldout.json").read_text(encoding="utf-8"))
    calibration = json.loads((ROOT / parity.DG_SENSITIVITY_EVIDENCE).read_text(encoding="utf-8"))
    assert heldout["calibration"] == parity.DG_SENSITIVITY_EVIDENCE
    assert heldout["calibration_unchanged"] is True
    assert heldout["reference_commit"] == calibration["llama_cpp_commit"]
    for configuration in (heldout["configuration"], calibration["reference_configuration"]):
        assert configuration["cpu_model"] and configuration["isa"]
        assert configuration["cpu_compilation_flags"]
    inputs = calibration["inputs"]["case0"]
    assert all(heldout[key] == inputs[key] for key in ("prompt", "prompt_tokens", "canvas_tokens", "prev_temp_inv"))
    frozen = parity.dg_reference_bound(calibration["records"])[heldout["prompt"]]
    loaded = parity.dg_sensitivity_bound(ROOT)["cases"][heldout["prompt"]]["states"]
    assert loaded == frozen
    threaded, unpacked = heldout["results"], heldout["repacking_off_results"]
    assert {row["variant"] for row in threaded} == set(parity.DG_VARIANTS)
    assert {row["variant"] for row in unpacked} == {"unified", "unified-sc"}
    for row in threaded + unpacked:
        state = "sc" if row["variant"].endswith("-sc") else "no-sc"
        assert row["bound"] == frozen[state]
        violations = parity.dg_violations(row, frozen[state])
        violations += parity.dg_softcap_violations(row["max_abs_logit"], float(inputs["final_logit_softcapping"]))
        assert row["violations"] == violations
        assert row["passed"] == (not violations)
    assert all(row["passed"] and row["max_rel_rms"] == 0 and row["byte_identical"] is True for row in threaded)
    assert all(not row["passed"] and not row["cpu_repacking"] and row["byte_identical"] is False for row in unpacked)
    assert heldout["all_passed"] == all(row["passed"] for row in threaded + unpacked)
    docs = (ROOT / "docs/correctness.md").read_text(encoding="utf-8")
    numbers = re.search(r"maximum relative RMS is ([0-9.]+) without SC and ([0-9.]+) with SC,\s*"
                        r"against bounds of ([0-9.]+) and ([0-9]+\.[0-9]+)", docs)
    assert numbers is not None
    by_variant = {row["variant"]: row for row in unpacked}
    expected = (by_variant["unified"]["max_rel_rms"], by_variant["unified-sc"]["max_rel_rms"],
                frozen["no-sc"]["max_rel_rms"], frozen["sc"]["max_rel_rms"])
    assert numbers.groups() == tuple(f"{value:.5f}" for value in expected)


@pytest.mark.parametrize("recorded", [False, True])
def test_make_mutant_optionally_records_the_actual_patch_and_file_identities(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch,
                                                                          recorded: bool) -> None:
    import check_dg_bound

    source, mutant, record_path = tmp_path / "source.gguf", tmp_path / "mutant.gguf", tmp_path / "mutation.json"
    source.write_bytes(b"GGUF" + struct.pack("<f", 0.5234375))

    def read_fixture(path: str) -> SimpleNamespace:
        value = struct.unpack("<f", Path(path).read_bytes()[4:8])[0]
        tensor = SimpleNamespace(name="blk.15.layer_output_scale.weight", data_offset=4, data=[value],
                                 n_elements=1, tensor_type=SimpleNamespace(name="F32"))
        return SimpleNamespace(tensors=[tensor])

    monkeypatch.setitem(sys.modules, "gguf", SimpleNamespace(GGUFReader=read_fixture))
    original_sha256 = check_dg_bound.parity._sha256
    hashed = []

    def file_identity(path: Path) -> str:
        hashed.append(path)
        return original_sha256(path)

    monkeypatch.setattr(check_dg_bound.parity, "_sha256", file_identity)
    args = SimpleNamespace(model=str(source), layer=15, out=str(mutant), record=str(record_path) if recorded else None)
    assert check_dg_bound.make_mutant(args) == 0
    assert source.read_bytes() == b"GGUF" + struct.pack("<f", 0.5234375)
    assert mutant.read_bytes() == b"GGUF" + struct.pack("<f", 1.0)
    assert hashed == ([source, mutant] if recorded else [])
    assert record_path.exists() is recorded
    if recorded:
        record = json.loads(record_path.read_text())
        assert record["tensor"] == dict(name="blk.15.layer_output_scale.weight", before=0.5234375, after=1.0,
                                        data_offset=4, type="F32")
        assert record["source"] == dict(file=source.name, bytes=8, sha256=original_sha256(source))
        assert record["mutant"] == dict(file=mutant.name, bytes=8, sha256=original_sha256(mutant))


@pytest.mark.parametrize("record_name", ["source.gguf", "mutant.gguf"])
def test_make_mutant_record_cannot_overwrite_either_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       record_name: str) -> None:
    import check_dg_bound

    monkeypatch.setitem(sys.modules, "gguf", SimpleNamespace(GGUFReader=lambda path: None))
    source, mutant = tmp_path / "source.gguf", tmp_path / "mutant.gguf"
    source.write_bytes(b"original model")
    args = SimpleNamespace(model=str(source), layer=15, out=str(mutant), record=str(tmp_path / record_name))
    with pytest.raises(SystemExit, match="record path must differ"):
        check_dg_bound.make_mutant(args)
    assert source.read_bytes() == b"original model"
    assert not mutant.exists()


def sensitivity_case_directories(tmp_path: Path) -> tuple[SimpleNamespace, Path, Path, dict]:
    """Write a tiny complete measurement with a changed-weight output and shared SC input."""
    np = pytest.importorskip("numpy")
    original, dropped = tmp_path / "original/case0", tmp_path / "dropped/case0"
    original.mkdir(parents=True)
    dropped.mkdir(parents=True)
    meta = dict(prompt="tiny", prompt_tokens=[2], canvas_tokens=[3], canvas_seed=66, canvas_length=1,
                n_vocab=2, prev_temp_inv=1.25, final_logit_softcapping="30", threads=1,
                model="tiny.gguf", model_sha256=None, llama_cpp_commit="pin", prev_file=str(original / "unified.ref.f32"))
    for variant in (*sensitivity.parity.DG_VARIANTS, "unified-repeat", "unified-sc-repeat", "unified-sc-noisy"):
        np.array([[1.0, 2.0]], dtype=np.float32).tofile(original / f"{variant}.ref.f32")
    meta["controls"] = [row for row in sensitivity.controls(original, 2) if row["kind"] == "cache-path"]
    (original / "case.json").write_text(json.dumps(meta))
    (dropped / "case.json").write_text(json.dumps(dict(meta, model="mutant.gguf")))
    for variant in sensitivity.parity.DG_VARIANTS:
        np.array([[2.0, 1.0]], dtype=np.float32).tofile(dropped / f"{variant}.ref.f32")
        command = ["tiny-ref", "--prev", meta["prev_file"], "--prev-temp-inv", "1.25"]
        (dropped / f"{variant}.ref.run.json").write_text(json.dumps(dict(command=command)))
    args = SimpleNamespace(cases=str(original.parent), dropped=str(dropped.parent), dropped_layer=15,
                           temperature=0.8, out=str(tmp_path / "check.json"))
    limits = dict(min_cosine=-1.0, argmax_fraction=1.0, mean_top5_overlap=2.0, max_rel_rms=0.0, max_top1_rel_err=0.0)
    bound = dict(protocol="dg-fixed-input-v1", model=meta["model"], llama_cpp_commit="pin",
                 cases={"tiny": dict(inputs=meta, states={"no-sc": limits, "sc": limits})})
    return args, original, dropped, bound


@pytest.mark.parametrize("operation", ["check", "sensitivity_report"])
def test_bound_entrypoints_accept_original_previous_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    import check_dg_bound

    args, _, _, bound = sensitivity_case_directories(tmp_path)
    monkeypatch.setattr(check_dg_bound.parity, "dg_sensitivity_bound", lambda root: bound)
    assert getattr(check_dg_bound, operation)(args) == 0


@pytest.mark.parametrize("operation", ["check", "sensitivity_report"])
@pytest.mark.parametrize("changed", ["metadata", "executed-previous", "executed-temperature", "missing-receipt"])
def test_bound_entrypoints_refuse_mutant_previous_or_temperature(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                               operation: str, changed: str) -> None:
    import check_dg_bound

    args, _, dropped, bound = sensitivity_case_directories(tmp_path)
    monkeypatch.setattr(check_dg_bound.parity, "dg_sensitivity_bound", lambda root: bound)
    if changed == "metadata":
        path = dropped / "case.json"
        meta = json.loads(path.read_text())
        meta["prev_file"] = str(dropped / "unified.ref.f32")
        path.write_text(json.dumps(meta))
    else:
        path = dropped / "unified-sc.ref.run.json"
        if changed == "missing-receipt":
            path.unlink()
        else:
            receipt = json.loads(path.read_text())
            if changed == "executed-previous":
                receipt["command"][2] = str(dropped / "unified.ref.f32")
            else:
                receipt["command"][4] = "1.0"
            path.write_text(json.dumps(receipt))
    with pytest.raises(SystemExit, match="previous logits|original reference's previous|receipt.*missing"):
        getattr(check_dg_bound, operation)(args)


def test_bound_check_refuses_a_mutant_with_extra_canvas_rows(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import check_dg_bound

    np = pytest.importorskip("numpy")
    args, _, dropped, bound = sensitivity_case_directories(tmp_path)
    monkeypatch.setattr(check_dg_bound.parity, "dg_sensitivity_bound", lambda root: bound)
    np.array([[2.0, 1.0], [2.0, 1.0]], dtype=np.float32).tofile(dropped / "unified.ref.f32")
    with pytest.raises(SystemExit, match="differs in shape"):
        check_dg_bound.check(args)


@pytest.mark.parametrize("fault", ["rejected-control", "accepted-mutant"])
def test_bound_check_reports_failures_and_cli_exits_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     fault: str) -> None:
    import check_dg_bound

    np = pytest.importorskip("numpy")
    args, original, dropped, bound = sensitivity_case_directories(tmp_path)
    if fault == "rejected-control":
        np.array([[2.0, 1.0]], dtype=np.float32).tofile(original / "cached.ref.f32")
        message = "unmodified reference was rejected"
    else:
        for variant in sensitivity.parity.DG_VARIANTS:
            np.array([[1.0, 2.0]], dtype=np.float32).tofile(dropped / f"{variant}.ref.f32")
        message = "layer 15 output scale dropped: accepted on every case and variant"
    monkeypatch.setattr(check_dg_bound.parity, "dg_sensitivity_bound", lambda root: bound)
    assert check_dg_bound.check(args) == 1
    report = json.loads(Path(args.out).read_text())
    assert report["all_as_expected"] is False
    if fault == "rejected-control":
        assert any(not row["is_mutant"] and row["rejected"] for row in report["results"])
    else:
        assert report["mutants_rejected"]["layer 15 output scale dropped"] is False
    bound_path = tmp_path / "fixture-bound.json"
    bound_path.write_text(json.dumps(bound))
    # Exercise the actual argument parser and process exit with the same tiny bound.
    program = (
        "import json,pathlib,sys; "
        f"sys.path.insert(0,{str(ROOT / 'bench')!r}); "
        "import check_dg_bound as check; "
        f"bound=json.loads(pathlib.Path({str(bound_path)!r}).read_text()); "
        "check.parity.dg_sensitivity_bound=lambda root: bound; "
        "sys.argv=['check_dg_bound.py',*sys.argv[1:]]; "
        "raise SystemExit(check.main())"
    )
    result = subprocess.run([sys.executable, "-c", program, "check", args.cases, "--dropped", args.dropped,
                             "--dropped-layer", "15", "--out", args.out], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    assert message in result.stderr
    assert json.loads(Path(args.out).read_text())["all_as_expected"] is False
