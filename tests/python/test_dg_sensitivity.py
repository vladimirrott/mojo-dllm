"""The sensitivity experiment must compare like inputs and retain its controls."""

from __future__ import annotations

import importlib.util
import json
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
