"""Run with the fixtures and test groups: numerical edge cases in DG diagnostics."""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))
check = importlib.import_module("check_dg_bound")
sensitivity = importlib.import_module("dg_sensitivity")


def test_uncapping_saturated_float32_logits_is_finite_and_detectable() -> None:
    logits = np.array([[-30.0, 0.0, 30.0]], dtype=np.float32)
    with np.errstate(all="raise"):
        mutant = check.remove_softcap(logits, 30.0)
    assert mutant.dtype == np.float64
    assert np.isfinite(mutant).all()
    assert mutant[0, 0] < -30.0 and mutant[0, 2] > 30.0
    assert np.argmax(mutant) == np.argmax(logits)


def test_distribution_distance_ignores_constant_logit_offset() -> None:
    a = np.array([[3.0, 1.0, -2.0], [-1.0, 4.0, 2.0]])
    metrics = sensitivity.sampling_metrics(a + 123.0, a, temperature=0.8)
    assert metrics["max_total_variation"] == pytest.approx(0.0, abs=1e-14)
    assert metrics["max_js_divergence"] == pytest.approx(0.0, abs=1e-14)


def test_distribution_distance_detects_changed_winner_and_stays_finite() -> None:
    a, b = np.array([[1000.0, -1000.0]]), np.array([[-1000.0, 1000.0]])
    metrics = sensitivity.sampling_metrics(a, b, temperature=0.8)
    assert metrics["max_total_variation"] == pytest.approx(1.0)
    assert metrics["max_js_divergence"] == pytest.approx(math.log(2.0))


@pytest.mark.parametrize("temperature", [0.0, -0.8, float("nan"), float("inf")])
def test_distribution_distance_refuses_invalid_temperature(temperature: float) -> None:
    logits = np.array([[1.0, 2.0]])
    with pytest.raises(SystemExit, match="temperature.*positive and finite"):
        sensitivity.sampling_metrics(logits, logits, temperature=temperature)


@pytest.mark.parametrize("candidate", [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 2.0]), np.empty((0, 2))])
def test_distribution_distance_refuses_incompatible_or_empty_matrices(candidate) -> None:
    with pytest.raises(SystemExit, match="non-empty.*same shape"):
        sensitivity.sampling_metrics(candidate, np.array([[1.0, 2.0]]), temperature=0.8)


@pytest.mark.parametrize("side", ["candidate", "reference"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_distribution_distance_refuses_nonfinite_logits(side: str, value: float) -> None:
    a, b = np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]])
    (a if side == "candidate" else b)[0, 0] = value
    with pytest.raises(SystemExit, match="non-finite logits"):
        sensitivity.sampling_metrics(a, b, temperature=0.8)


@pytest.mark.parametrize("cap", [0.0, -30.0, float("nan"), float("inf")])
def test_softcap_inversion_refuses_invalid_cap(cap: float) -> None:
    with pytest.raises(SystemExit, match="positive finite cap"):
        check.remove_softcap(np.array([[1.0, 2.0]]), cap)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_softcap_inversion_refuses_nonfinite_logits(value: float) -> None:
    with pytest.raises(SystemExit, match="finite logits"):
        check.remove_softcap(np.array([[value, 2.0]]), 30.0)
