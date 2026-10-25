"""Tests for the row comparison used by scripts/legacy_parity.py."""

from pathlib import Path
import importlib.util
import math

import pytest

pd = pytest.importorskip("pandas")

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "legacy_parity.py"
spec = importlib.util.spec_from_file_location("legacy_parity", SCRIPT)
legacy_parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy_parity)
compare = legacy_parity.compare_episodes


def frame(**columns):
    return pd.DataFrame({"seed": [1, 2], **columns})


def test_identical_frames_match_exactly():
    a = frame(steps=[10, 20], risk=[0.5, 1.5], termination=["success", "timeout"])
    r = compare(a, a.copy(), atol=1e-9, rtol=1e-5)
    assert r["mismatches"] == [] and r["episodes_exact_within_atol"] == 2


def test_nan_on_one_side_is_a_mismatch_and_not_exact():
    a = frame(risk=[0.5, 1.5])
    b = frame(risk=[0.5, math.nan])
    r = compare(a, b, atol=1e-9, rtol=1e-5)
    assert [m["seed"] for m in r["mismatches"]] == [2]
    assert r["episodes_exact_within_atol"] == 1 and math.isinf(r["max_abs_diff"]["risk"])


def test_equal_infinities_match():
    a = frame(risk=[math.inf, 1.0])
    assert compare(a, a.copy(), atol=1e-9)["mismatches"] == []


def test_seed_missing_from_archive():
    a = frame(risk=[0.5, 1.5])
    b = pd.DataFrame({"seed": [1, 3], "risk": [0.5, 9.0]})
    r = compare(a, b, atol=1e-9)
    assert r["episodes_fully_matching"] == 1 and r["episodes_exact_within_atol"] == 1


def test_column_on_one_side_fails():
    a = frame(risk=[0.5, 1.5], steps=[1, 2])
    b = frame(risk=[0.5, 1.5])
    r = compare(a, b, atol=1e-9)
    assert r["columns_only_in_archive"] == ["steps"] and r["episodes_fully_matching"] == 0


def test_rtol_never_applies_to_integer_columns():
    a = frame(steps=[1000, 20], risk=[1000.0, 2.0])
    b = frame(steps=[1001, 20], risk=[1000.5, 2.0])
    r = compare(a, b, atol=1e-9, rtol=1e-3)
    assert [m["column"] for m in r["mismatches"]] == ["steps"]


def test_episode_counter_is_ignored():
    a = frame(episode=[100, 101], risk=[0.5, 1.5])
    b = frame(episode=[0, 1], risk=[0.5, 1.5])
    assert compare(a, b, atol=1e-9)["mismatches"] == []
