"""The guard's own logic, tested without timing anything (runs in every build)."""

import json
import math

import pytest

from . import timing
from .timing import (
    MAX_EXPONENT,
    OPERATIONS,
    TOLERANCE,
    best_time,
    exponent_of,
    load_baseline,
    problems,
    summary,
)


def result(**overrides):
    values = {
        "operation": "solver",
        "small_grid": 10,
        "large_grid": 20,
        "small_seconds": 1.0,
        "large_seconds": 4.0,
        "reference_seconds": 1.0,
        "exponent": 0.67,
        "ratio": 10.0,
    }
    values.update(overrides)
    return values


def test_exponent_of_linear_and_quadratic_growth():
    assert exponent_of(1.0, 8.0, 10, 20) == pytest.approx(1.0)
    assert exponent_of(1.0, 64.0, 10, 20) == pytest.approx(2.0)
    assert exponent_of(1.0, 1.0, 10, 20) == pytest.approx(0.0)


def test_a_healthy_measurement_has_no_problems():
    assert problems(result(), baseline_ratio=10.0) == []


@pytest.mark.parametrize("exponent", [MAX_EXPONENT + 0.01, 1.8, 2.0])
def test_super_linear_growth_is_reported(exponent):
    found = problems(result(exponent=exponent), baseline_ratio=10.0)
    assert len(found) == 1
    assert "grows as cells" in found[0]
    assert "solver" in found[0]


def test_growth_up_to_the_limit_is_accepted():
    assert problems(result(exponent=MAX_EXPONENT), baseline_ratio=10.0) == []


def test_a_slowdown_beyond_the_tolerance_is_reported():
    found = problems(result(ratio=10.0 * TOLERANCE + 0.1), baseline_ratio=10.0)
    assert len(found) == 1
    assert "reference kernel" in found[0]


def test_a_slowdown_within_the_tolerance_is_accepted():
    assert problems(result(ratio=10.0 * TOLERANCE), baseline_ratio=10.0) == []


def test_both_problems_can_be_reported_together():
    assert len(problems(result(exponent=1.9, ratio=500.0), baseline_ratio=10.0)) == 2


def test_best_time_warms_up_and_keeps_the_fastest_run(monkeypatch):
    clock = iter([0.0, 5.0, 10.0, 11.0, 20.0, 22.0, 30.0, 31.0])
    monkeypatch.setattr(timing.time, "perf_counter", lambda: next(clock))
    calls = []
    assert best_time(lambda: calls.append(1), repeats=4) == 1.0
    assert len(calls) == 5  # one warm-up plus four timed runs


def test_best_time_needs_a_callable_that_runs_cleanly():
    with pytest.raises(ZeroDivisionError):
        best_time(lambda: 1 / 0)


def test_the_baseline_covers_every_guarded_operation():
    baseline = load_baseline()
    assert set(baseline) == set(OPERATIONS)
    for entry in baseline.values():
        assert math.isfinite(entry["ratio"]) and entry["ratio"] > 0


def test_the_baseline_file_records_where_it_was_measured():
    machine = json.loads(timing.BASELINE_FILE.read_text())["machine"]
    assert {"python", "numpy", "platform"} <= set(machine)


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_every_guarded_operation_still_runs(name):
    """A small grid keeps this fast; it fails if an API change breaks the guard."""
    prepare, _, _ = OPERATIONS[name]
    prepare(7)()


def test_the_reference_kernel_is_deterministic():
    first = timing.reference_workload()
    assert first.shape == (19, 19, 19)
    assert (timing.reference_workload() == first).all()


def test_the_summary_lists_every_operation_with_its_limits(tmp_path):
    report = {
        name: result(operation=name, exponent=0.5, ratio=3.0) for name in OPERATIONS
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    table = summary(path)
    assert table.startswith("### Performance guard (advisory)")
    for name in OPERATIONS:
        assert f"| {name} |" in table
    assert f"limit {MAX_EXPONENT}" in table


def test_the_summary_copes_with_a_missing_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(timing, "BASELINE_FILE", tmp_path / "absent.json")
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"solver": result()}))
    assert "| solver |" in summary(path)
