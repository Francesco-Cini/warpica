"""Ensure the public teaching examples stay executable as the API evolves."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.parametrize(
    "module",
    [
        "examples.basic.minkowski",
        "examples.basic.custom_metric",
        "examples.analysis.alcubierre",
        "examples.analysis.compare_orders",
    ],
)
def test_example_runs(module):
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()


def test_surface_plot_example_runs():
    """Exercises the themed 3D surface helper through its only real caller."""
    pytest.importorskip("matplotlib")
    result = subprocess.run(
        [sys.executable, "-m", "examples.visualisation.alcubierre_plots"],
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip()
    assert "Traceback" not in result.stderr
