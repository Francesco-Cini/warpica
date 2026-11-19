import importlib.util

import numpy as np
import pytest

from warpica.metrics import metric_get_alcubierre
from warpica.solver import get_energy_tensor
from warpica.ui.core import solving
from warpica.ui.core.solving import (
    DIFF_ORDERS,
    available_backends,
    solve_energy_tensor,
)


def test_fourth_order_is_the_default():
    assert next(iter(DIFF_ORDERS.values())) == "fourth"
    assert set(DIFF_ORDERS.values()) == {"second", "fourth"}


def test_the_cpu_is_always_offered_first():
    backends = available_backends()

    assert next(iter(backends.values())) is None
    assert list(backends)[0] == "CPU (NumPy)"


def test_a_gpu_backend_is_offered_only_when_its_library_is_installed(monkeypatch):
    installed = {"cupy"}
    monkeypatch.setattr(
        solving.importlib.util,
        "find_spec",
        lambda name: object() if name in installed else None,
    )

    assert set(available_backends().values()) == {None, "cupy"}

    installed.add("torch_directml")
    assert set(available_backends().values()) == {None, "cupy", "torch-directml"}


def test_without_gpu_libraries_only_the_cpu_is_offered(monkeypatch):
    monkeypatch.setattr(solving.importlib.util, "find_spec", lambda name: None)

    assert available_backends() == {"CPU (NumPy)": None}


@pytest.mark.parametrize("order", ["second", "fourth"])
def test_solving_matches_calling_the_solver_directly(order):
    grid = np.array([1, 7, 7, 7])
    scale = np.array([1.0, 0.5, 0.5, 0.5])
    metric = metric_get_alcubierre(grid, (grid + 1) / 2 * scale, 0.4, 1.0, 1.0, scale)

    result = solve_energy_tensor(metric, order, None)
    expected = get_energy_tensor(metric, order)

    assert result["order"] == order
    np.testing.assert_allclose(result["tensor"][0][0], expected["tensor"][0][0])


def test_find_spec_really_reports_missing_gpu_libraries_here():
    # Sanity check of the real lookup, not a patched one.
    assert (importlib.util.find_spec("cupy") is None) == (
        "cupy" not in available_backends().values()
    )
