import numpy as np
import pytest

from warplab.analyser.eval_metric import eval_metric
from warplab.gpu import asarray, asnumpy
from warplab.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre
from warplab.solver.utils.second_order.take_finite_difference_1_2 import (
    take_finite_difference_1_2 as d12,
)
from warplab.solver.utils.second_order.take_finite_difference_2_2 import (
    take_finite_difference_2_2 as d22,
)
from warplab.solver.utils.take_finite_difference_1 import (
    take_finite_difference_1 as d14,
)
from warplab.solver.utils.take_finite_difference_2 import (
    take_finite_difference_2 as d24,
)

pytestmark = pytest.mark.gpu


def test_device_round_trip(gpu_backend):
    values = np.linspace(-1, 1, 32).reshape(4, 8)
    device = asarray(values, library=gpu_backend)
    if gpu_backend == "torch-directml":
        assert device.device.type == "privateuseone"
    else:
        assert device.device.id >= 0
    np.testing.assert_allclose(asnumpy(device), values, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("first,second", [(d12, d22), (d14, d24)])
def test_all_derivative_axes(gpu_backend, first, second):
    grid = np.indices((5, 6, 7, 8)).astype(float)
    field = np.sin(0.1 * grid[0] + 0.2 * grid[1] - 0.3 * grid[2] + 0.15 * grid[3])
    device = asarray(field, library=gpu_backend)
    delta = np.array([0.3, 0.4, 0.5, 0.6])
    for i in range(4):
        np.testing.assert_allclose(
            asnumpy(first(device, i, delta)),
            first(field, i, delta),
            rtol=1e-4,
            atol=3e-6,
        )
        for j in range(4):
            np.testing.assert_allclose(
                asnumpy(second(device, i, j, delta)),
                second(field, i, j, delta),
                rtol=1e-4,
                atol=1e-5,
            )


@pytest.mark.parametrize("order", ["second", "fourth"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_full_analysis_matches_cpu(gpu_backend, order, dynamic):
    from warplab.units.universal_constants.c import c

    grid = np.array([5 if dynamic else 1, 5, 5, 5])
    spacing = np.array([0.25 / c(), 0.5, 0.5, 0.5])
    metric = metric_get_alcubierre(grid, (grid + 1) / 2 * spacing, 0.2, 0.8, 2, spacing)
    before = np.asarray(metric["tensor"]).copy()
    expected = eval_metric(metric, num_angular_vec=8, num_time_vec=3, diff_order=order)
    actual = eval_metric(
        metric, num_angular_vec=8, num_time_vec=3, diff_order=order, gpu=gpu_backend
    )
    tolerance = 2e-5 if gpu_backend == "torch-directml" else 1e-10
    for key in (
        "energy_tensor",
        "energy_tensor_eulerian",
        "null",
        "weak",
        "strong",
        "dominant",
        "expansion",
        "shear",
        "vorticity",
    ):
        ref = np.asarray(
            expected[key]["tensor"]
            if isinstance(expected[key], dict)
            else expected[key]
        )
        result = np.asarray(
            actual[key]["tensor"] if isinstance(actual[key], dict) else actual[key]
        )
        assert np.isfinite(result).all()
        np.testing.assert_allclose(
            result, ref, rtol=tolerance, atol=max(np.max(abs(ref)) * tolerance, 1e-30)
        )
    np.testing.assert_array_equal(metric["tensor"], before)
