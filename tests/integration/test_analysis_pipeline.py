import numpy as np
import pytest

from warpica.analyser.eval_metric import eval_metric
from warpica.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre
from warpica.metrics.minkowski.metric_get_minkowski import metric_get_minkowski

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("order", ["second", "fourth"])
@pytest.mark.parametrize("curved", [False, True])
def test_full_cpu_analysis(order, curved):
    grid = np.array([1, 7, 7, 7])
    metric = (
        metric_get_alcubierre(grid, (grid + 1) / 2, 0.2, 1.5, 1)
        if curved
        else metric_get_minkowski(grid)
    )
    original = np.asarray(metric["tensor"]).copy()
    result = eval_metric(metric, num_angular_vec=8, num_time_vec=3, diff_order=order)
    for name in (
        "null",
        "weak",
        "strong",
        "dominant",
        "expansion",
        "shear",
        "vorticity",
    ):
        assert result[name].shape == tuple(grid)
        assert np.isfinite(result[name]).all()
        if not curved:
            np.testing.assert_allclose(result[name], 0, atol=1e-12)
    assert result["energy_tensor"]["index"] == "contravariant"
    assert result["energy_tensor"]["order"] == order
    assert result["energy_tensor_eulerian"]["frame"] == "Eulerian"
    np.testing.assert_array_equal(metric["tensor"], original)
    if curved:
        assert np.any(result["null"] < 0)
        clipped = eval_metric(metric, False, 8, 3, diff_order=order)
        np.testing.assert_allclose(clipped["null"], np.minimum(result["null"], 0))
