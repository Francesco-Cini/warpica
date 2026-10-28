"""Frozen outputs were computed in MATLAB, not by the Python implementation."""

import json
from pathlib import Path

import numpy as np
import pytest

from warplab.analyser.eval_metric import eval_metric
from warplab.metrics.alcubierre.metric_get_alcubierre import metric_get_alcubierre
from warplab.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warplab.solver.get_energy_tensor import get_energy_tensor
from warplab.solver.utils.c4_inv import c4_inv
from warplab.solver.utils.ricci_t import ricci_t
from warplab.solver.utils.second_order.ricci_t_2 import ricci_t_2

pytestmark = pytest.mark.parity
DIRECTORY = Path(__file__).parent / "matlab"


@pytest.fixture(
    scope="module", params=["minkowski", "alcubierre_static", "alcubierre_dynamic"]
)
def comparison(request):
    case = request.param
    metadata = json.loads((DIRECTORY / "metadata.json").read_text())
    item = metadata["cases"][case]
    grid, spacing, center = map(
        np.asarray, (item["grid"], item["spacing"], item["center"])
    )
    metric = (
        metric_get_minkowski(grid, spacing)
        if case == "minkowski"
        else metric_get_alcubierre(grid, center, 0.2, 0.8, 2, spacing)
    )
    result = eval_metric(metric, num_angular_vec=12, num_time_vec=3)
    actual = {
        "metric": metric["tensor"],
        "ricci_second": ricci_t_2(c4_inv(metric["tensor"]), metric["tensor"], spacing),
        "ricci_fourth": ricci_t(c4_inv(metric["tensor"]), metric["tensor"], spacing),
        "energy_second": get_energy_tensor(metric, "second")["tensor"],
        "energy_fourth": result["energy_tensor"]["tensor"],
        "eulerian": result["energy_tensor_eulerian"]["tensor"],
    }
    actual.update({name: result[name] for name in ("dominant", "vorticity")})
    with np.load(DIRECTORY / "warp_factory.npz", allow_pickle=False) as fixture:
        expected = {name: fixture[case + "__" + name] for name in actual}
    return actual, expected


@pytest.mark.parametrize(
    "field",
    [
        "metric",
        "ricci_second",
        "ricci_fourth",
        "energy_second",
        "energy_fourth",
        "eulerian",
        "dominant",
        "vorticity",
    ],
)
def test_independent_matlab_arrays(comparison, field):
    actual, expected = comparison
    scale = float(np.max(np.abs(expected[field])))
    np.testing.assert_allclose(
        actual[field], expected[field], rtol=1e-10, atol=max(1e-30, scale * 1e-12)
    )
