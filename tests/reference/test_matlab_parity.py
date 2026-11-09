"""Frozen outputs were computed in MATLAB, not by the Python implementation."""

import json
from pathlib import Path

import numpy as np
import pytest

from warpica.analyser import eval_metric
from warpica.metrics import (
    metric_get_alcubierre,
    metric_get_alcubierre_comoving,
    metric_get_lentz,
    metric_get_lentz_comoving,
    metric_get_minkowski,
    metric_get_schwarzschild,
    metric_get_van_den_broeck,
)
from warpica.solver import get_energy_tensor
from warpica.solver.utils.c4_inv import c4_inv
from warpica.solver.utils.ricci_t import ricci_t
from warpica.solver.utils.second_order.ricci_t_2 import ricci_t_2
from warpica.units.universal_constants.c import c
from warpica.units.universal_constants.G import G

pytestmark = pytest.mark.parity
DIRECTORY = Path(__file__).parent / "matlab"

# Each case rebuilds the metric that tools/export_matlab_reference.m exported,
# with the same parameters. Grid, spacing and centre come from the metadata.
BUILDERS = {
    "minkowski": lambda grid, centre, spacing: metric_get_minkowski(grid, spacing),
    "alcubierre_static": lambda g, c, s: metric_get_alcubierre(g, c, 0.2, 0.8, 2, s),
    "alcubierre_dynamic": lambda g, c, s: metric_get_alcubierre(g, c, 0.2, 0.8, 2, s),
    "alcubierre_comoving": lambda g, c, s: metric_get_alcubierre_comoving(
        g, c, 0.2, 0.8, 2, s
    ),
    "lentz_dynamic": lambda g, c, s: metric_get_lentz(g, c, 0.2, 1, s),
    "lentz_comoving": lambda g, c, s: metric_get_lentz_comoving(g, c, 0.2, 1, s),
    "van_den_broeck_dynamic": lambda g, c, s: metric_get_van_den_broeck(
        g, c, 0.2, 1, 1, 2, 1, 0.1, s
    ),
    "schwarzschild": lambda g, c, s: metric_get_schwarzschild(g, c, 0.05, s),
}

# Relative tolerance against the largest reference value of each array.
# Upstream Schwarzschild adds 1e-10 to r, which the exact Python formula
# does not (see test_schwarzschild_difference_is_the_upstream_epsilon).
TOLERANCE = {"schwarzschild": 1e-7}
DEFAULT_TOLERANCE = 1e-10

FIELDS = [
    "metric",
    "ricci_second",
    "ricci_fourth",
    "energy_second",
    "energy_fourth",
    "eulerian",
    "dominant",
    "vorticity",
]


def case_item(case):
    metadata = json.loads((DIRECTORY / "metadata.json").read_text())
    item = metadata["cases"][case]
    return tuple(np.asarray(item[key]) for key in ("grid", "spacing", "center"))


@pytest.fixture(scope="module", params=list(BUILDERS))
def comparison(request):
    case = request.param
    grid, spacing, center = case_item(case)
    metric = BUILDERS[case](grid, center, spacing)
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
    # MATLAB's second-order path uses the rounded constants G = 6.6740e-11 and
    # c = 2.99792e8; this port uses warpica.units. Rescale the fixture by the exact
    # ratio so the comparison stays tight (the ratio is about 1 - 3.9e-5).
    expected["energy_second"] = expected["energy_second"] * (
        (c() ** 4 / G()) / (2.99792e8**4 / 6.6740e-11)
    )
    return case, actual, expected


@pytest.mark.parametrize("field", FIELDS)
def test_independent_matlab_arrays(comparison, field):
    case, actual, expected = comparison
    scale = float(np.max(np.abs(expected[field])))
    tolerance = TOLERANCE.get(case, DEFAULT_TOLERANCE)
    np.testing.assert_allclose(
        actual[field],
        expected[field],
        rtol=tolerance,
        atol=max(1e-30, scale * max(1e-12, tolerance)),
    )


def test_fixtures_are_finite():
    """Non-finite reference values would make comparisons meaningless."""
    with np.load(DIRECTORY / "warp_factory.npz", allow_pickle=False) as fixture:
        for name in fixture.files:
            assert np.isfinite(fixture[name]).all(), name


def test_schwarzschild_difference_is_the_upstream_epsilon():
    """Upstream adds 1e-10 to r; with it, its formula reproduces MATLAB exactly.

    This pins down why Python differs from the fixture at about 1e-9: the
    reference metric equals upstream's formula evaluated with r + 1e-10, to
    roundoff, while Python evaluates the exact formula.
    """
    grid, spacing, center = case_item("schwarzschild")
    r_s = 0.05
    x, y, z = np.meshgrid(
        *[(np.arange(grid[i]) + 1) * spacing[i] - center[i] for i in (1, 2, 3)],
        indexing="ij",
    )
    r = np.sqrt(x * x + y * y + z * z) + 1e-10
    f = 1 - r_s / r
    metric = np.zeros((4, 4, *x.shape))
    metric[0, 0] = -f
    metric[1, 1] = (x * x / f + y * y + z * z) / r**2
    metric[2, 2] = (x * x + y * y / f + z * z) / r**2
    metric[3, 3] = (x * x + y * y + z * z / f) / r**2
    for i, j, a, b in ((1, 2, x, y), (1, 3, x, z), (2, 3, y, z)):
        metric[i, j] = metric[j, i] = r_s / (r**3 - r**2 * r_s) * a * b
    with np.load(DIRECTORY / "warp_factory.npz", allow_pickle=False) as fixture:
        reference = fixture["schwarzschild__metric"][:, :, 0]
    np.testing.assert_allclose(
        metric, reference, rtol=0, atol=1e-14 * np.abs(reference).max()
    )
