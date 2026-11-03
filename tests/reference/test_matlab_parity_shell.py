"""Warp-shell parity with MATLAB, which needs per-case care.

Python and upstream sample the shell's radial profiles over different ranges
and Python adds a future-direction rule to the dominant condition, so these
comparisons use their own tolerances. See docs/reference-validation.md.
"""

import importlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from warplab.analyser import change_tensor_index, eval_metric
from warplab.analyser.utils.generate_uniform_field import generate_uniform_field
from warplab.analyser.utils.get_inner_product import get_inner_product
from warplab.metrics import metric_get_minkowski, metric_get_warp_shell_comoving
from warplab.solver import get_energy_tensor
from warplab.solver.utils.c4_inv import c4_inv
from warplab.solver.utils.ricci_t import ricci_t
from warplab.solver.utils.second_order.ricci_t_2 import ricci_t_2

pytestmark = pytest.mark.parity
DIRECTORY = Path(__file__).parent / "matlab"
MODULE = "warplab.metrics.warp_shell.metric_get_warp_shell_comoving"

# smoothFactor 1 makes upstream's smooth() an identity; 502 does not.
SMOOTH_FACTORS = {"warp_shell_comoving": 1, "warp_shell_smoothed": 502}
FIELDS = [
    "metric",
    "ricci_second",
    "ricci_fourth",
    "energy_second",
    "energy_fourth",
    "eulerian",
    "vorticity",
]

# With Python's own sampling range the 100,000 radial samples fall differently
# from upstream's, because the two ranges differ on these deliberately
# off-centre grids (Python: 1.2 x the largest grid radius; upstream: 1.2 x the
# far-corner distance). That moves the samples relative to the density edges
# (an O(h) quadrature effect, about 1e-4 relative). smooth() windows are also
# counted in samples, so for the smoothed case the physical window differs by
# the ratio of the ranges (6% here), which raises the difference to about 1e-3.
# Rescaling Python's window to the same physical width removes that extra
# error. The tight comparison below uses upstream's range instead.
OWN_RANGE_TOLERANCE = {"warp_shell_comoving": 5e-4, "warp_shell_smoothed": 3e-3}
# With upstream's range substituted, only linear versus cubic interpolation
# differs; measured agreement is about 1e-9.
UPSTREAM_RANGE_TOLERANCE = 1e-7


def case_geometry(case):
    metadata = json.loads((DIRECTORY / "metadata.json").read_text())
    item = metadata["cases"][case]
    return tuple(np.asarray(item[key]) for key in ("grid", "spacing", "center"))


def build(case):
    grid, spacing, center = case_geometry(case)
    return metric_get_warp_shell_comoving(
        grid, center, 2e26, 0.5, 1.5, 0.1, 1, SMOOTH_FACTORS[case], 0.2, True, spacing
    )


def compute(case):
    _, spacing, _ = case_geometry(case)
    metric = build(case)
    result = eval_metric(metric, num_angular_vec=12, num_time_vec=3)
    actual = {
        "metric": metric["tensor"],
        "ricci_second": ricci_t_2(c4_inv(metric["tensor"]), metric["tensor"], spacing),
        "ricci_fourth": ricci_t(c4_inv(metric["tensor"]), metric["tensor"], spacing),
        "energy_second": get_energy_tensor(metric, "second")["tensor"],
        "energy_fourth": result["energy_tensor"]["tensor"],
        "eulerian": result["energy_tensor_eulerian"]["tensor"],
        "vorticity": result["vorticity"],
        "dominant": result["dominant"],
    }
    with np.load(DIRECTORY / "warp_factory.npz", allow_pickle=False) as fixture:
        expected = {name: fixture[case + "__" + name] for name in actual}
    return actual, expected, result


cached_compute = lru_cache(maxsize=None)(compute)


def assert_matches(actual, expected, tolerance):
    scale = float(np.max(np.abs(expected)))
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=tolerance,
        atol=max(1e-30, scale * max(1e-12, tolerance)),
    )


@pytest.mark.parametrize("case", SMOOTH_FACTORS)
@pytest.mark.parametrize("field", FIELDS)
def test_shell_matches_matlab_with_its_own_sampling_range(case, field):
    actual, expected, _ = cached_compute(case)
    assert_matches(actual[field], expected[field], OWN_RANGE_TOLERANCE[case])


def upstream_radial_limit(case):
    """Upstream samples radii up to 1.2 x the distance to the far grid corner."""
    grid, spacing, center = case_geometry(case)
    far = np.sqrt(sum((grid[i] * spacing[i] - center[i]) ** 2 for i in (1, 2, 3)))
    return 1.2 * far


@pytest.fixture(scope="module", params=list(SMOOTH_FACTORS))
def upstream_sampled(request):
    module = importlib.import_module(MODULE)
    limit = upstream_radial_limit(request.param)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, "_radial_limit", lambda radius, R_2: limit)
        actual, expected, _ = compute(request.param)
    return actual, expected


@pytest.mark.parametrize("field", FIELDS)
def test_shell_agrees_closely_when_sampling_like_upstream(upstream_sampled, field):
    """With upstream's radial range only the interpolation method differs.

    Python interpolates linearly where upstream uses a cubic Lagrange formula.
    For the smoothed case this also checks that Python's reimplementation of
    MATLAB's smooth() applies the same window rules as the real function.
    """
    actual, expected = upstream_sampled
    assert_matches(actual[field], expected[field], UPSTREAM_RANGE_TOLERANCE)


def upstream_dominant(eulerian, num_angular_vec=12):
    """Upstream's dominant condition, which has no future-direction rule."""
    shape = eulerian["tensor"][0][0].shape
    minkowski = change_tensor_index(metric_get_minkowski(np.array(shape)), "covariant")
    mixed = change_tensor_index(eulerian, "mixedupdown", minkowski)
    field = generate_uniform_field("nulllike", num_angular_vec, 1)
    best = np.full(shape, -np.inf)
    for ii in range(num_angular_vec):
        flux = np.zeros((*shape, 4))
        for mu in range(4):
            for nu in range(4):
                flux[..., mu] -= mixed["tensor"][mu][nu] * field[nu, ii]
        vector = {
            "field": np.array([flux[..., q] for q in range(4)]),
            "index": "contravariant",
            "type": "4-vector",
        }
        norm = get_inner_product(vector, vector, minkowski)
        best = np.maximum(best, np.sign(norm) * np.sqrt(np.abs(norm)))
    return -best


@pytest.mark.parametrize("case", SMOOTH_FACTORS)
def test_shell_dominant_differs_only_by_the_future_direction_rule(case):
    """Python's dominant map is upstream's formula plus one extra rule.

    The rule can only enlarge a violation, so Python is never less severe than
    upstream and the two always agree on the sign. On the shell the extra rule
    changes the magnitude by up to about 13%.
    """
    actual, expected, result = cached_compute(case)
    upstream_style = upstream_dominant(result["energy_tensor_eulerian"])
    assert_matches(upstream_style, expected["dominant"], OWN_RANGE_TOLERANCE[case])
    scale = float(np.max(np.abs(expected["dominant"])))
    assert np.all(actual["dominant"] <= upstream_style + 1e-9 * scale)
    np.testing.assert_array_equal(np.sign(actual["dominant"]), np.sign(upstream_style))
