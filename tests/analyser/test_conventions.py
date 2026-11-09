"""Tests for the documented differences from upstream WarpFactory.

See docs/reference-validation.md. Each test checks the analytically correct
behaviour that Python promises, independently of the MATLAB fixtures.
"""

import copy

import numpy as np
import pytest

from warpica.analyser import get_energy_conditions, get_scalars
from warpica.analyser.utils.generate_uniform_field import generate_uniform_field
from warpica.metrics import metric_get_minkowski


def adm_constant_metric(alpha, gamma, beta):
    """Return flat spacetime in coordinates with constant lapse, scale and shift.

    The metric is covariant with g_ij = gamma * delta_ij, shift beta^i, and
    g_00 = -alpha**2 + beta_i beta^i. The Eulerian observers have four-velocity
    u^mu = (1, -beta^i) / alpha.
    """
    metric = metric_get_minkowski([1, 2, 2, 2])
    beta = np.asarray(beta, dtype=float)
    beta_down = gamma * beta
    tensor = np.zeros((4, 4))
    tensor[0, 0] = -(alpha**2) + beta_down @ beta
    tensor[0, 1:] = tensor[1:, 0] = beta_down
    tensor[1:, 1:] = gamma * np.eye(3)
    metric["tensor"][:] = tensor[:, :, None, None, None, None]
    u_up = np.array([1.0, *-beta]) / alpha
    return metric, u_up


def dust_at_eulerian_rest(metric, u_up, density):
    energy = copy.deepcopy(metric)
    energy.update(type="Stress-Energy", index="contravariant")
    energy["tensor"][:] = (density * np.outer(u_up, u_up))[:, :, None, None, None, None]
    return energy


CURVED_FRAMES = [
    pytest.param(1.0, 1.0, [0, 0, 0], id="flat-control"),
    pytest.param(0.5, 1.0, [0, 0, 0], id="lapse"),
    pytest.param(1.0, 2.0, [0, 0, 0], id="spatial-scale"),
    pytest.param(0.7, 1.5, [0.2, -0.1, 0.3], id="lapse-scale-shift"),
]


@pytest.mark.parametrize(("alpha", "gamma", "beta"), CURVED_FRAMES)
@pytest.mark.parametrize(
    ("condition", "factor"), [("Null", 0.5), ("Weak", 1.0), ("Strong", 0.5)]
)
def test_conditions_are_frame_independent_in_a_curved_metric(
    alpha, gamma, beta, condition, factor
):
    """Observer-measured values must not depend on the coordinates chosen.

    Dust of density rho at rest in the Eulerian frame gives T_ab V^a V^b = rho
    for the observer at rest, half of that for the null vector (1, n)/sqrt(2),
    and rho/2 for the strong condition. Lowering the orthonormal-frame
    components with the coordinate metric instead of Minkowski would scale
    these by powers of the lapse and spatial scale.
    """
    density = 3.0
    metric, u_up = adm_constant_metric(alpha, gamma, beta)
    energy = dust_at_eulerian_rest(metric, u_up, density)
    values = get_energy_conditions(energy, metric, condition, 8, 1, False)
    np.testing.assert_allclose(values, factor * density, rtol=1e-12)


def test_timelike_speeds_include_rest_and_exclude_the_null_limit():
    num_time = 5
    field = generate_uniform_field("timelike", 12, num_time)
    speed = np.linalg.norm(field[1:], axis=0) / field[0]
    expected = np.linspace(0, 1, num_time, endpoint=False)
    np.testing.assert_allclose(speed, np.broadcast_to(expected, speed.shape))
    assert speed.min() == 0
    assert speed.max() < 1


def test_null_vectors_are_exactly_null():
    field = generate_uniform_field("nulllike", 12, 1)
    speed = np.linalg.norm(field[1:], axis=0) / field[0]
    np.testing.assert_allclose(speed, 1, rtol=1e-14)


def shift_flow_expansion(spacing, rate, order):
    """Expansion of Eulerian observers for the linear shift beta^x = rate * x."""
    n = 9
    metric = metric_get_minkowski([5, n, 1, 1], [1, spacing, 1, 1])
    x = (np.arange(n) * spacing).reshape(1, n, 1, 1)
    beta = rate * x
    metric["tensor"][0, 0] = -1 + beta**2
    metric["tensor"][0, 1] = metric["tensor"][1, 0] = beta
    expansion, shear, vorticity = get_scalars(metric, diff_order=order)
    trim = 2
    return expansion[trim:-trim, trim:-trim], shear, vorticity


@pytest.mark.parametrize("order", ["second", "fourth"])
@pytest.mark.parametrize("spacing", [0.25, 1.0, 4.0])
def test_expansion_uses_the_supplied_spatial_spacing(order, spacing):
    """The expansion of a given physical flow must not depend on grid spacing.

    Upstream hard-codes unit spacing, so the same physical field sampled on a
    0.25 m grid would report an expansion four times too small.
    """
    rate = 0.05
    expansion, _, _ = shift_flow_expansion(spacing, rate, order)
    # Divergence of u^x = -beta^x, with sqrt(-g) = 1 for these metrics.
    np.testing.assert_allclose(expansion, -rate, rtol=1e-9)
