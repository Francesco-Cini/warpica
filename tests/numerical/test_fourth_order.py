import numpy as np
import pytest

from warpica.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warpica.solver.utils.c4_inv import c4_inv
from warpica.solver.utils.ricci_t import ricci_t
from warpica.solver.utils.take_finite_difference_1 import take_finite_difference_1 as d1
from warpica.solver.utils.take_finite_difference_2 import take_finite_difference_2 as d2
from warpica.units.universal_constants.c import c

pytestmark = pytest.mark.numerical


def test_fourth_order_convergence():
    errors = []
    for n in (33, 65):
        x = np.linspace(0, 2 * np.pi, n)
        field = np.sin(x).reshape(1, n, 1, 1)
        delta = [1, x[1] - x[0], 1, 1]
        errors.append(
            [
                np.max(abs(d1(field, 1, delta)[0, 2:-2, 0, 0] - np.cos(x[2:-2]))),
                np.max(abs(d2(field, 1, 1, delta)[0, 2:-2, 0, 0] + np.sin(x[2:-2]))),
            ]
        )
    np.testing.assert_allclose(np.array(errors[0]) / errors[1], 16, rtol=0.02)


def test_curved_static_lapse():
    x = np.linspace(-0.5, 0.5, 17).reshape(1, 17, 1, 1)
    f = 1 + x * x
    gl = metric_get_minkowski([1, 17, 1, 1])["tensor"]
    gl[0, 0] = -f * f
    ricci = np.asarray(ricci_t(c4_inv(gl), gl, [1, 1 / 16, 1, 1]))
    np.testing.assert_allclose(ricci[0, 0, :, 2:-2], (2 * f)[:, 2:-2], atol=1e-11)
    np.testing.assert_allclose(ricci[1, 1, :, 2:-2], (-2 / f)[:, 2:-2], atol=1e-11)


def test_time_dependent_flrw():
    t = np.arange(9).reshape(9, 1, 1, 1) * 0.1
    gl = metric_get_minkowski([9, 1, 1, 1])["tensor"]
    for i in range(1, 4):
        gl[i, i] = (1 + 0.2 * t) ** 2
    ricci = np.asarray(ricci_t(c4_inv(gl), gl, [0.1, 1, 1, 1]))
    np.testing.assert_allclose(ricci[0, 0, 2:-2], 0, atol=1e-28)
    for i in range(1, 4):
        np.testing.assert_allclose(ricci[i, i, 2:-2], 2 * 0.2**2 / c() ** 2, rtol=1e-10)
