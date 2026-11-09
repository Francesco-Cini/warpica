import numpy as np
import pytest

from warpica.solver.utils.c4_inv import c4_inv
from warpica.solver.utils.ricci_t import ricci_t
from warpica.solver.utils.second_order.ricci_t_2 import ricci_t_2
from warpica.units.universal_constants.c import c

pytestmark = pytest.mark.numerical


@pytest.mark.parametrize("solver,ratio", [(ricci_t_2, 4), (ricci_t, 16)])
def test_nondiagonal_conformal_metric(solver, ratio):
    # g = exp(2 phi) h, with constant non-diagonal h and linear phi.
    # R_ab = 2 phi_a phi_b - 2 h_ab h^cd phi_c phi_d.
    h = np.array(
        [
            [-1, 0.1, 0.02, 0],
            [0.1, 1.2, 0.1, 0],
            [0.02, 0.1, 1.1, 0.1],
            [0, 0, 0.1, 0.9],
        ]
    )
    gradient = np.array([0.3, 0.2, 0, 0])
    exact = 2 * np.outer(gradient, gradient) - 2 * h * (
        gradient @ np.linalg.inv(h) @ gradient
    )
    errors = []
    for n in (9, 17):
        t, x = np.meshgrid(
            np.linspace(-0.5, 0.5, n), np.linspace(-0.5, 0.5, n), indexing="ij"
        )
        phi = 0.3 * t + 0.2 * x
        gl = (
            h[:, :, None, None, None, None]
            * np.exp(2 * phi)[None, None, :, :, None, None]
        )
        result = np.asarray(
            solver(c4_inv(gl), gl, [1 / (n - 1) / c(), 1 / (n - 1), 1, 1])
        )
        errors.append(np.max(abs(result[:, :, n // 2, n // 2, 0, 0] - exact)))
    assert ratio * 0.85 < errors[0] / errors[1] < ratio * 1.15
