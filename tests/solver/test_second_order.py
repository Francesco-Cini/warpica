import itertools

import numpy as np
import pytest

from warpica.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warpica.solver.get_energy_tensor import get_energy_tensor
from warpica.solver.utils.second_order.c4_inv_2 import c4_inv_2
from warpica.solver.utils.second_order.c_det_2 import c_det_2
from warpica.solver.utils.second_order.ein_e_2 import ein_e_2
from warpica.solver.utils.second_order.ricci_t_2 import ricci_t_2
from warpica.solver.utils.second_order.ricci_t_mem_2 import ricci_t_mem_2
from warpica.solver.utils.second_order.take_finite_difference_1_2 import (
    take_finite_difference_1_2 as d1,
)
from warpica.solver.utils.second_order.take_finite_difference_2_2 import (
    take_finite_difference_2_2 as d2,
)
from warpica.units.universal_constants.c import c
from warpica.units.universal_constants.G import G


@pytest.mark.parametrize("nested", [False, True])
def test_pointwise_inverse_and_determinant(nested):
    rng = np.random.default_rng(17)
    matrices = rng.normal(size=(3, 5, 4, 4)) + 5 * np.eye(4)
    components = np.moveaxis(matrices, (-2, -1), (0, 1)).copy()
    original = components.copy()
    cells = (
        [[components[i, j] for j in range(4)] for i in range(4)]
        if nested
        else components
    )
    np.testing.assert_allclose(c_det_2(cells), np.linalg.det(matrices))
    inverse = np.moveaxis(c4_inv_2(cells), (0, 1), (-2, -1))
    np.testing.assert_allclose(inverse, np.linalg.inv(matrices), atol=1e-14)
    np.testing.assert_array_equal(components, original)


@pytest.mark.parametrize("axis", range(4))
def test_unmixed_derivatives_and_copied_boundaries(axis):
    delta = np.array([0.2, 0.3, 0.4, 0.5])
    coords = np.indices((5, 6, 7, 8))[axis] * delta[axis]
    field = coords**2
    expected = 2 * coords
    first, last, inside_first, inside_last = (
        [slice(None)] * 4,
        [slice(None)] * 4,
        [slice(None)] * 4,
        [slice(None)] * 4,
    )
    first[axis], last[axis], inside_first[axis], inside_last[axis] = 0, -1, 1, -2
    expected[tuple(first)] = expected[tuple(inside_first)]
    expected[tuple(last)] = expected[tuple(inside_last)]
    np.testing.assert_allclose(d1(field, axis, delta), expected, atol=1e-13)
    np.testing.assert_allclose(d2(field, axis, axis, delta), 2, atol=1e-13)


@pytest.mark.parametrize("a,b", list(itertools.permutations(range(4), 2)))
def test_all_mixed_derivatives_and_zero_boundaries(a, b):
    delta = np.array([0.2, 0.3, 0.4, 0.5])
    coords = np.indices((5, 6, 7, 8))
    field = coords[a] * delta[a] * coords[b] * delta[b]
    expected = np.zeros_like(field)
    interior = [slice(None)] * 4
    interior[a] = interior[b] = slice(1, -1)
    expected[tuple(interior)] = 1
    np.testing.assert_allclose(d2(field, a, b, delta), expected, atol=1e-13)


@pytest.mark.parametrize("axis", range(4))
@pytest.mark.parametrize("size", [1, 2])
def test_short_axes_return_zero(axis, size):
    shape = [4] * 4
    shape[axis] = size
    field = np.arange(np.prod(shape)).reshape(shape)
    assert not np.any(d1(field, axis, np.ones(4)))
    for other in range(4):
        assert not np.any(d2(field, axis, other, np.ones(4)))


def test_derivative_second_order_convergence():
    errors = []
    for n in (33, 65):
        x = np.linspace(0, 2 * np.pi, n)
        field = np.sin(x).reshape(1, n, 1, 1)
        delta = [1, x[1] - x[0], 1, 1]
        errors.append(
            [
                np.max(np.abs(d1(field, 1, delta)[0, 1:-1, 0, 0] - np.cos(x[1:-1]))),
                np.max(np.abs(d2(field, 1, 1, delta)[0, 1:-1, 0, 0] + np.sin(x[1:-1]))),
            ]
        )
    np.testing.assert_allclose(np.array(errors[0]) / errors[1], 4, rtol=0.01)


@pytest.mark.parametrize("gpu", [None, "numpy"])
def test_public_minkowski_pipeline(gpu):
    metric = metric_get_minkowski(np.array([3, 4, 4, 4]))
    energy = get_energy_tensor(metric, "second", gpu=gpu)
    assert energy["order"] == "second"
    assert energy["index"] == "contravariant"
    np.testing.assert_array_equal(energy["tensor"], np.zeros((4, 4, 3, 4, 4, 4)))


def test_energy_index_contraction_and_nondefault_units():
    rng = np.random.default_rng(18)
    einstein = rng.normal(size=(4, 4, 2, 3))
    inverse = rng.normal(size=(4, 4, 2, 3))
    units = [2, 3, 5]
    grav = G() * 5**2 * 3 / 2**3
    speed = c() * 5 / 2
    expected = np.einsum("ab...,am...,bn...->mn...", einstein, inverse, inverse)
    expected *= speed**4 / (8 * np.pi * grav)
    np.testing.assert_allclose(ein_e_2(einstein, inverse, units), expected, rtol=1e-13)


def test_time_dependent_flat_flrw_and_memory_convention():
    # ds^2 = -c^2 dt^2 + a(t)^2 dx_i dx_i, a(t) = 1 + q*t.
    # R_00 = 0 and R_ii = 2*q^2/c^2 in the ct coordinate basis.
    metric = metric_get_minkowski(np.array([7, 1, 1, 1]))
    q = 0.2
    t = np.arange(7).reshape(7, 1, 1, 1) * 0.1
    for i in range(1, 4):
        metric["tensor"][i, i] = (1 + q * t) ** 2
    gl = metric["tensor"]
    gu = c4_inv_2(gl)
    delta = np.array([0.1, 1, 1, 1])
    ricci = np.asarray(ricci_t_2(gu, gl, delta))
    np.testing.assert_allclose(ricci[0, 0, 1:-1], 0, atol=1e-29)
    for i in range(1, 4):
        np.testing.assert_allclose(ricci[i, i, 1:-1], 2 * q**2 / c() ** 2, rtol=1e-10)
    delta[0] *= c()
    memory = np.asarray(ricci_t_mem_2(gu, gl, delta))
    np.testing.assert_allclose(memory, ricci, rtol=1e-10, atol=1e-29)


def test_static_curved_metric_convergence():
    # ds^2 = -f(x)^2 d(ct)^2 + dx^2 + dy^2 + dz^2, f = 1+x^2.
    # R_00 = f*f'' and R_11 = -f''/f.
    errors = []
    for n in (17, 33):
        x = np.linspace(-0.5, 0.5, n).reshape(1, n, 1, 1)
        f = 1 + x**2
        metric = metric_get_minkowski(np.array([1, n, 1, 1]))
        metric["tensor"][0, 0] = -(f**2)
        gl = metric["tensor"]
        ricci = np.asarray(ricci_t_2(c4_inv_2(gl), gl, [1, 1 / (n - 1), 1, 1]))
        interior = (0, slice(1, -1), 0, 0)
        errors.append(
            max(
                np.max(np.abs((ricci[0, 0] - 2 * f)[interior])),
                np.max(np.abs((ricci[1, 1] + 2 / f)[interior])),
            )
        )
    assert 3.8 < errors[0] / errors[1] < 4.2


def test_public_curved_metric_energy():
    n = 33
    x = np.linspace(-0.5, 0.5, n).reshape(1, n, 1, 1)
    f = 1 + x**2
    metric = metric_get_minkowski(
        np.array([1, n, 1, 1]), np.array([1, 1 / (n - 1), 1, 1])
    )
    metric["tensor"][0, 0] = -(f**2)
    energy = np.asarray(get_energy_tensor(metric, "second")["tensor"])
    factor = c() ** 4 / (8 * np.pi * G())
    # This metric has T^22 = T^33 = (c^4 / 8*pi*G) * f''/f.
    for i in (2, 3):
        np.testing.assert_allclose(
            energy[i, i, 0, 1:-1, 0, 0] / factor,
            (2 / f)[0, 1:-1, 0, 0],
            rtol=0.003,
        )
    assert np.all(np.isfinite(energy))
    np.testing.assert_array_equal(energy, energy.swapaxes(0, 1))


def test_output_preserves_tuple_grid_scaling():
    metric = metric_get_minkowski([1, 3, 3, 3], (1, 2, 3, 4))
    result = get_energy_tensor(metric, "second")
    assert result["scaling"] == (1, 2, 3, 4)
