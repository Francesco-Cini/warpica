import itertools

import numpy as np
import pytest

from warpica.solver.utils.take_finite_difference_2 import (
    take_finite_difference_2 as derivative,
)


@pytest.mark.parametrize("axis", range(4))
def test_constant_linear_and_quadratic(axis):
    x = np.indices((7, 8, 9, 10))[axis]
    for field, expected in ((np.ones_like(x), 0), (x, 0), (x * x, 2)):
        np.testing.assert_allclose(
            derivative(field, axis, axis, [1] * 4), expected, atol=1e-12
        )


@pytest.mark.parametrize("a,b", list(itertools.permutations(range(4), 2)))
def test_mixed_derivatives(a, b):
    delta = np.array([0.2, 0.3, 0.4, 0.5])
    coords = np.indices((7, 8, 9, 10))
    field = coords[a] * delta[a] * coords[b] * delta[b]
    expected = np.zeros_like(field)
    interior = [slice(None)] * 4
    interior[a] = interior[b] = slice(2, -2)
    expected[tuple(interior)] = 1
    np.testing.assert_allclose(derivative(field, a, b, delta), expected, atol=1e-12)
