import numpy as np
import pytest

from warpica.solver.utils.take_finite_difference_1 import (
    take_finite_difference_1 as derivative,
)


@pytest.mark.parametrize("axis", range(4))
def test_polynomial_derivative_and_boundaries(axis):
    x = np.indices((7, 8, 9, 10))[axis] * 0.2
    expected = 4 * x**3
    for target, source in ((0, 2), (1, 2), (-1, -3), (-2, -3)):
        dst, src = [slice(None)] * 4, [slice(None)] * 4
        dst[axis], src[axis] = target, source
        expected[tuple(dst)] = expected[tuple(src)]
    np.testing.assert_allclose(derivative(x**4, axis, [0.2] * 4), expected, atol=1e-12)


@pytest.mark.parametrize("axis", range(4))
def test_short_axis_returns_zero(axis):
    shape = [5] * 4
    shape[axis] = 4
    assert not np.any(derivative(np.ones(shape), axis, [1] * 4))
