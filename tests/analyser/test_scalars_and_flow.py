import numpy as np
import pytest

from warpica.analyser.get_momentum_flow_lines import get_momentum_flow_lines
from warpica.analyser.get_scalars import get_scalars
from warpica.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warpica.solver.utils.trilinear_interp import trilinear_interp
from warpica.units.universal_constants.c import c


@pytest.mark.parametrize("order,trim", [("second", 1), ("fourth", 2)])
def test_flrw_expansion_shear_and_vorticity(order, trim):
    t = np.arange(9).reshape(9, 1, 1, 1) * 0.1
    metric = metric_get_minkowski([9, 1, 1, 1], [0.1, 1, 1, 1])
    for i in range(1, 4):
        metric["tensor"][i, i] = (1 + 0.2 * t) ** 2
    before = metric["tensor"].copy()
    expansion, shear, vorticity = get_scalars(metric, diff_order=order)
    np.testing.assert_allclose(
        expansion[trim:-trim], (3 * 0.2 / (1 + 0.2 * t) / c())[trim:-trim], rtol=1e-11
    )
    np.testing.assert_allclose(shear[trim:-trim], 0, atol=1e-30)
    np.testing.assert_allclose(vorticity, 0, atol=1e-30)
    np.testing.assert_array_equal(metric["tensor"], before)


def test_constant_momentum_flow_includes_start_and_last_point():
    energy = {"index": "contravariant", "tensor": np.zeros((4, 4, 8, 8, 8))}
    energy["tensor"][0, 1] = 1
    paths = get_momentum_flow_lines(energy, [[3], [3], [3]], 0.25, 4, 1)
    expected = np.column_stack((3 + np.arange(5) * 0.25, np.full(5, 3), np.full(5, 3)))
    np.testing.assert_allclose(paths[0], expected)
    assert len(get_momentum_flow_lines(energy, [[3], [3], [3]], 0.25, 0, 1)[0]) == 1


def test_trilinear_interpolation_of_linear_field():
    x, y, z = np.indices((5, 5, 5)) + 1
    np.testing.assert_allclose(
        trilinear_interp(x + 2 * y + 3 * z, [2.3, 2.7, 3.1]), 17, atol=1e-7
    )


@pytest.mark.parametrize("point", [[1, 1, 1], [5, 5, 5], [3, 2, 4]])
def test_interpolation_at_integer_points_including_edges(point):
    field = np.arange(125).reshape(5, 5, 5)
    assert trilinear_interp(field, point) == field[tuple(np.array(point) - 1)]
