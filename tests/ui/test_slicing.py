import numpy as np
import pytest

from warpica.ui.core.slicing import (
    AXIS_NAMES,
    PLANES,
    axis_extent,
    grid_axes,
    take_slice,
)


def test_axes_are_centred_on_zero():
    t, x, y, z = grid_axes((1, 5, 5, 3), [1.0, 0.5, 0.5, 2.0])

    np.testing.assert_allclose(t, [0.0])
    np.testing.assert_allclose(x, [-1.0, -0.5, 0.0, 0.5, 1.0])
    np.testing.assert_allclose(z, [-2.0, 0.0, 2.0])


def test_an_axis_with_an_even_number_of_points_is_symmetric():
    (axis,) = grid_axes((4,), [1.0])

    np.testing.assert_allclose(axis, [-1.5, -0.5, 0.5, 1.5])


def test_the_axes_match_the_grid_the_metric_page_builds():
    # The page builds coordinates as (index + 1) * spacing - centre, with the
    # centre at (size + 1) / 2 * spacing.
    size, spacing = 7, 0.25
    centre = (size + 1) / 2 * spacing
    expected = (np.arange(size) + 1) * spacing - centre

    (axis,) = grid_axes((size,), [spacing])

    np.testing.assert_allclose(axis, expected)


def test_the_extent_reaches_half_a_cell_beyond_the_first_and_last_point():
    assert axis_extent(np.array([-1.0, -0.5, 0.0, 0.5, 1.0])) == (-1.25, 1.25)


def test_a_single_point_gets_a_cell_one_unit_wide():
    assert axis_extent(np.array([0.0])) == (-0.5, 0.5)


@pytest.fixture
def field():
    # Every value is its own flat index, so a slice shows where it came from.
    return np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)


def test_the_x_y_plane_holds_z_fixed(field):
    result = take_slice(field, "x-y", position=2, time_index=1)

    np.testing.assert_array_equal(result, field[1, :, :, 2])
    assert result.shape == (3, 4)


def test_the_x_z_plane_holds_y_fixed(field):
    result = take_slice(field, "x-z", position=1)

    np.testing.assert_array_equal(result, field[0, :, 1, :])
    assert result.shape == (3, 5)


def test_the_y_z_plane_holds_x_fixed(field):
    result = take_slice(field, "y-z", position=0)

    np.testing.assert_array_equal(result, field[0, 0, :, :])
    assert result.shape == (4, 5)


def test_each_plane_names_two_different_free_axes_and_one_fixed_axis():
    for first, second, fixed in PLANES.values():
        assert {first, second, fixed} == {1, 2, 3}
    assert AXIS_NAMES == ("t", "x", "y", "z")
