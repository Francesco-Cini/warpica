import numpy as np
import pytest

from warplab.analyser import eval_metric, get_scalars
from warplab.metrics import metric_get_minkowski
from warplab.solver import get_energy_tensor
from warplab.solver.utils.verify_grid_size import verify_grid_size

AXES = ["time", "x", "y", "z"]


def shape_with(axis, size, others=9):
    shape = [others] * 4
    shape[axis] = size
    return shape


@pytest.mark.parametrize("axis", range(4))
@pytest.mark.parametrize("size", [1, 5, 6, 20])
def test_fourth_order_accepts_one_sample_or_at_least_five(axis, size):
    verify_grid_size(shape_with(axis, size), "fourth")


@pytest.mark.parametrize("axis", range(4))
@pytest.mark.parametrize("size", [2, 3, 4])
def test_fourth_order_rejects_two_to_four_samples(axis, size):
    with pytest.raises(ValueError) as error:
        verify_grid_size(shape_with(axis, size), "fourth")
    message = str(error.value)
    assert f"along the {AXES[axis]} axis" in message
    assert f"got {size}" in message
    assert "at least 5" in message


@pytest.mark.parametrize("axis", range(4))
@pytest.mark.parametrize("size", [1, 3, 4, 20])
def test_second_order_accepts_one_sample_or_at_least_three(axis, size):
    verify_grid_size(shape_with(axis, size), "second")


@pytest.mark.parametrize("axis", range(4))
def test_second_order_rejects_two_samples(axis):
    with pytest.raises(ValueError, match="at least 3"):
        verify_grid_size(shape_with(axis, 2), "second")


@pytest.mark.parametrize("shape", [(1, 1, 1, 1), (1, 5, 5, 5), (5, 1, 1, 1)])
def test_static_and_homogeneous_grids_are_accepted(shape):
    verify_grid_size(shape, "fourth")
    verify_grid_size(shape, "second")


@pytest.mark.parametrize("order", ["Fourth", "FOURTH", "Second"])
def test_order_is_compared_case_insensitively(order):
    with pytest.raises(ValueError):
        verify_grid_size((2, 9, 9, 9), order)


@pytest.mark.parametrize("order", [None, "sixth", 4])
def test_unrecognised_orders_are_left_for_the_caller_to_reject(order):
    verify_grid_size((2, 9, 9, 9), order)


def test_the_message_suggests_second_order_only_when_it_would_work():
    with pytest.raises(ValueError, match="diff_order='second'"):
        verify_grid_size((3, 9, 9, 9), "fourth")
    with pytest.raises(ValueError) as error:
        verify_grid_size((2, 9, 9, 9), "fourth")
    assert "diff_order='second'" not in str(error.value)


def test_the_first_offending_axis_is_reported():
    with pytest.raises(ValueError, match="along the x axis"):
        verify_grid_size((9, 3, 4, 9), "fourth")


def minkowski(shape):
    return metric_get_minkowski(shape)


ENTRY_POINTS = {
    "get_energy_tensor": lambda metric, order: get_energy_tensor(metric, order),
    "get_scalars": lambda metric, order: get_scalars(metric, diff_order=order),
    "eval_metric": lambda metric, order: eval_metric(
        metric, num_angular_vec=4, num_time_vec=2, diff_order=order
    ),
}


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
@pytest.mark.parametrize(
    ("shape", "axis"), [([3, 9, 9, 9], "time"), ([1, 4, 9, 9], "x")]
)
def test_entry_points_reject_a_short_axis_instead_of_returning_zeros(name, shape, axis):
    with pytest.raises(ValueError, match=f"along the {axis} axis"):
        ENTRY_POINTS[name](minkowski(shape), "fourth")


@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_second_order_runs_where_fourth_order_is_too_short(name):
    result = ENTRY_POINTS[name](minkowski([3, 5, 5, 5]), "second")
    assert result is not None


@pytest.mark.parametrize("shape", [[1, 9, 9, 9], [9, 1, 1, 1], [1, 5, 1, 5]])
@pytest.mark.parametrize("name", list(ENTRY_POINTS))
def test_single_sample_axes_still_work(name, shape):
    ENTRY_POINTS[name](minkowski(shape), "fourth")


def test_a_moving_bubble_on_three_time_slices_is_refused():
    """Regression: with 3 slices the time derivative was silently dropped."""
    from warplab.metrics import metric_get_alcubierre

    metric = metric_get_alcubierre(
        np.array([3, 9, 9, 9]),
        np.array([1e-9, 2.5, 2.5, 2.5]),
        0.9,
        0.8,
        4,
        np.array([1e-9, 0.5, 0.5, 0.5]),
    )
    with pytest.raises(ValueError, match="time axis"):
        get_energy_tensor(metric, "fourth")
