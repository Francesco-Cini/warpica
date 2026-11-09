import copy

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from warpica.metrics import metric_get_minkowski
from warpica.visualiser import plot_tensor, plot_three_plus_one
from warpica.visualiser.utils.get_slice_data import get_slice_data
from warpica.visualiser.utils.label_cartesian_axis import label_cartesian_axis
from warpica.visualiser.utils.plot_component import plot_component


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def metric():
    return metric_get_minkowski([1, 5, 5, 5], [1, 2, 3, 4])


def stress_energy(index):
    tensor = metric()
    tensor.update(type="Stress-Energy", index=index, name="dust")
    rng = np.random.default_rng(0)
    tensor["tensor"] = np.array(rng.standard_normal((4, 4, 1, 5, 5, 5)))
    return tensor


# ---- plot_tensor and plot_three_plus_one --------------------------------------

PLOTTERS = [plot_tensor, plot_three_plus_one]


@pytest.mark.parametrize("plotter", PLOTTERS)
def test_a_non_cartesian_tensor_is_rejected(plotter):
    tensor = metric()
    tensor["coords"] = "spherical"
    with pytest.warns(UserWarning, match="Non-cartesian"):
        with pytest.raises(ValueError, match="Invalid tensor"):
            plotter(tensor)


@pytest.mark.parametrize("plotter", PLOTTERS)
def test_a_malformed_tensor_is_rejected_cleanly(plotter):
    """plot_three_plus_one used to fail with a KeyError before verifying."""
    with pytest.warns(UserWarning, match="type does not exist"):
        with pytest.raises(ValueError, match="Invalid tensor"):
            plotter({})
    broken = {key: value for key, value in metric().items() if key != "index"}
    with pytest.warns(UserWarning, match="index: Empty"):
        with pytest.raises(ValueError, match="Invalid tensor"):
            plotter(broken)


@pytest.mark.parametrize("plotter", PLOTTERS)
def test_plotting_does_not_modify_the_tensor(plotter):
    tensor = metric()
    before = copy.deepcopy(tensor)
    plotter(tensor)
    np.testing.assert_array_equal(
        np.array(tensor["tensor"]), np.array(before["tensor"])
    )
    assert tensor["index"] == before["index"]


def test_axes_are_labelled_in_metres_when_the_grid_has_a_spacing():
    _, axes = plot_tensor(metric())
    assert axes[0, 0].get_xlabel() == "x (m)"
    assert axes[0, 0].get_ylabel() == "y (m)"


def test_axes_fall_back_to_grid_indices_without_a_spacing():
    tensor = metric()
    del tensor["scaling"]
    _, axes = plot_tensor(tensor)
    assert axes[0, 0].get_xlabel() == "x (grid index)"
    assert axes[0, 0].get_ylabel() == "y (grid index)"


def test_a_time_axis_is_labelled_in_seconds():
    _, axes = plot_tensor(metric(), sliced_planes=[3, 4], slice_locations=[3, 3])
    assert axes[0, 0].get_xlabel() == "t (s)"
    assert axes[0, 0].get_ylabel() == "x (m)"


@pytest.mark.parametrize(
    ("index", "first", "mixed"),
    [
        ("covariant", "$T_{00}$", "$T_{12}$"),
        ("contravariant", "$T^{00}$", "$T^{12}$"),
        ("mixedupdown", "$T^{0}_{0}$", "$T^{1}_{2}$"),
        ("mixeddownup", "$T_{0}^{0}$", "$T_{1}^{2}$"),
    ],
)
def test_component_titles_show_the_index_positions(index, first, mixed):
    _, axes = plot_tensor(stress_energy(index))
    assert axes[0, 0].get_title() == first
    assert axes[1, 2].get_title() == mixed


def test_a_metric_is_titled_with_g():
    _, axes = plot_tensor(metric())
    assert axes[0, 0].get_title() == "$g_{00}$"


# ---- slice selection --------------------------------------------------------------


@pytest.mark.parametrize("locations", [[1], [1, 2, 3], [np.nan, 1], [1, np.inf]])
def test_slice_locations_must_be_two_finite_numbers(locations):
    with pytest.raises(ValueError, match="two finite slice locations"):
        get_slice_data([1, 4], locations, metric())


@pytest.mark.parametrize(
    ("locations", "expected"), [([2.4, 2.4], 1), ([2.5, 2.5], 2), ([2.6, 3.4], 2)]
)
def test_slice_locations_round_half_up(locations, expected):
    indices = get_slice_data([2, 3], locations, metric())
    assert indices[1] == expected


@pytest.mark.parametrize("plane", [[1], [1, 2, 3], [2, 2], [0, 2], [2, 5]], ids=str)
def test_axis_labels_need_two_distinct_axes_between_one_and_four(plane):
    with pytest.raises(ValueError, match="two distinct axes"):
        label_cartesian_axis(plane)


@pytest.mark.parametrize(
    ("plane", "labels"),
    [((1, 4), ("x", "y")), ((2, 3), ("t", "z")), ((3, 4), ("t", "x"))],
)
def test_axis_labels_name_the_remaining_axes(plane, labels):
    assert label_cartesian_axis(plane) == labels


# ---- plot_component ------------------------------------------------------------------


@pytest.mark.parametrize(
    "values",
    [
        np.ones(4),
        np.ones((2, 2, 2)),
        np.array([[1.0, np.nan], [0, 1]]),
        np.array([[np.inf, 0], [0, 1]]),
    ],
    ids=["1D", "3D", "NaN", "inf"],
)
def test_components_must_be_finite_and_two_dimensional(values):
    with pytest.raises(ValueError, match="finite 2D array"):
        plot_component(values, "title", "x", "y")


@pytest.mark.parametrize("alpha", [-0.1, 1.1, 2])
def test_opacity_must_be_between_zero_and_one(alpha):
    with pytest.raises(ValueError, match="between zero and one"):
        plot_component(np.ones((2, 2)), "title", "x", "y", alpha=alpha)


@pytest.mark.parametrize("alpha", [0, 0.5, 1])
def test_the_opacity_limits_are_accepted(alpha):
    _, ax = plot_component(np.ones((2, 2)), "title", "x", "y", alpha=alpha)
    assert ax.get_title() == "title"
    assert (ax.get_xlabel(), ax.get_ylabel()) == ("x", "y")
