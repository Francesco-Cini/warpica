import itertools

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from warpica.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warpica.visualiser import plot_tensor, plot_three_plus_one
from warpica.visualiser.utils.get_slice_data import get_slice_data
from warpica.visualiser.utils.label_cartesian_axis import label_cartesian_axis
from warpica.visualiser.utils.plot_component import plot_component


@pytest.mark.parametrize("planes", list(itertools.combinations(range(1, 5), 2)))
def test_slice_orientation(planes):
    metric = metric_get_minkowski([3, 4, 5, 6])
    data = np.arange(3 * 4 * 5 * 6).reshape(3, 4, 5, 6)
    metric["tensor"][0, 0] = data
    indices = get_slice_data(planes, [1, 2], metric)
    expected = [slice(None)] * 4
    expected[planes[0] - 1] = 0
    expected[planes[1] - 1] = 1
    np.testing.assert_array_equal(data[indices], data[tuple(expected)])
    assert label_cartesian_axis(planes) == tuple(
        x for i, x in enumerate("txyz", 1) if i not in planes
    )


@pytest.mark.parametrize(
    "planes,locations",
    [([1, 1], [1, 1]), ([0, 4], [1, 1]), ([1, 4], [0, 1]), ([1, 4], [1, 99])],
)
def test_bad_slices(planes, locations):
    with pytest.raises(ValueError):
        get_slice_data(planes, locations, metric_get_minkowski([1, 3, 3, 3]))


def test_tensor_and_decomposition_return_figures_without_show(monkeypatch, tmp_path):
    monkeypatch.setattr(
        plt, "show", lambda: pytest.fail("Plotting library must not call show")
    )
    metric = metric_get_minkowski([1, 5, 6, 7])
    before = metric["tensor"].copy()
    for plotter, shape in (
        (plot_tensor, (4, 4)),
        (plot_three_plus_one, (2, 5)),
    ):
        fig, axes = plotter(metric)
        assert axes.shape == shape
        fig.canvas.draw()
        fig.savefig(tmp_path / (plotter.__name__ + ".png"))
        plt.close(fig)
    np.testing.assert_array_equal(metric["tensor"], before)


def test_component_data_and_opacity():
    field = np.arange(6).reshape(2, 3)
    fig, ax = plot_component(field, "test", "x", "y", alpha=0.4)
    np.testing.assert_array_equal(ax.collections[0].get_array(), field.T)
    assert ax.collections[0].get_alpha() == 0.4
    plt.close(fig)
