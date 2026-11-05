import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from warplab.visualiser.utils.plot_themed_surface import plot_themed_surface

DARK_BACKGROUND = "#0e1117"
DARK_TEXT = "#e6edf3"


@pytest.fixture
def ax():
    fig = plt.figure()
    yield fig.add_subplot(projection="3d")
    plt.close(fig)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def heights(shape=(5, 7)):
    return np.arange(np.prod(shape), dtype=float).reshape(shape)


def test_height_array_plots_x_along_its_first_axis(ax):
    surface = plot_themed_surface(heights((5, 7)), ax=ax, colorbar=False)
    # An (x, y) array of 5 by 7 has 4 by 6 cells; the axes span 0..4 and 0..6.
    assert len(surface.get_facecolor()) == 24
    assert 4 <= ax.get_xlim3d()[1] < 5
    assert 6 <= ax.get_ylim3d()[1] < 7


def test_height_values_set_the_vertical_range(ax):
    """The range contains the data; newer Matplotlib also adds a small margin."""
    plot_themed_surface(heights((5, 7)), ax=ax, colorbar=False)
    low, high = ax.get_zlim3d()
    assert low <= 0
    assert high >= 34


def test_singleton_dimensions_are_squeezed(ax):
    plot_themed_surface(heights((5, 7)).reshape(1, 5, 7, 1), ax=ax, colorbar=False)
    assert 4 <= ax.get_xlim3d()[1] < 5
    assert 6 <= ax.get_ylim3d()[1] < 7


def test_coordinate_arrays_are_used_as_given(ax):
    x, y = np.meshgrid(np.linspace(10, 20, 6), np.linspace(-5, 5, 4))
    plot_themed_surface(x, y, x + y, ax=ax, colorbar=False)
    assert 9 < ax.get_xlim3d()[0] <= 10
    assert 20 <= ax.get_xlim3d()[1] < 21
    assert -6 < ax.get_ylim3d()[0] <= -5
    assert 5 <= ax.get_ylim3d()[1] < 6


@pytest.mark.parametrize("count", [0, 2, 4])
def test_wrong_number_of_arrays_is_rejected(ax, count):
    arrays = [heights((3, 3))] * count
    with pytest.raises(ValueError, match="X, Y and Z"):
        plot_themed_surface(*arrays, ax=ax)


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4)])
def test_height_array_must_be_two_dimensional(ax, shape):
    with pytest.raises(ValueError, match="two-dimensional"):
        plot_themed_surface(heights(shape), ax=ax)


def test_invalid_arguments_leave_no_figure_behind():
    plt.close("all")
    with pytest.raises(ValueError):
        plot_themed_surface(heights((5,)))
    with pytest.raises(ValueError):
        plot_themed_surface()
    assert plt.get_fignums() == []


def test_dark_theme_is_the_default(ax):
    surface = plot_themed_surface(heights(), ax=ax, colorbar=False)
    assert surface.cmap.name == "magma"
    assert to_hex(ax.figure.patch.get_facecolor()) == DARK_BACKGROUND
    assert to_hex(ax.get_facecolor()) == DARK_BACKGROUND
    assert to_hex(ax.xaxis.label.get_color()) == DARK_TEXT


def test_light_theme(ax):
    surface = plot_themed_surface(heights(), ax=ax, dark=False, colorbar=False)
    assert surface.cmap.name == "viridis"
    assert to_hex(ax.get_facecolor()) == "#ffffff"
    assert to_hex(ax.xaxis.label.get_color()) == "#000000"


def test_colormap_and_background_can_be_overridden(ax):
    surface = plot_themed_surface(
        heights(), ax=ax, cmap="plasma", background_color="#102030", colorbar=False
    )
    assert surface.cmap.name == "plasma"
    assert to_hex(ax.get_facecolor()) == "#102030"
    assert to_hex(ax.figure.patch.get_facecolor()) == "#102030"


def test_title_labels_and_view(ax):
    plot_themed_surface(
        heights(),
        ax=ax,
        title="T",
        xlabel="a",
        ylabel="b",
        zlabel="c",
        elev=10,
        azim=20,
        colorbar=False,
    )
    assert ax.get_title() == "T"
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("a", "b", "c")
    assert (ax.elev, ax.azim) == (10, 20)


def test_default_title_labels_and_view(ax):
    plot_themed_surface(heights(), ax=ax, colorbar=False)
    assert ax.get_title() == ""
    assert (ax.get_xlabel(), ax.get_ylabel(), ax.get_zlabel()) == ("x", "y", "z")
    assert (ax.elev, ax.azim) == (28, -135)


@pytest.mark.parametrize(("colorbar", "axes_count"), [(True, 2), (False, 1)])
def test_colorbar_is_optional(ax, colorbar, axes_count):
    plot_themed_surface(heights(), ax=ax, colorbar=colorbar)
    assert len(ax.figure.axes) == axes_count


def test_extra_keywords_reach_the_surface(ax):
    surface = plot_themed_surface(heights(), ax=ax, alpha=0.5, colorbar=False)
    assert surface.get_alpha() == 0.5


def test_styling_is_applied(ax):
    """The helper guards its styling with bare excepts, so check it took effect."""
    plot_themed_surface(
        heights(), ax=ax, pane_alpha=0.2, grid_alpha=0.3, colorbar=False
    )
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert tuple(axis.pane.get_facecolor()) == (1, 1, 1, 0.2)
        assert axis._axinfo["grid"]["color"] == (1, 1, 1, 0.3)
        assert axis._axinfo["grid"]["linewidth"] == 0.8
    assert isinstance(ax.zaxis.get_major_locator(), MaxNLocator)
    assert ax.zaxis.get_major_locator()._nbins == 4
    formatter = ax.zaxis.get_major_formatter()
    assert isinstance(formatter, ScalarFormatter)
    assert formatter._powerlimits == (-2, 2)
    assert to_hex(ax.zaxis.get_offset_text().get_color()) == DARK_TEXT
    x_length, y_length, z_length = ax.get_box_aspect()
    assert x_length == pytest.approx(y_length)
    assert x_length == pytest.approx(2 * z_length)


def test_light_theme_styling_is_applied(ax):
    plot_themed_surface(heights(), ax=ax, dark=False, colorbar=False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        assert tuple(axis.pane.get_facecolor())[:3] == (0, 0, 0)
    assert axis._axinfo["grid"]["color"][:3] == pytest.approx(
        (0.502, 0.502, 0.502), abs=1e-3
    )


def test_supplied_axes_are_used_and_not_shown(ax, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: pytest.fail("show must not be called"))
    surface = plot_themed_surface(heights(), ax=ax, colorbar=False)
    assert surface in ax.collections


def test_created_axes_are_3d_sized_and_shown_once(monkeypatch):
    calls = []
    monkeypatch.setattr(plt, "show", lambda: calls.append(1))
    surface = plot_themed_surface(heights(), figsize=(4, 3), colorbar=False)
    assert calls == [1]
    assert surface.axes.name == "3d"
    assert tuple(surface.axes.figure.get_size_inches()) == (4, 3)


def raising(*args, **kwargs):
    raise RuntimeError("this Matplotlib version does not support the call")


@pytest.mark.parametrize(
    "step",
    ["locator", "formatter", "offset text", "pane colour", "grid style", "box aspect"],
)
def test_a_failing_cosmetic_step_never_breaks_the_plot(ax, monkeypatch, step):
    """Each styling step is guarded so an older or newer Matplotlib still plots."""
    if step == "locator":
        monkeypatch.setattr(ax.zaxis, "set_major_locator", raising)
    elif step == "formatter":
        monkeypatch.setattr(ax.zaxis, "set_major_formatter", raising)
    elif step == "offset text":
        monkeypatch.setattr(ax.zaxis, "get_offset_text", raising)
    elif step == "pane colour":
        monkeypatch.setattr(ax.xaxis, "set_pane_color", raising)
    elif step == "grid style":
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            monkeypatch.setattr(axis, "_axinfo", {})
    else:
        monkeypatch.setattr(ax, "set_box_aspect", raising)
    surface = plot_themed_surface(heights(), ax=ax, colorbar=False)
    assert surface in ax.collections
