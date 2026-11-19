import matplotlib.pyplot as plt
import numpy as np
import pytest

from warpica.metrics import metric_get_alcubierre, metric_get_minkowski
from warpica.solver import get_energy_tensor
from warpica.ui.core.analysis import FlowLines, momentum_flow_lines
from warpica.ui.core.flow_plot import flow_line_figure
from warpica.ui.core.slicing import grid_axes

SPACING = np.array([1.0, 0.5, 0.5, 0.5])


def solve(size):
    """Return a small Alcubierre metric and its solved energy tensor."""
    grid = np.array(size)
    metric = metric_get_alcubierre(
        grid, (grid + 1) / 2 * SPACING, 0.4, 1.0, 1.0, SPACING
    )
    return metric, get_energy_tensor(metric, "fourth")


@pytest.fixture(scope="module")
def static():
    return solve((1, 7, 7, 7))


@pytest.fixture(scope="module")
def timed():
    # This generator puts the bubble at the middle time step only.
    return solve((5, 7, 7, 7))


def trace(tensor, **overrides):
    arguments = {
        "plane": "x-y",
        "position": 3,
        "seeds_per_side": 3,
        "step": 0.25,
        "max_steps": 40,
    }
    return momentum_flow_lines(tensor, **{**arguments, **overrides})


# --- Tracing ---------------------------------------------------------------


def test_there_is_one_line_for_every_seed(static):
    _, tensor = static

    flow = trace(tensor, seeds_per_side=3)

    assert len(flow.paths) == 9
    for path in flow.paths:
        assert path.ndim == 2 and path.shape[1] == 3


def test_the_flow_records_where_it_was_seeded(static):
    _, tensor = static

    flow = trace(tensor, plane="x-z", position=2, time_index=0)

    assert (flow.plane, flow.position, flow.time_index) == ("x-z", 2, 0)


@pytest.mark.parametrize(
    ("plane", "fixed_column"), [("x-y", 2), ("x-z", 1), ("y-z", 0)]
)
def test_seeds_lie_on_the_chosen_plane(static, plane, fixed_column):
    _, tensor = static
    position = 3
    axes = grid_axes((1, 7, 7, 7), SPACING)

    flow = trace(tensor, plane=plane, position=position)

    expected = axes[fixed_column + 1][position]
    for path in flow.paths:
        assert path[0, fixed_column] == pytest.approx(expected)


def test_seeds_are_spread_evenly_over_the_inside_of_the_plane(static):
    _, tensor = static
    axes = grid_axes((1, 7, 7, 7), SPACING)

    flow = trace(tensor, seeds_per_side=3)

    # The first and last seed sit on the second and second-to-last grid point.
    seeds_x = sorted({round(path[0, 0], 9) for path in flow.paths})
    assert seeds_x == pytest.approx([axes[1][1], 0.0, axes[1][-2]])


def test_a_step_never_moves_a_line_further_than_the_step_size(static):
    _, tensor = static
    step = 0.3

    flow = trace(tensor, step=step, max_steps=30)

    # Distances in grid cells: all three spatial spacings are 0.5 m here.
    for path in flow.paths:
        moves = np.linalg.norm(np.diff(path, axis=0), axis=1) / 0.5
        assert np.all(moves <= step * (1 + 1e-9))


def test_a_larger_step_carries_a_line_further(static):
    _, tensor = static

    short = trace(tensor, step=0.1, max_steps=5)
    long = trace(tensor, step=0.4, max_steps=5)

    travelled_short = sum(np.linalg.norm(p[-1] - p[0]) for p in short.paths)
    travelled_long = sum(np.linalg.norm(p[-1] - p[0]) for p in long.paths)
    assert travelled_long > travelled_short


def test_a_line_takes_at_most_the_requested_number_of_steps(static):
    _, tensor = static

    flow = trace(tensor, max_steps=7)

    assert all(len(path) <= 8 for path in flow.paths)


def test_zero_steps_leaves_every_line_at_its_seed(static):
    _, tensor = static

    flow = trace(tensor, max_steps=0)

    assert all(len(path) == 1 for path in flow.paths)


def test_lines_stay_inside_the_grid(static):
    _, tensor = static
    axes = grid_axes((1, 7, 7, 7), SPACING)

    flow = trace(tensor, max_steps=500, step=1.0)

    # A line may take one step past the last interior point before it stops.
    margin = 1.0 * SPACING[1]
    for path in flow.paths:
        for column in range(3):
            low, high = axes[column + 1][0] - margin, axes[column + 1][-1] + margin
            assert np.all((path[:, column] >= low) & (path[:, column] <= high))


def test_the_same_inputs_give_the_same_lines(static):
    _, tensor = static

    first, second = trace(tensor), trace(tensor)

    for a, b in zip(first.paths, second.paths, strict=True):
        np.testing.assert_array_equal(a, b)


def test_the_tracer_leaves_the_tensor_unchanged(static):
    _, tensor = static
    before = np.array(tensor["tensor"][0][1], copy=True)

    trace(tensor)

    np.testing.assert_array_equal(tensor["tensor"][0][1], before)


# --- Time-dependent grids --------------------------------------------------


def test_a_time_step_with_a_bubble_can_be_traced(timed):
    _, tensor = timed

    flow = trace(tensor, time_index=2)

    assert flow.time_index == 2
    assert len(flow.paths) == 9


def test_a_flat_time_step_has_no_flow_to_trace(timed):
    _, tensor = timed

    with pytest.raises(ValueError, match="zero everywhere"):
        trace(tensor, time_index=0)


# --- Errors ----------------------------------------------------------------


def test_flat_space_has_no_flow_to_trace():
    grid = np.array([1, 7, 7, 7])
    tensor = get_energy_tensor(metric_get_minkowski(grid, SPACING), "fourth")

    with pytest.raises(ValueError, match="zero everywhere"):
        trace(tensor)


@pytest.mark.parametrize("time_index", [-1, 1])
def test_a_time_step_outside_the_grid_is_rejected(static, time_index):
    _, tensor = static

    with pytest.raises(ValueError, match="Time step"):
        trace(tensor, time_index=time_index)


@pytest.mark.parametrize("position", [0, 6, -1, 7])
def test_a_plane_on_the_edge_of_the_grid_is_rejected(static, position):
    _, tensor = static

    with pytest.raises(ValueError, match="inside the grid"):
        trace(tensor, position=position)


def test_there_must_be_at_least_one_seed(static):
    _, tensor = static

    with pytest.raises(ValueError, match="at least one seed"):
        trace(tensor, seeds_per_side=0)


def test_the_step_must_be_positive(static):
    _, tensor = static

    with pytest.raises(ValueError, match="step_size positive"):
        trace(tensor, step=0.0)


def test_one_seed_is_a_valid_request(static):
    _, tensor = static

    flow = trace(tensor, seeds_per_side=1)

    assert len(flow.paths) == 1


# --- The figure ------------------------------------------------------------

AXES = grid_axes((1, 7, 7, 7), SPACING)


def make_flow(paths, plane="x-y", position=3):
    return FlowLines(
        paths=[np.array(p, dtype=float) for p in paths],
        plane=plane,
        position=position,
        time_index=0,
    )


def test_the_figure_has_one_line_and_one_seed_marker_per_path():
    flow = make_flow([[[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 0], [0, 2, 0]]])

    figure = flow_line_figure(flow, AXES)

    # Each path draws a line and a marker for its seed.
    assert len(figure.axes[0].lines) == 4


def test_a_direction_arrow_is_drawn_only_for_lines_long_enough():
    flow = make_flow(
        [
            [[0, 0, 0]],  # one point: nothing to point along
            [[0, 1, 0], [0, 2, 0]],  # two points: too short
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],  # long enough
        ]
    )

    figure = flow_line_figure(flow, AXES)

    assert len(figure.axes[0].texts) == 1


def test_the_figure_is_not_attached_to_pyplot():
    before = plt.get_fignums()

    flow_line_figure(make_flow([[[0, 0, 0], [1, 0, 0]]]), AXES)

    assert plt.get_fignums() == before


def test_the_title_names_the_plane_and_its_position():
    flow = make_flow([[[0, 0, 0], [1, 0, 0]]], plane="x-z", position=3)

    title = flow_line_figure(flow, AXES).axes[0].get_title()

    assert "x-z plane" in title
    assert "y = 0 m" in title


def test_the_axes_cover_the_whole_grid():
    flow = make_flow([[[0, 0, 0], [1, 0, 0]]])

    plot = flow_line_figure(flow, AXES).axes[0]

    assert plot.get_xlim() == pytest.approx((-1.75, 1.75))
    assert plot.get_ylim() == pytest.approx((-1.75, 1.75))
    assert plot.get_xlabel() == "x (m)"
    assert plot.get_ylabel() == "y (m)"


def test_a_background_field_is_drawn_behind_the_lines():
    flow = make_flow([[[0, 0, 0], [1, 0, 0]]])
    background = np.random.default_rng(0).normal(size=(1, 7, 7, 7))

    figure = flow_line_figure(flow, AXES, background)

    plot = figure.axes[0]
    assert len(plot.images) == 1
    low, high = plot.images[0].get_clim()
    assert low == -high


@pytest.mark.parametrize("fill", [0.0, np.nan], ids=["all zero", "all NaN"])
def test_a_background_with_nothing_to_scale_still_draws(fill):
    flow = make_flow([[[0, 0, 0], [1, 0, 0]]])

    figure = flow_line_figure(flow, AXES, np.full((1, 7, 7, 7), fill))

    assert figure.axes[0].images[0].get_clim() == (-1.0, 1.0)


def test_without_a_background_there_is_no_image():
    figure = flow_line_figure(make_flow([[[0, 0, 0], [1, 0, 0]]]), AXES)

    assert len(figure.axes[0].images) == 0


def test_real_lines_over_a_real_density_draw(static):
    metric, tensor = static
    flow = trace(tensor, seeds_per_side=4)

    figure = flow_line_figure(flow, AXES, np.asarray(tensor["tensor"][0][0]))

    assert len(figure.axes[0].lines) == 2 * len(flow.paths)
