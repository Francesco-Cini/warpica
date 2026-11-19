import time

import numpy as np
import pytest

# The tab tests need the optional ui extra and a Qt platform plugin.
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from warpica.metrics import metric_get_alcubierre, metric_get_minkowski  # noqa: E402
from warpica.solver import get_energy_tensor  # noqa: E402
from warpica.ui.core.analysis import ENERGY_DENSITY, FLOW_LINES  # noqa: E402
from warpica.ui.core.jobs import BackgroundJob  # noqa: E402
from warpica.ui.core.state import Session  # noqa: E402
from warpica.ui.pages.results_page import EnergyDensityTab, FlowLinesTab  # noqa: E402
from warpica.ui.widgets import run_bar as run_bar_module  # noqa: E402

# Generous: a job starts a new Python process and imports NumPy in it.
JOB_TIMEOUT_MS = 60_000
SPACING = np.array([1.0, 0.5, 0.5, 0.5])


def solve(size=(1, 7, 7, 7)):
    """Return a small Alcubierre metric and its solved energy tensor."""
    grid = np.array(size)
    metric = metric_get_alcubierre(
        grid, (grid + 1) / 2 * SPACING, 0.4, 1.0, 1.0, SPACING
    )
    return metric, get_energy_tensor(metric, "fourth")


def use_a_job_that_never_ends(monkeypatch):
    """Make the tab start a 30 second sleep instead of tracing."""
    monkeypatch.setattr(
        run_bar_module,
        "BackgroundJob",
        lambda *args, **kwargs: BackgroundJob(time.sleep, 30),
    )


@pytest.fixture
def session():
    return Session()


@pytest.fixture
def tab(qtbot, session):
    flow_tab = FlowLinesTab(session)
    qtbot.addWidget(flow_tab)
    yield flow_tab
    flow_tab.cancel()


def solved(session, size=(1, 7, 7, 7)):
    """Put a solved metric and tensor in the session."""
    metric, tensor = solve(size)
    session.set_metric(metric)
    session.set_energy_tensor(tensor)


def wait_until_done(tab, qtbot):
    qtbot.waitUntil(lambda: not tab._run_bar.is_running, timeout=JOB_TIMEOUT_MS)


def small_run(tab):
    """Use few seeds and steps so a real trace is quick."""
    tab._seeds.setValue(2)
    tab._max_steps.setValue(30)


# --- Options ---------------------------------------------------------------


def test_the_tab_cannot_run_before_a_solve(tab):
    assert not tab._run_bar.run_button.isEnabled()
    assert tab._run_bar.status.text() == "Press Run to compute."


def test_the_defaults_are_a_five_by_five_lattice_of_quarter_cell_steps(tab):
    assert tab._plane.currentText() == "x-y"
    assert tab._seeds.value() == 5
    assert tab._step.value() == 0.25
    assert tab._max_steps.value() == 400


def test_the_plane_can_sit_anywhere_inside_the_grid_and_starts_in_the_middle(
    tab, session
):
    solved(session)

    # Seven points along z: the plane may use points 2 to 6, starting at 4.
    assert (tab._position.minimum(), tab._position.maximum()) == (2, 6)
    assert tab._position.value() == 4
    assert "z index" in tab._position_label.text()


def test_changing_the_plane_refits_the_position_to_its_fixed_axis(tab, session):
    solved(session, size=(1, 5, 7, 9))

    tab._plane.setCurrentText("y-z")

    # The fixed axis is now x, which has five points.
    assert (tab._position.minimum(), tab._position.maximum()) == (2, 4)
    assert tab._position.value() == 3
    assert "x index" in tab._position_label.text()


def test_a_position_the_user_chose_is_kept_while_it_still_fits(tab, session):
    solved(session)
    tab._position.setValue(5)

    solved(session)  # a new solve on the same grid

    assert tab._position.value() == 5


def test_there_is_no_time_step_choice_for_a_single_time_step(tab, session):
    solved(session)

    assert tab._time.isHidden()


def test_several_time_steps_can_be_chosen_and_start_in_the_middle(tab, session):
    solved(session, size=(5, 7, 7, 7))

    assert not tab._time.isHidden()
    assert (tab._time.minimum(), tab._time.maximum()) == (1, 5)
    assert tab._time.value() == 3


# --- Running ---------------------------------------------------------------


def test_tracing_stores_and_draws_the_lines(tab, session, qtbot):
    solved(session)
    small_run(tab)

    tab._run_bar.run_button.click()
    wait_until_done(tab, qtbot)

    flow = session.results[FLOW_LINES]
    assert len(flow.paths) == 4
    assert flow.plane == "x-y"
    assert flow.position == 3  # the middle, as a zero-based index
    assert tab._run_bar.status.text().startswith("Done in ")
    assert tab._view.figure is not None
    assert "Momentum flow lines" in tab._view.figure.axes[0].get_title()


def test_the_lines_are_drawn_over_the_energy_density(tab, session, qtbot):
    # The density tab is what computes the density when the tensor is solved.
    density_tab = EnergyDensityTab(session)
    qtbot.addWidget(density_tab)
    solved(session)
    assert ENERGY_DENSITY in session.results
    small_run(tab)

    tab._run_bar.run_button.click()
    wait_until_done(tab, qtbot)

    assert len(tab._view.figure.axes[0].images) == 1


def test_the_chosen_options_reach_the_trace(tab, session, qtbot):
    solved(session)
    tab._plane.setCurrentText("x-z")
    tab._position.setValue(5)
    tab._seeds.setValue(3)
    tab._max_steps.setValue(10)

    tab._run_bar.run_button.click()
    wait_until_done(tab, qtbot)

    flow = session.results[FLOW_LINES]
    assert flow.plane == "x-z"
    assert flow.position == 4  # one-based 5, zero-based 4
    assert len(flow.paths) == 9
    assert all(len(path) <= 11 for path in flow.paths)


def test_a_time_dependent_grid_is_traced_at_the_chosen_time_step(tab, session, qtbot):
    solved(session, size=(5, 7, 7, 7))
    small_run(tab)

    tab._run_bar.run_button.click()  # the middle step, which holds the bubble
    wait_until_done(tab, qtbot)

    assert session.results[FLOW_LINES].time_index == 2


def test_a_flat_time_step_reports_that_there_is_no_flow(tab, session, qtbot):
    solved(session, size=(5, 7, 7, 7))
    small_run(tab)
    tab._time.setValue(1)  # the first step is flat

    tab._run_bar.run_button.click()
    wait_until_done(tab, qtbot)

    assert "zero everywhere" in tab._run_bar.status.text()
    assert FLOW_LINES not in session.results
    assert tab._view.figure is None


def test_flat_space_reports_that_there_is_no_flow(tab, session, qtbot):
    grid = np.array([1, 7, 7, 7])
    metric = metric_get_minkowski(grid, SPACING)
    session.set_metric(metric)
    session.set_energy_tensor(get_energy_tensor(metric, "fourth"))

    tab._run_bar.run_button.click()
    wait_until_done(tab, qtbot)

    assert tab._run_bar.status.text().startswith("Run failed: ValueError")
    assert "zero everywhere" in tab._run_bar.status.text()


def test_the_options_are_locked_while_tracing(tab, session, monkeypatch):
    use_a_job_that_never_ends(monkeypatch)
    solved(session)
    options = (
        tab._plane,
        tab._position,
        tab._time,
        tab._seeds,
        tab._step,
        tab._max_steps,
    )

    tab._run_bar.run_button.click()
    assert not any(option.isEnabled() for option in options)

    tab.cancel()
    assert all(option.isEnabled() for option in options)


def test_a_replaced_tensor_cancels_a_trace_in_progress(tab, session, monkeypatch):
    use_a_job_that_never_ends(monkeypatch)
    solved(session)
    tab._run_bar.run_button.click()

    session.set_energy_tensor(solve()[1])

    assert not tab._run_bar.is_running
    assert (
        tab._run_bar.status.text() == "Run cancelled because the energy tensor changed."
    )


def test_a_new_solve_removes_the_lines(tab, session, qtbot):
    solved(session)
    small_run(tab)
    tab._run_bar.run_button.click()
    wait_until_done(tab, qtbot)
    assert tab._view.figure is not None

    session.set_energy_tensor(solve()[1])

    assert tab._view.figure is None
    assert FLOW_LINES not in session.results
    assert tab._run_bar.status.text() == "Press Run to compute."
