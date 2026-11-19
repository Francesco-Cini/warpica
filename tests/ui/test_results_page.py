import time

import numpy as np
import pytest

# The page tests need the optional ui extra and a Qt platform plugin.
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from warpica.metrics import metric_get_alcubierre  # noqa: E402
from warpica.solver import get_energy_tensor  # noqa: E402
from warpica.ui.core.analysis import (  # noqa: E402
    ENERGY_DENSITY,
    condition_key,
    scalar_key,
)
from warpica.ui.core.jobs import BackgroundJob  # noqa: E402
from warpica.ui.core.state import Session  # noqa: E402
from warpica.ui.pages import results_page as results_page_module  # noqa: E402
from warpica.ui.pages.results_page import (  # noqa: E402
    EnergyConditionsTab,
    EnergyDensityTab,
    FlowLinesTab,
    ResultsPage,
    ScalarsTab,
)
from warpica.ui.widgets import run_bar as run_bar_module  # noqa: E402

# Generous: a job starts a new Python process and imports NumPy in it.
JOB_TIMEOUT_MS = 60_000


def solve(velocity=0.4):
    """Return a small Alcubierre metric and its solved energy tensor."""
    grid = np.array([1, 7, 7, 7])
    scale = np.array([1.0, 0.5, 0.5, 0.5])
    metric = metric_get_alcubierre(
        grid, (grid + 1) / 2 * scale, velocity, 1.0, 1.0, scale
    )
    return metric, get_energy_tensor(metric, "fourth")


def use_a_job_that_never_ends(monkeypatch):
    """Make a tab start a 30 second sleep instead of its real calculation."""
    monkeypatch.setattr(
        run_bar_module,
        "BackgroundJob",
        lambda *args, **kwargs: BackgroundJob(time.sleep, 30),
    )


@pytest.fixture
def session():
    return Session()


@pytest.fixture
def solved_session(session):
    """A session that is ready for analysis, but whose pages are not yet built."""
    metric, tensor = solve()
    session.set_metric(metric)
    return session, tensor


def wait_until_done(tab, qtbot):
    """Wait for the calculation running in a tab to end."""
    qtbot.waitUntil(lambda: not tab._run_bar.is_running, timeout=JOB_TIMEOUT_MS)


def make_tab(cls, qtbot, session):
    tab = cls(session)
    qtbot.addWidget(tab)
    return tab


@pytest.fixture
def conditions(qtbot, session):
    tab = make_tab(EnergyConditionsTab, qtbot, session)
    yield tab
    tab.cancel()


@pytest.fixture
def scalars(qtbot, session):
    tab = make_tab(ScalarsTab, qtbot, session)
    yield tab
    tab.cancel()


@pytest.fixture
def density(qtbot, session):
    return make_tab(EnergyDensityTab, qtbot, session)


# --- Energy density -------------------------------------------------------


def test_before_a_solve_the_density_tab_is_empty(density):
    assert density._view.figure is None


def test_the_density_is_computed_and_shown_as_soon_as_a_tensor_exists(density, session):
    metric, tensor = solve()
    session.set_metric(metric)

    session.set_energy_tensor(tensor)

    assert ENERGY_DENSITY in session.results
    assert session.results[ENERGY_DENSITY].shape == (1, 7, 7, 7)
    assert density._view.figure is not None
    assert "Energy density (Eulerian)" in density._view.figure.axes[0].get_title()


def test_a_new_metric_removes_the_density_from_the_tab(density, session):
    metric, tensor = solve()
    session.set_metric(metric)
    session.set_energy_tensor(tensor)

    session.set_metric(metric)

    assert density._view.figure is None
    assert session.results == {}


def test_a_density_that_cannot_be_computed_is_reported(density, session, monkeypatch):
    def failing(tensor, metric):
        raise ValueError("boom")

    monkeypatch.setattr(results_page_module, "eulerian_energy_density", failing)
    metric, tensor = solve()
    session.set_metric(metric)

    session.set_energy_tensor(tensor)

    assert density._status.text() == "Could not compute it: boom"
    assert ENERGY_DENSITY not in session.results
    assert density._view.figure is None


# --- Energy conditions ----------------------------------------------------


def test_the_conditions_tab_cannot_run_before_a_solve(conditions):
    assert not conditions._run_bar.run_button.isEnabled()
    assert conditions._run_bar.status.text() == "Press Run to compute."


def test_a_solve_lets_the_conditions_tab_run(conditions, solved_session):
    session, tensor = solved_session

    session.set_energy_tensor(tensor)

    assert conditions._run_bar.run_button.isEnabled()


def test_running_a_condition_stores_and_shows_its_map(
    conditions, solved_session, qtbot
):
    session, tensor = solved_session
    session.set_energy_tensor(tensor)
    conditions._condition.setCurrentText("Null")
    conditions._directions.setValue(20)
    conditions._speeds.setValue(5)

    conditions._run_bar.run_button.click()
    wait_until_done(conditions, qtbot)

    result = session.results[condition_key("Null")]
    assert result.shape == (1, 7, 7, 7)
    assert conditions._run_bar.status.text().startswith("Done in ")
    assert conditions._view.figure is not None
    summary = conditions._results.text()
    assert "Null energy condition" in summary
    assert "violated in" in summary


def test_each_condition_run_adds_to_the_fields_that_can_be_shown(
    conditions, solved_session, qtbot
):
    session, tensor = solved_session
    session.set_energy_tensor(tensor)
    conditions._directions.setValue(20)
    conditions._speeds.setValue(5)

    for name in ("Null", "Weak"):
        conditions._condition.setCurrentText(name)
        conditions._run_bar.run_button.click()
        wait_until_done(conditions, qtbot)

    shown = [
        conditions._view._field.itemText(i)
        for i in range(conditions._view._field.count())
    ]
    assert shown == [condition_key("Null"), condition_key("Weak")]


def test_the_options_are_locked_while_a_condition_runs(
    conditions, solved_session, monkeypatch
):
    use_a_job_that_never_ends(monkeypatch)
    session, tensor = solved_session
    session.set_energy_tensor(tensor)

    conditions._run_bar.run_button.click()
    assert not conditions._condition.isEnabled()
    assert not conditions._directions.isEnabled()
    assert not conditions._speeds.isEnabled()

    conditions.cancel()
    assert conditions._condition.isEnabled()
    assert conditions._directions.isEnabled()
    assert conditions._speeds.isEnabled()


def test_a_replaced_tensor_cancels_a_running_condition(
    conditions, solved_session, monkeypatch
):
    use_a_job_that_never_ends(monkeypatch)
    session, tensor = solved_session
    session.set_energy_tensor(tensor)
    conditions._run_bar.run_button.click()

    session.set_energy_tensor(solve()[1])

    assert not conditions._run_bar.is_running
    assert (
        conditions._run_bar.status.text()
        == "Run cancelled because the energy tensor changed."
    )


def test_a_failing_calculation_reports_the_error(
    conditions, solved_session, monkeypatch, qtbot
):
    # int is a picklable stand-in that fails when given these arguments.
    monkeypatch.setattr(results_page_module, "energy_condition_map", int)
    session, tensor = solved_session
    session.set_energy_tensor(tensor)

    conditions._run_bar.run_button.click()
    wait_until_done(conditions, qtbot)

    assert conditions._run_bar.status.text().startswith("Run failed: TypeError")
    assert condition_key("Null") not in session.results


def test_a_new_solve_clears_the_condition_maps(conditions, solved_session, qtbot):
    session, tensor = solved_session
    session.set_energy_tensor(tensor)
    conditions._directions.setValue(20)
    conditions._speeds.setValue(5)
    conditions._run_bar.run_button.click()
    wait_until_done(conditions, qtbot)

    session.set_energy_tensor(solve()[1])

    assert conditions._view.figure is None
    assert conditions._results.text() == ""
    assert conditions._run_bar.status.text() == "Press Run to compute."


# --- Scalars --------------------------------------------------------------


def test_running_the_scalars_stores_and_shows_all_three(scalars, solved_session, qtbot):
    session, tensor = solved_session
    session.set_energy_tensor(tensor)

    scalars._run_bar.run_button.click()
    wait_until_done(scalars, qtbot)

    for name in ("Expansion", "Shear", "Vorticity"):
        assert session.results[scalar_key(name)].shape == (1, 7, 7, 7)
    shown = [
        scalars._view._field.itemText(i) for i in range(scalars._view._field.count())
    ]
    assert shown == [scalar_key(n) for n in ("Expansion", "Shear", "Vorticity")]
    assert scalars._run_bar.status.text().startswith("Done in ")


def test_the_scalars_use_the_order_the_tensor_was_solved_with(
    scalars, solved_session, monkeypatch, qtbot
):
    seen = {}

    # Look at what the tab hands to the run bar instead of running the
    # calculation.
    monkeypatch.setattr(
        scalars._run_bar,
        "start",
        lambda function, metric, order: seen.update(order=order),
    )
    session, tensor = solved_session
    tensor["order"] = "second"
    session.set_energy_tensor(tensor)

    scalars._run_bar.run_button.click()

    assert seen["order"] == "second"


# --- The page -------------------------------------------------------------


def test_the_page_has_a_tab_for_each_analysis(qtbot, session):
    page = ResultsPage(session)
    qtbot.addWidget(page)

    assert [type(tab) for tab in page._tabs] == [
        EnergyDensityTab,
        EnergyConditionsTab,
        ScalarsTab,
        FlowLinesTab,
    ]


def test_cancelling_the_page_stops_every_running_calculation(
    qtbot, session, solved_session, monkeypatch
):
    use_a_job_that_never_ends(monkeypatch)
    page = ResultsPage(session)
    qtbot.addWidget(page)
    _, tensor = solved_session
    session.set_energy_tensor(tensor)
    running = page._tabs[1:]  # every tab that has a Run button
    for tab in running:
        tab._run_bar.run_button.click()
    assert all(tab._run_bar.is_running for tab in running)

    page.cancel()

    assert not any(tab._run_bar.is_running for tab in running)


def test_energy_condition_maps_are_labelled_in_energy_density_units(
    conditions, solved_session, qtbot
):
    session, tensor = solved_session
    session.set_energy_tensor(tensor)
    conditions._directions.setValue(20)
    conditions._speeds.setValue(5)

    conditions._run_bar.run_button.click()
    wait_until_done(conditions, qtbot)

    assert conditions._view.figure.axes[1].get_ylabel() == "J/m³"


def test_the_scalars_are_not_labelled_with_an_energy_unit(
    scalars, solved_session, qtbot
):
    session, tensor = solved_session
    session.set_energy_tensor(tensor)

    scalars._run_bar.run_button.click()
    wait_until_done(scalars, qtbot)

    assert scalars._view.figure.axes[1].get_ylabel() == ""
