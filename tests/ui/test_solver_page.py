import time

import numpy as np
import pytest

# The page tests need the optional ui extra and a Qt platform plugin.
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from warpica.metrics import metric_get_alcubierre  # noqa: E402
from warpica.ui.core.jobs import BackgroundJob  # noqa: E402
from warpica.ui.core.state import Session, Stage  # noqa: E402
from warpica.ui.pages import solver_page as solver_page_module  # noqa: E402
from warpica.ui.pages.solver_page import SolverPage  # noqa: E402
from warpica.ui.widgets import run_bar as run_bar_module  # noqa: E402

# Generous: a solve starts a new Python process and imports NumPy in it.
SOLVE_TIMEOUT_MS = 60_000


def make_metric(size=(1, 5, 5, 5)):
    """Build a small Alcubierre metric on a grid of the given size."""
    grid = np.array(size)
    scale = np.array([1.0, 0.5, 0.5, 0.5])
    return metric_get_alcubierre(grid, (grid + 1) / 2 * scale, 0.4, 1.0, 1.0, scale)


def use_a_job_that_never_ends(monkeypatch):
    """Make the page start a 30 second sleep instead of a real solve."""
    monkeypatch.setattr(
        run_bar_module,
        "BackgroundJob",
        lambda *args, **kwargs: BackgroundJob(time.sleep, 30),
    )


def wait_until_done(page, qtbot):
    """Wait for the running solve to end."""
    qtbot.waitUntil(lambda: not page._run_bar.is_running, timeout=SOLVE_TIMEOUT_MS)


@pytest.fixture
def session():
    return Session()


@pytest.fixture
def page(qtbot, session):
    solver_page = SolverPage(session)
    qtbot.addWidget(solver_page)
    yield solver_page
    # Never leave a background process running after a test.
    solver_page.cancel()


def test_without_a_metric_there_is_nothing_to_solve(page):
    bar = page._run_bar

    assert not bar.run_button.isEnabled()
    assert not bar.cancel_button.isEnabled()
    assert bar.progress.isHidden()
    assert "No metric yet" in page._summary.text()


def test_a_built_metric_is_summarised_and_can_be_solved(page, session):
    session.set_metric(make_metric())

    assert page._run_bar.run_button.isEnabled()
    assert page._summary.text() == "Metric: Alcubierre, grid 1 x 5 x 5 x 5"
    assert page._run_bar.status.text() == "Press Solve to compute the energy tensor."


def test_fourth_order_and_the_cpu_are_the_defaults(page):
    assert page._order.currentData() == "fourth"
    assert page._backend.currentData() is None
    assert page._backend.currentText() == "CPU (NumPy)"


def test_solving_stores_the_tensor_opens_results_and_draws_it(page, session, qtbot):
    session.set_metric(make_metric())
    bar = page._run_bar

    bar.run_button.click()

    # While it runs the user can cancel, and the settings are locked.
    assert bar.cancel_button.isEnabled()
    assert not bar.run_button.isEnabled()
    assert not page._order.isEnabled()
    assert not page._backend.isEnabled()
    assert not bar.progress.isHidden()

    wait_until_done(page, qtbot)

    assert session.energy_tensor["type"] == "Stress-Energy"
    assert session.energy_tensor["order"] == "fourth"
    assert session.is_unlocked(Stage.RESULTS)
    assert bar.status.text().startswith("Solved (fourth order, CPU (NumPy)) in ")
    assert page._view.figure is not None
    # The page is ready for another run.
    assert bar.run_button.isEnabled()
    assert not bar.cancel_button.isEnabled()
    assert bar.progress.isHidden()
    assert page._order.isEnabled()
    assert page._backend.isEnabled()


def test_the_chosen_derivative_order_is_used(page, session, qtbot):
    session.set_metric(make_metric())
    page._order.setCurrentIndex(page._order.findData("second"))

    page._run_bar.run_button.click()
    wait_until_done(page, qtbot)

    assert session.energy_tensor["order"] == "second"
    assert "second order" in page._run_bar.status.text()


def test_a_failed_solve_reports_the_error_and_stores_nothing(page, session, qtbot):
    # Fourth order needs five points on every axis, and this metric has three.
    session.set_metric(make_metric(size=(1, 3, 3, 3)))

    page._run_bar.run_button.click()
    wait_until_done(page, qtbot)

    assert page._run_bar.status.text().startswith("Solve failed: ValueError")
    assert session.energy_tensor is None
    assert not session.is_unlocked(Stage.RESULTS)
    assert page._view.figure is None
    assert page._run_bar.run_button.isEnabled()


def test_cancel_stops_the_solve_and_changes_nothing(page, session, monkeypatch):
    use_a_job_that_never_ends(monkeypatch)
    session.set_metric(make_metric())
    page._run_bar.run_button.click()

    page._run_bar.cancel_button.click()

    assert not page._run_bar.is_running
    assert page._run_bar.status.text() == "Solve cancelled."
    assert session.energy_tensor is None
    assert page._run_bar.run_button.isEnabled()
    assert page._order.isEnabled()


def test_rebuilding_the_metric_cancels_a_solve_in_progress(page, session, monkeypatch):
    use_a_job_that_never_ends(monkeypatch)
    session.set_metric(make_metric())
    page._run_bar.run_button.click()

    session.set_metric(make_metric())

    assert not page._run_bar.is_running
    assert page._run_bar.status.text() == "Solve cancelled because the metric changed."
    assert session.energy_tensor is None


def test_a_new_metric_clears_the_old_solution_from_the_page(page, session, qtbot):
    session.set_metric(make_metric())
    page._run_bar.run_button.click()
    wait_until_done(page, qtbot)
    assert page._view.figure is not None

    session.set_metric(make_metric())

    assert page._view.figure is None
    assert page._run_bar.status.text() == "Press Solve to compute the energy tensor."


def test_a_failing_preview_keeps_the_solved_tensor_and_says_so(
    page, session, monkeypatch, qtbot
):
    def failing_plot(tensor):
        raise ValueError("not finite")

    monkeypatch.setattr(solver_page_module, "plot_tensor", failing_plot)
    session.set_metric(make_metric())

    page._run_bar.run_button.click()
    wait_until_done(page, qtbot)

    assert session.energy_tensor is not None
    assert "The preview failed: not finite" in page._run_bar.status.text()
    assert page._view.figure is None
