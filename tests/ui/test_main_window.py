import time

import numpy as np
import pytest

# The window tests need the optional ui extra and a Qt platform plugin.
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import Qt  # noqa: E402

from warpica.metrics import metric_get_minkowski  # noqa: E402
from warpica.solver import get_energy_tensor  # noqa: E402
from warpica.ui.core.jobs import BackgroundJob  # noqa: E402
from warpica.ui.core.state import Stage  # noqa: E402
from warpica.ui.widgets import run_bar as run_bar_module  # noqa: E402
from warpica.ui.windows.main_window import MainWindow  # noqa: E402


@pytest.fixture
def window(qtbot):
    main_window = MainWindow()
    qtbot.addWidget(main_window)
    return main_window


def _metric():
    """Build a small flat metric, which is all the sidebar tests need."""
    return metric_get_minkowski(np.array([1, 5, 5, 5]), np.ones(4))


def _tensor(session):
    """Solve the energy tensor of the metric stored in the session."""
    return get_energy_tensor(session.metric, "fourth")


def _is_open(window, stage):
    """Say whether the sidebar entry of a stage can be clicked."""
    item = window._sidebar.item(list(Stage).index(stage))
    return bool(item.flags() & Qt.ItemFlag.ItemIsEnabled)


def test_the_sidebar_lists_the_stages_in_order(window):
    names = [window._sidebar.item(row).text().split("\n")[0] for row in range(3)]

    assert names == ["Metric", "Solver", "Results"]


def test_at_the_start_only_the_metric_page_can_be_opened(window):
    assert window._sidebar.currentRow() == 0
    assert window._pages.currentIndex() == 0
    assert _is_open(window, Stage.METRIC)
    assert not _is_open(window, Stage.SOLVER)
    assert not _is_open(window, Stage.RESULTS)


def test_a_metric_opens_the_solver_entry_and_shows_its_status(window):
    window.session.set_metric(_metric())

    assert _is_open(window, Stage.SOLVER)
    assert "not solved" in window._sidebar.item(1).text()
    assert "built" in window._sidebar.item(0).text()


def test_selecting_an_open_entry_shows_its_page(window):
    window.session.set_metric(_metric())
    window.session.set_energy_tensor(_tensor(window.session))

    window._sidebar.setCurrentRow(2)

    assert window._pages.currentIndex() == 2


def test_rebuilding_the_metric_sends_the_user_back_from_results(window):
    window.session.set_metric(_metric())
    window.session.set_energy_tensor(_tensor(window.session))
    window._sidebar.setCurrentRow(2)

    window.session.set_metric(_metric())

    assert window._pages.currentIndex() == 0
    assert not _is_open(window, Stage.RESULTS)


def test_closing_the_window_stops_a_running_solve(window, monkeypatch):
    monkeypatch.setattr(
        run_bar_module,
        "BackgroundJob",
        lambda *args, **kwargs: BackgroundJob(time.sleep, 30),
    )
    window.session.set_metric(_metric())
    window._solver_page._run_bar.run_button.click()
    assert window._solver_page._run_bar.is_running

    window.close()

    assert not window._solver_page._run_bar.is_running
