import dataclasses

import pytest

# The page tests need the optional ui extra and a Qt platform plugin.
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from warpica.ui.core.registry import METRICS  # noqa: E402
from warpica.ui.core.state import Session, Stage  # noqa: E402
from warpica.ui.pages import metric_page as metric_page_module  # noqa: E402
from warpica.ui.pages.metric_page import MetricPage  # noqa: E402


@pytest.fixture
def session():
    return Session()


@pytest.fixture
def page(qtbot, session):
    metric_page = MetricPage(session)
    qtbot.addWidget(metric_page)
    # A small grid keeps every build fast.
    metric_page._grid._size[1].setValue(5)
    metric_page._grid._size[2].setValue(5)
    metric_page._grid._size[3].setValue(5)
    return metric_page


def test_the_chooser_lists_every_metric(page):
    labels = [page._chooser.itemText(i) for i in range(page._chooser.count())]

    assert labels == [spec.label for spec in METRICS.values()]


def test_the_form_shows_the_parameters_of_the_chosen_metric(page):
    assert set(page._parameters.values()) == {"v", "R", "sigma"}

    page._chooser.setCurrentIndex(page._chooser.findData("minkowski"))

    assert page._parameters.values() == {}


def test_building_stores_the_metric_and_unlocks_the_solver(page, session):
    assert not session.is_unlocked(Stage.SOLVER)

    page._build_button.click()

    assert session.metric["name"] == "Alcubierre"
    assert session.is_unlocked(Stage.SOLVER)
    assert "Built Alcubierre on a 1 x 5 x 5 x 5 grid." in page._status.text()


def test_the_form_values_reach_the_metric(page, session):
    page._parameters._fields["v"].setValue(0.7)

    page._build_button.click()

    assert session.metric["params"]["velocity"] == 0.7
    assert list(session.metric["params"]["grid_size"]) == [1, 5, 5, 5]


def test_the_bubble_is_centred_in_the_grid(page, session):
    page._build_button.click()

    # Grid coordinates are (index + 1) * spacing - centre, so the middle
    # point of an odd-sized grid sits at the origin.
    centre = session.metric["params"]["world_centre"]
    spacing = session.metric["scaling"]
    middle = (5 // 2 + 1) * spacing[1] - centre[1]
    assert middle == 0


def test_building_draws_a_preview(page):
    assert page._view.figure is None

    page._build_button.click()

    assert page._view.figure is not None


def test_the_preview_kind_can_be_switched(page):
    page._build_button.click()
    first = page._view.figure

    page._preview_kind.setCurrentIndex(1)

    assert page._view.figure is not first


def test_flat_space_can_be_built_and_previewed(page, session):
    page._chooser.setCurrentIndex(page._chooser.findData("minkowski"))

    page._build_button.click()

    assert session.metric["name"] == "Minkowski"
    assert page._view.figure is not None


def test_a_failing_build_reports_the_error_and_keeps_the_old_metric(
    page, session, monkeypatch
):
    page._build_button.click()
    old_metric = session.metric

    def failing_build(*args, **kwargs):
        raise ValueError("grid too small")

    # MetricSpec is frozen, so replace the whole entry instead of editing it.
    broken = dataclasses.replace(METRICS["alcubierre"], build=failing_build)
    monkeypatch.setitem(METRICS, "alcubierre", broken)
    page._build_button.click()

    assert page._status.text() == "Could not build the metric: grid too small"
    assert session.metric is old_metric


def test_a_failing_preview_keeps_the_metric_and_says_so(page, session, monkeypatch):
    def failing_plot(metric):
        raise ValueError("not finite")

    monkeypatch.setitem(metric_page_module._PREVIEWS, "Metric components", failing_plot)
    page._build_button.click()

    assert session.metric is not None
    assert "preview failed: not finite" in page._status.text()
    assert page._view.figure is None
