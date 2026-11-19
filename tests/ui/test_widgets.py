import pytest

# The widget tests need the optional ui extra and a Qt platform plugin.
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from warpica.ui.core.registry import METRICS  # noqa: E402
from warpica.ui.widgets.figure_view import FigureView  # noqa: E402
from warpica.ui.widgets.grid_form import GridForm  # noqa: E402
from warpica.ui.widgets.parameter_form import ParameterForm  # noqa: E402


def test_parameter_form_starts_at_the_defaults(qtbot):
    form = ParameterForm()
    qtbot.addWidget(form)
    spec = METRICS["alcubierre"]

    form.set_parameters(spec.parameters)

    assert form.values() == pytest.approx(spec.defaults())


def test_parameter_form_returns_what_the_user_typed(qtbot):
    form = ParameterForm()
    qtbot.addWidget(form)
    form.set_parameters(METRICS["alcubierre"].parameters)

    form._fields["v"].setValue(1.5)

    assert form.values()["v"] == 1.5


def test_parameter_form_replaces_its_fields_for_a_new_metric(qtbot):
    form = ParameterForm()
    qtbot.addWidget(form)
    form.set_parameters(METRICS["alcubierre"].parameters)

    form.set_parameters(METRICS["minkowski"].parameters)

    assert form.values() == {}


def test_parameter_form_enforces_the_range_of_each_parameter(qtbot):
    form = ParameterForm()
    qtbot.addWidget(form)
    form.set_parameters(METRICS["alcubierre"].parameters)

    form._fields["R"].setValue(100.0)

    assert form.values()["R"] == 3.0  # the maximum for R


def test_grid_form_starts_at_the_default_grid(qtbot):
    form = GridForm()
    qtbot.addWidget(form)

    np.testing.assert_array_equal(form.grid_size(), [1, 17, 17, 9])
    np.testing.assert_allclose(form.grid_scale(), [1, 0.5, 0.5, 0.5])


def test_grid_form_returns_what_the_user_typed(qtbot):
    form = GridForm()
    qtbot.addWidget(form)

    form._size[1].setValue(11)
    form._spacing[1].setValue(0.25)

    assert form.grid_size()[1] == 11
    assert form.grid_scale()[1] == 0.25


def test_figure_view_shows_and_clears_a_figure(qtbot):
    view = FigureView()
    qtbot.addWidget(view)
    figure = Figure()

    view.show_figure(figure)
    assert view.figure is figure

    view.clear()
    assert view.figure is None


def test_figure_view_replaces_the_previous_figure(qtbot):
    view = FigureView()
    qtbot.addWidget(view)
    second = Figure()

    view.show_figure(Figure())
    view.show_figure(second)

    assert view.figure is second
    assert len(view._widgets) == 2  # one toolbar and one scroll area


def test_figure_view_does_not_squeeze_a_figure_below_a_readable_size(qtbot):
    view = FigureView()
    qtbot.addWidget(view)
    figure = Figure(figsize=(10, 8))  # 1000 x 800 pixels at 100 dpi

    view.show_figure(figure)

    canvas = view._widgets[1].widget()
    assert canvas.minimumWidth() == 700
    assert canvas.minimumHeight() == 560
