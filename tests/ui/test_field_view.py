import numpy as np
import pytest

# The widget tests need the optional ui extra and a Qt platform plugin.
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from warpica.ui.core.slicing import grid_axes  # noqa: E402
from warpica.ui.widgets.field_view import FieldView  # noqa: E402

SHAPE = (1, 5, 5, 3)
SPACING = [1.0, 0.5, 0.5, 0.5]


@pytest.fixture
def view(qtbot):
    field_view = FieldView()
    qtbot.addWidget(field_view)
    return field_view


@pytest.fixture
def fields():
    random = np.random.default_rng(0)
    return {"a": random.normal(size=SHAPE), "b": random.normal(size=SHAPE)}


@pytest.fixture
def axes():
    return grid_axes(SHAPE, SPACING)


def test_a_new_view_is_empty(view):
    assert view.figure is None


def test_showing_fields_draws_the_first_one_in_the_middle_of_the_x_y_plane(
    view, fields, axes
):
    view.show_fields(fields, axes, "J/m³")

    assert view.figure is not None
    assert [view._field.itemText(i) for i in range(view._field.count())] == ["a", "b"]
    assert view._plane.currentText() == "x-y"
    assert (view._position.minimum(), view._position.maximum()) == (1, 3)
    assert view._position.value() == 2
    assert "a, x-y plane at z = 0 m" in view.figure.axes[0].get_title()


def test_the_colour_bar_carries_the_unit(view, fields, axes):
    view.show_fields(fields, axes, "J/m³")

    colour_bar = view.figure.axes[1]
    assert colour_bar.get_ylabel() == "J/m³"


def test_the_colour_scale_is_centred_on_zero(view, fields, axes):
    view.show_fields(fields, axes, "")

    low, high = view.figure.axes[0].images[0].get_clim()
    assert low == -high


def test_changing_the_plane_restarts_the_position_in_its_middle(view, fields, axes):
    view.show_fields(fields, axes, "")

    view._plane.setCurrentText("y-z")

    # The fixed axis is now x, which has five points.
    assert (view._position.minimum(), view._position.maximum()) == (1, 5)
    assert view._position.value() == 3
    assert view._position_label.text() == "x index"
    assert "y-z plane at x = 0 m" in view.figure.axes[0].get_title()


def test_choosing_another_field_redraws(view, fields, axes):
    view.show_fields(fields, axes, "")
    first = view.figure

    view._field.setCurrentText("b")

    assert view.figure is not first
    assert view.figure.axes[0].get_title().startswith("b,")


def test_moving_the_slice_redraws(view, fields, axes):
    view.show_fields(fields, axes, "")
    first = view.figure

    view._position.setValue(3)

    assert view.figure is not first
    assert "z = 0.5 m" in view.figure.axes[0].get_title()


def test_showing_the_same_grid_again_keeps_the_users_choices(view, fields, axes):
    view.show_fields(fields, axes, "")
    view._field.setCurrentText("b")
    view._position.setValue(3)

    view.show_fields(fields, axes, "")

    assert view._field.currentText() == "b"
    assert view._position.value() == 3


def test_a_new_grid_resets_the_slice_position(view, fields, axes):
    view.show_fields(fields, axes, "")
    view._position.setValue(3)
    bigger = (1, 7, 7, 7)

    view.show_fields({"a": np.zeros(bigger)}, grid_axes(bigger, SPACING), "")

    assert (view._position.minimum(), view._position.maximum()) == (1, 7)
    assert view._position.value() == 4


def test_the_time_step_control_appears_only_for_several_time_steps(view, axes):
    view.show_fields({"a": np.zeros(SHAPE)}, axes, "")
    assert view._time.isHidden()

    several = (3, 5, 5, 3)
    view.show_fields({"a": np.zeros(several)}, grid_axes(several, SPACING), "")

    assert not view._time.isHidden()
    assert view._time.maximum() == 3
    # A grid with several time steps starts in the middle one.
    assert view._time.value() == 2
    assert "t = 0 s" in view.figure.axes[0].get_title()


def test_the_time_step_chooses_the_slice_in_time(view):
    several = (2, 5, 5, 3)
    field = np.zeros(several)
    field[1] = 7.0
    view.show_fields({"a": field}, grid_axes(several, SPACING), "")

    view._time.setValue(2)

    low, high = view.figure.axes[0].images[0].get_clim()
    assert high == 7.0


@pytest.mark.parametrize("fill", [0.0, np.nan], ids=["all zero", "all NaN"])
def test_a_slice_with_nothing_to_scale_still_draws(view, axes, fill):
    view.show_fields({"a": np.full(SHAPE, fill)}, axes, "")

    assert view.figure is not None
    assert view.figure.axes[0].images[0].get_clim() == (-1.0, 1.0)


def test_clearing_removes_the_picture(view, fields, axes):
    view.show_fields(fields, axes, "")

    view.clear()

    assert view.figure is None
    assert view._field.count() == 0


def test_showing_no_fields_clears_the_view(view, fields, axes):
    view.show_fields(fields, axes, "")

    view.show_fields({}, axes, "")

    assert view.figure is None


def test_the_controls_are_off_while_there_is_nothing_to_show(view, fields, axes):
    assert not view._controls.isEnabled()

    view.show_fields(fields, axes, "")
    assert view._controls.isEnabled()

    view.clear()
    assert not view._controls.isEnabled()
