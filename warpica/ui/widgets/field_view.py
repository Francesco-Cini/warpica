"""A viewer for fields on the (t, x, y, z) grid, one flat slice at a time."""

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from warpica._typing import Array
from warpica.ui.core.slicing import AXIS_NAMES, PLANES, axis_extent, take_slice
from warpica.ui.widgets.figure_view import FigureView


class FieldView(QWidget):
    """Show a field as a colour map on a slice the user can move.

    Above the picture are four controls: which field to show (if there are
    several), which plane to cut along, where along the remaining axis to cut,
    and, for a grid with more than one time step, which time step.

    The colour scale is centred on zero, so negative values are blue and
    positive values are red.
    """

    def __init__(self) -> None:
        super().__init__()
        self._fields: dict[str, Array] = {}
        self._axes: Sequence[Array] = ()
        self._unit = ""

        self._field = QComboBox()
        self._plane = QComboBox()
        self._plane.addItems(list(PLANES))
        self._position = QSpinBox()
        self._time = QSpinBox()
        self._view = FigureView()

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.addWidget(QLabel("Field"))
        controls.addWidget(self._field, stretch=1)
        controls.addWidget(QLabel("Plane"))
        controls.addWidget(self._plane)
        self._position_label = QLabel()
        controls.addWidget(self._position_label)
        controls.addWidget(self._position)
        self._time_label = QLabel("Time step")
        controls.addWidget(self._time_label)
        controls.addWidget(self._time)

        # With nothing to show there is nothing to choose, so the controls
        # are only enabled while a field is loaded.
        self._controls = QWidget()
        self._controls.setLayout(controls)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._controls)
        layout.addWidget(self._view, stretch=1)

        self._field.currentIndexChanged.connect(self._draw)
        self._plane.currentIndexChanged.connect(self._plane_changed)
        self._position.valueChanged.connect(self._draw)
        self._time.valueChanged.connect(self._draw)
        self.clear()

    @property
    def figure(self) -> Figure | None:
        """The figure being shown, or None if the view is empty."""
        return self._view.figure

    def show_fields(
        self, fields: Mapping[str, Array], axes: Sequence[Array], unit: str
    ) -> None:
        """Show one or more fields, all on the same grid.

        Parameters
        ----------
        fields : mapping of str to ndarray
            The fields to choose between, each with shape (nt, nx, ny, nz).
        axes : sequence of ndarray
            Coordinates of the grid points along t, x, y and z.
        unit : str
            Unit of the values, for the colour bar label.
        """
        if not fields:
            self.clear()
            return

        # Keep the field and the slice position the user had chosen, when the
        # same fields are shown again, so a refresh does not reset the view.
        chosen = self._field.currentText()
        same_grid = [len(a) for a in axes] == [len(a) for a in self._axes]

        self._fields = dict(fields)
        self._axes = axes
        self._unit = unit
        self._controls.setEnabled(True)

        self._field.blockSignals(True)
        self._field.clear()
        self._field.addItems(list(self._fields))
        if chosen in self._fields:
            self._field.setCurrentText(chosen)
        self._field.blockSignals(False)

        self._time.blockSignals(True)
        self._time.setRange(1, len(axes[0]))
        if not same_grid:
            # Start in the middle, as for the slice position: a grid with
            # several time steps has its interesting content there.
            self._time.setValue((len(axes[0]) + 1) // 2)
        self._time.blockSignals(False)
        # With a single time step there is nothing to choose.
        self._time.setVisible(len(axes[0]) > 1)
        self._time_label.setVisible(len(axes[0]) > 1)

        if not same_grid:
            self._set_position_range()
        self._draw()

    def clear(self) -> None:
        """Remove the picture and forget the fields."""
        self._fields = {}
        self._axes = ()
        self._controls.setEnabled(False)
        self._field.blockSignals(True)
        self._field.clear()
        self._field.blockSignals(False)
        self._view.clear()

    def _plane_changed(self) -> None:
        """Start the slice in the middle of the newly chosen plane."""
        if self._fields:
            self._set_position_range()
            self._draw()

    def _set_position_range(self) -> None:
        """Let the position run along the axis the plane holds fixed."""
        _, _, fixed = PLANES[self._plane.currentText()]
        points = len(self._axes[fixed])
        self._position_label.setText(f"{AXIS_NAMES[fixed]} index")
        self._position.blockSignals(True)
        self._position.setRange(1, points)
        self._position.setValue((points + 1) // 2)
        self._position.blockSignals(False)

    def _draw(self) -> None:
        """Draw the chosen field on the chosen slice."""
        name = self._field.currentText()
        plane = self._plane.currentText()
        first, second, fixed = PLANES[plane]
        position = self._position.value() - 1
        time_index = self._time.value() - 1

        values = take_slice(self._fields[name], plane, position, time_index)
        x_axis, y_axis = self._axes[first], self._axes[second]

        # Centre the colour scale on zero so the sign is visible. The limit is
        # the largest finite value, or 1 if there is none, which keeps the
        # scale valid for an all-zero or all-NaN slice.
        finite = np.abs(values[np.isfinite(values)])
        limit = float(finite.max()) if finite.size and finite.max() > 0 else 1.0

        figure = Figure()
        axes = figure.add_subplot()
        # values[i, j] is the point (x[i], y[j]). imshow expects rows to run
        # along the vertical axis, so we transpose, and origin="lower" puts
        # small values at the bottom.
        image = axes.imshow(
            values.T,
            origin="lower",
            extent=(*axis_extent(x_axis), *axis_extent(y_axis)),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axes.set_xlabel(f"{AXIS_NAMES[first]} (m)")
        axes.set_ylabel(f"{AXIS_NAMES[second]} (m)")
        title = (
            f"{name}, {plane} plane at "
            f"{AXIS_NAMES[fixed]} = {self._axes[fixed][position]:.3g} m"
        )
        if len(self._axes[0]) > 1:
            title += f", t = {self._axes[0][time_index]:.3g} s"
        axes.set_title(title, fontsize=10)
        colour_bar = figure.colorbar(image, ax=axes)
        if self._unit:
            colour_bar.set_label(self._unit)

        self._view.show_figure(figure)
