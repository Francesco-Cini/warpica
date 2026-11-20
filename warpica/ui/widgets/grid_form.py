"""Fields for the size and spacing of the grid the metric is sampled on."""

import numpy as np
from PySide6.QtWidgets import QDoubleSpinBox, QGridLayout, QLabel, QSpinBox, QWidget

from warpica._typing import Array

_AXES = ("t", "x", "y", "z")
_DEFAULT_SIZE = (1, 17, 17, 9)
# Spacing between points: seconds for t, metres for x, y and z.
_DEFAULT_SPACING = (1.0, 0.5, 0.5, 0.5)

# Fewest and most points allowed along each axis. The metric builders are plain
# Python loops that run on the window's thread for now, so the upper limits
# keep one build to about a second. Space needs a few points to draw a slice.
_MIN_SIZE = (1, 3, 3, 3)
_MAX_SIZE = (16, 41, 41, 41)
_SPACING_RANGE = ((0.001, 1000.0), (0.01, 10.0), (0.01, 10.0), (0.01, 10.0))

# Narrowest a spin box may get, in pixels, so its value and arrows stay visible.
_MIN_FIELD_WIDTH = 90


class GridForm(QWidget):
    """Choose how many points to use along t, x, y, z and how far apart they are.

    The bubble is always placed at the middle of the grid, so the form has no
    field for its position.
    """

    def __init__(self) -> None:
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._size: list[QSpinBox] = []
        self._spacing: list[QDoubleSpinBox] = []
        # One row per axis and one column per quantity. Four spin boxes side by
        # side were too narrow to show their values in the side panel.
        layout.addWidget(QLabel("Points"), 0, 1)
        layout.addWidget(QLabel("Spacing"), 0, 2)
        for row, axis in enumerate(_AXES, start=1):
            layout.addWidget(QLabel(axis), row, 0)

            points = QSpinBox()
            points.setRange(_MIN_SIZE[row - 1], _MAX_SIZE[row - 1])
            points.setValue(_DEFAULT_SIZE[row - 1])
            points.setMinimumWidth(_MIN_FIELD_WIDTH)
            layout.addWidget(points, row, 1)
            self._size.append(points)

            spacing = QDoubleSpinBox()
            spacing.setDecimals(3)
            spacing.setRange(*_SPACING_RANGE[row - 1])
            spacing.setValue(_DEFAULT_SPACING[row - 1])
            spacing.setMinimumWidth(_MIN_FIELD_WIDTH)
            layout.addWidget(spacing, row, 2)
            self._spacing.append(spacing)

    def grid_size(self) -> Array:
        """Return the number of points along (t, x, y, z) as an integer array."""
        return np.array([field.value() for field in self._size])

    def grid_scale(self) -> Array:
        """Return the spacing along (t, x, y, z): seconds, then metres."""
        return np.array([field.value() for field in self._spacing])
