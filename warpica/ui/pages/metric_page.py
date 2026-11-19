"""The Metric page: choose a metric, set its parameters and the grid."""

from collections.abc import Callable

from matplotlib.figure import Figure
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from warpica._typing import Array, TensorDict
from warpica.ui.core.registry import METRICS
from warpica.ui.core.state import Session
from warpica.ui.widgets.figure_view import FigureView
from warpica.ui.widgets.grid_form import GridForm
from warpica.ui.widgets.parameter_form import ParameterForm
from warpica.visualiser import plot_tensor, plot_three_plus_one

# The two ways to look at a built metric. Both need no solving: they draw the
# metric itself, on a slice through the middle of the grid.
_PREVIEWS: dict[str, Callable[[TensorDict], tuple[Figure, Array]]] = {
    "Metric components": plot_tensor,
    "3+1 fields (lapse, shift, space)": plot_three_plus_one,
}


class MetricPage(QWidget):
    """Configure a metric, build it and preview it.

    Pressing Build stores the metric in the session, which opens the Solver
    stage and discards anything computed from an earlier metric.

    Parameters
    ----------
    session : Session
        The state shared with the other pages.
    """

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session

        # Left side: everything that configures the metric.
        self._chooser = QComboBox()
        for key, spec in METRICS.items():
            # The key is stored with the entry so we can find the metric again.
            self._chooser.addItem(spec.label, key)
        self._parameters = ParameterForm()
        self._grid = GridForm()
        self._preview_kind = QComboBox()
        self._preview_kind.addItems(list(_PREVIEWS))
        self._build_button = QPushButton("Build metric")
        self._status = QLabel("Choose a metric and press Build.")
        self._status.setWordWrap(True)

        panel = QWidget()
        panel.setFixedWidth(330)
        panel_layout = QVBoxLayout(panel)
        panel_layout.addWidget(QLabel("<h2>Metric</h2>"))
        panel_layout.addWidget(self._chooser)
        panel_layout.addWidget(QLabel("<b>Parameters</b>"))
        panel_layout.addWidget(self._parameters)
        panel_layout.addWidget(QLabel("<b>Grid</b>"))
        panel_layout.addWidget(self._grid)
        panel_layout.addWidget(QLabel("<b>Preview</b>"))
        panel_layout.addWidget(self._preview_kind)
        panel_layout.addWidget(self._build_button)
        panel_layout.addWidget(self._status)
        panel_layout.addStretch()

        # Right side: the preview.
        self._view = FigureView()

        layout = QHBoxLayout(self)
        layout.addWidget(panel)
        layout.addWidget(self._view, stretch=1)

        self._chooser.currentIndexChanged.connect(self._show_parameters)
        self._preview_kind.currentIndexChanged.connect(self._show_preview)
        self._build_button.clicked.connect(self._build)
        self._show_parameters()

    def _show_parameters(self) -> None:
        """Show the input fields of the metric chosen in the list."""
        spec = METRICS[self._chooser.currentData()]
        self._parameters.set_parameters(spec.parameters)

    def _build(self) -> None:
        """Build the metric from the form, store it and draw the preview."""
        spec = METRICS[self._chooser.currentData()]
        grid_size = self._grid.grid_size()
        grid_scale = self._grid.grid_scale()

        # Grid coordinates are (index + 1) * spacing - centre, so this centre
        # puts the middle of the grid at the origin.
        world_centre = (grid_size + 1) / 2 * grid_scale

        try:
            metric = spec.build(
                grid_size, world_centre, grid_scale, **self._parameters.values()
            )
        except (ValueError, ArithmeticError) as error:
            self._status.setText(f"Could not build the metric: {error}")
            return

        self._session.set_metric(metric)
        size = " x ".join(str(n) for n in grid_size)
        self._status.setText(f"Built {spec.label} on a {size} grid.")
        self._show_preview()

    def _show_preview(self) -> None:
        """Draw the built metric in the way chosen in the preview list."""
        metric = self._session.metric
        if metric is None:
            return
        plot = _PREVIEWS[self._preview_kind.currentText()]
        try:
            figure, _ = plot(metric)
        except ValueError as error:
            # The metric is built and stored; only the picture failed.
            self._view.clear()
            self._status.setText(
                f"The metric is built, but the preview failed: {error}"
            )
            return
        self._view.show_figure(figure)
