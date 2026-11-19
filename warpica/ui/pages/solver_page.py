"""The Solver page: compute the stress-energy tensor of the built metric."""

from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from warpica._typing import TensorDict
from warpica.ui.core.solving import DIFF_ORDERS, available_backends, solve_energy_tensor
from warpica.ui.core.state import Session
from warpica.ui.widgets.figure_view import FigureView
from warpica.ui.widgets.run_bar import RunBar
from warpica.visualiser import plot_tensor


class SolverPage(QWidget):
    """Choose how to solve, run the solver and look at the energy tensor.

    The solver can take a long time, so it runs in a separate process. The
    window stays usable while it works, and Cancel really stops it. Solving
    again replaces the stored tensor and discards the results computed from the
    old one.

    Parameters
    ----------
    session : Session
        The state shared with the other pages.
    """

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session

        # What is being solved right now, so a result for a metric that has
        # since been replaced can be thrown away.
        self._job_metric: object = None
        self._job_description = ""

        # Left side: the settings and the run controls.
        self._summary = QLabel()
        self._summary.setWordWrap(True)
        self._order = QComboBox()
        for label, order in DIFF_ORDERS.items():
            self._order.addItem(label, order)
        order_note = QLabel(
            "Fourth order needs at least 5 points along every axis that has more "
            "than one point. Second order needs 3."
        )
        order_note.setWordWrap(True)
        self._backend = QComboBox()
        for label, backend in available_backends().items():
            self._backend.addItem(label, backend)
        backend_note = QLabel(
            "GPU backends are experimental and appear here when CuPy or "
            "torch-directml is installed."
        )
        backend_note.setWordWrap(True)
        self._run_bar = RunBar("Solve", "Solving")

        panel = QWidget()
        panel.setFixedWidth(330)
        panel_layout = QVBoxLayout(panel)
        panel_layout.addWidget(QLabel("<h2>Solver</h2>"))
        panel_layout.addWidget(self._summary)
        panel_layout.addWidget(QLabel("<b>Derivative order</b>"))
        panel_layout.addWidget(self._order)
        panel_layout.addWidget(order_note)
        panel_layout.addWidget(QLabel("<b>Backend</b>"))
        panel_layout.addWidget(self._backend)
        panel_layout.addWidget(backend_note)
        panel_layout.addWidget(self._run_bar)
        panel_layout.addStretch()

        # Right side: the energy tensor.
        self._view = FigureView()

        layout = QHBoxLayout(self)
        layout.addWidget(panel)
        layout.addWidget(self._view, stretch=1)

        self._run_bar.run_requested.connect(self._solve)
        self._run_bar.finished.connect(self._solved)
        # The settings are read when Solve is pressed, so lock them while a
        # solve is running to keep what the status says true.
        self._run_bar.running_changed.connect(self._lock_settings)
        self._session.subscribe(self._on_session_changed)
        self._refresh()
        self._run_bar.set_status("Press Solve to compute the energy tensor.")

    def cancel(self) -> None:
        """Stop the running solve, if there is one."""
        self._run_bar.cancel()

    def _on_session_changed(self) -> None:
        """React to a change in the shared state."""
        metric = self._session.metric
        message = None

        if self._run_bar.is_running and metric is not self._job_metric:
            # The metric was rebuilt while solving, so the answer would belong
            # to a metric that is no longer there.
            self._run_bar.cancel()
            message = "Solve cancelled because the metric changed."
        self._refresh()

        if self._session.energy_tensor is None and not self._run_bar.is_running:
            # Nothing is solved for this metric, so there is nothing to show
            # and any earlier message about a solve is out of date.
            self._view.clear()
            message = message or "Press Solve to compute the energy tensor."

        if message is not None:
            self._run_bar.set_status(message)

    def _refresh(self) -> None:
        """Update the summary line and whether Solve can be pressed."""
        metric = self._session.metric
        if metric is None:
            self._summary.setText("No metric yet. Build one on the Metric page.")
        else:
            shape = metric["tensor"][0][0].shape
            size = " x ".join(str(n) for n in shape)
            self._summary.setText(f"Metric: {metric['name']}, grid {size}")
        self._run_bar.set_ready(metric is not None)

    def _lock_settings(self, running: bool) -> None:
        """Enable the settings only while no solve is running."""
        self._order.setEnabled(not running)
        self._backend.setEnabled(not running)

    def _solve(self) -> None:
        """Start solving the current metric in a separate process."""
        metric = self._session.metric
        order = self._order.currentData()
        backend = self._backend.currentData()
        self._job_metric = metric
        self._job_description = f"{order} order, {self._backend.currentText()}"
        self._run_bar.start(solve_energy_tensor, metric, order, backend)

    def _solved(self, energy_tensor: TensorDict, seconds: float) -> None:
        """Store a finished solve and draw it.

        Parameters
        ----------
        energy_tensor : dict
            The tensor the solver returned.
        seconds : float
            How long the solve took.
        """
        # Storing the tensor opens the Results stage and discards any results
        # computed from an earlier tensor.
        self._session.set_energy_tensor(energy_tensor)
        self._run_bar.set_status(
            f"Solved ({self._job_description}) in {seconds:.1f} s."
        )
        self._show_tensor(energy_tensor)

    def _show_tensor(self, energy_tensor: TensorDict) -> None:
        """Draw a solved energy tensor.

        Parameters
        ----------
        energy_tensor : dict
            The tensor to draw.
        """
        try:
            figure, _ = plot_tensor(energy_tensor)
        except ValueError as error:
            # The tensor is solved and stored; only the picture failed.
            self._view.clear()
            self._run_bar.set_status(
                f"{self._run_bar.status.text()} The preview failed: {error}"
            )
            return
        self._view.show_figure(figure)
