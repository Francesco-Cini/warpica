"""The Results page: analyse the solved stress-energy tensor.

The page has one tab for each kind of analysis. Every tab reads what it shows
from ``session.results`` and writes what it computes back to it. That keeps one
copy of each result, and means a result disappears the moment the metric or
the tensor it came from is replaced.
"""

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from warpica._typing import Array
from warpica.ui.core.analysis import (
    CONDITIONS,
    ENERGY_DENSITY,
    FLOW_LINES,
    SCALARS,
    condition_key,
    energy_condition_map,
    eulerian_energy_density,
    momentum_flow_lines,
    scalar_fields,
    scalar_key,
    summarise_condition,
)
from warpica.ui.core.flow_plot import flow_line_figure
from warpica.ui.core.slicing import AXIS_NAMES, PLANES, grid_axes
from warpica.ui.core.state import Session
from warpica.ui.widgets.field_view import FieldView
from warpica.ui.widgets.figure_view import FigureView
from warpica.ui.widgets.run_bar import RunBar

# Unit of the energy density and of the energy condition maps.
_ENERGY_UNIT = "J/m³"


class _AnalysisTab(QWidget):
    """The layout shared by every tab: options on the left, a viewer on the right.

    A subclass adds its options with ``self._panel_layout`` and its viewer with
    ``_add_view``.

    Parameters
    ----------
    session : Session
        The state shared with the other pages.
    title : str
        Heading shown above the options.
    """

    def __init__(self, session: Session, title: str) -> None:
        super().__init__()
        self._session = session

        self._panel_layout = QVBoxLayout()
        self._panel_layout.addWidget(QLabel(f"<h3>{title}</h3>"))
        panel = QWidget()
        panel.setMinimumWidth(280)
        panel.setLayout(self._panel_layout)

        # A splitter lets the user drag the edge between options and viewer.
        self._splitter = QSplitter()
        self._splitter.addWidget(panel)
        self._splitter.setChildrenCollapsible(False)
        self._layout = QHBoxLayout(self)
        self._layout.addWidget(self._splitter)

        session.subscribe(self._on_session_changed)

    def cancel(self) -> None:
        """Stop any calculation running in this tab."""

    def _add_view(self, view: QWidget) -> None:
        """Put the viewer to the right of the options."""
        self._splitter.addWidget(view)
        self._splitter.setStretchFactor(1, 1)

    def _add_note(self, text: str) -> None:
        """Add a line of explanation to the options."""
        note = QLabel(text)
        note.setWordWrap(True)
        self._panel_layout.addWidget(note)

    def _finish_panel(self) -> None:
        """Push the options to the top. Call once, after adding them."""
        self._panel_layout.addStretch()

    def _grid_axes(self) -> tuple[Array, ...]:
        """Return the coordinates of the grid of the current metric."""
        metric = self._session.metric
        assert metric is not None  # a tab is only used once a tensor exists
        return grid_axes(metric["tensor"][0][0].shape, metric["scaling"])

    def _on_session_changed(self) -> None:
        """Update the tab to match the session."""
        raise NotImplementedError


class _JobTab(_AnalysisTab):
    """A tab whose calculation runs in the background.

    The tab has a Run button, which starts the calculation, and shows the
    results it has stored in the session. A subclass says what to run
    (``_start``), what to keep (``_store``) and how to show it
    (``_has_results``, ``_show_results`` and ``_clear_results``).

    Parameters
    ----------
    session : Session
        The state shared with the other pages.
    title : str
        Heading shown above the options.
    """

    def __init__(self, session: Session, title: str) -> None:
        super().__init__(session, title)
        self._run_bar = RunBar("Run", "Computing")
        # The tensor being analysed, so a result for a tensor that has since
        # been replaced can be thrown away.
        self._job_tensor: object = None
        self._run_bar.run_requested.connect(self._run)
        self._run_bar.finished.connect(self._finished)

    def cancel(self) -> None:
        """Stop the running calculation, if there is one."""
        self._run_bar.cancel()

    def _start(self) -> None:
        """Start the calculation with ``self._run_bar.start``."""
        raise NotImplementedError

    def _store(self, value: object) -> None:
        """Put the finished calculation output into the session."""
        raise NotImplementedError

    def _has_results(self) -> bool:
        """Say whether this tab has anything stored in the session to show."""
        raise NotImplementedError

    def _show_results(self) -> None:
        """Show what this tab has stored in the session."""
        raise NotImplementedError

    def _clear_results(self) -> None:
        """Empty the viewer."""
        raise NotImplementedError

    def _update_options(self) -> None:
        """Adapt the options to the current grid. Subclasses may override it."""

    def _show_summary(self) -> None:
        """Update the text about the results. Subclasses may override it."""

    def _run(self) -> None:
        """Remember which tensor is analysed, then start."""
        self._job_tensor = self._session.energy_tensor
        self._start()

    def _finished(self, value: object, seconds: float) -> None:
        """Store a finished calculation and report how long it took."""
        self._store(value)
        self._run_bar.set_status(f"Done in {seconds:.1f} s.")

    def _on_session_changed(self) -> None:
        """Update the tab to match the session."""
        message = None
        if (
            self._run_bar.is_running
            and self._session.energy_tensor is not self._job_tensor
        ):
            # The tensor was replaced while computing, so the answer would
            # belong to a tensor that is no longer there.
            self._run_bar.cancel()
            message = "Run cancelled because the energy tensor changed."

        self._run_bar.set_ready(self._session.energy_tensor is not None)
        self._update_options()

        if self._has_results():
            self._show_results()
        else:
            self._clear_results()
            if not self._run_bar.is_running:
                message = message or "Press Run to compute."
        self._show_summary()
        if message is not None:
            self._run_bar.set_status(message)


class _FieldJobTab(_JobTab):
    """A background tab whose results are fields on the grid.

    The results are shown in a ``FieldView``. A subclass says which stored
    results to show (``_fields``) and sets ``_unit`` to their unit, or leaves it
    empty if it makes no claim about the unit.

    Parameters
    ----------
    session : Session
        The state shared with the other pages.
    title : str
        Heading shown above the options.
    """

    def __init__(self, session: Session, title: str) -> None:
        super().__init__(session, title)
        self._unit = ""
        self._view = FieldView()
        self._add_view(self._view)

    def _fields(self) -> dict[str, Array]:
        """Return the stored results to show, keyed by the name to display."""
        raise NotImplementedError

    def _has_results(self) -> bool:
        """Say whether any field has been computed."""
        return bool(self._fields())

    def _show_results(self) -> None:
        """Show the fields computed so far."""
        self._view.show_fields(self._fields(), self._grid_axes(), self._unit)

    def _clear_results(self) -> None:
        """Empty the viewer."""
        self._view.clear()


class EnergyConditionsTab(_FieldJobTab):
    """Sample the null, weak, strong and dominant energy conditions.

    Every condition is evaluated over a set of observers. Each cell shows the
    worst value found, and a negative value means the condition is violated
    there. Sampling finitely many observers cannot prove a condition holds.

    Parameters
    ----------
    session : Session
        The state shared with the other pages.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, "Energy conditions")
        self._unit = _ENERGY_UNIT

        self._condition = QComboBox()
        self._condition.addItems(CONDITIONS)
        self._directions = QSpinBox()
        self._directions.setRange(1, 1000)
        self._directions.setValue(100)
        self._speeds = QSpinBox()
        self._speeds.setRange(1, 100)
        self._speeds.setValue(10)
        self._results = QLabel()
        self._results.setWordWrap(True)

        self._panel_layout.addWidget(QLabel("<b>Condition</b>"))
        self._panel_layout.addWidget(self._condition)
        self._panel_layout.addWidget(QLabel("<b>Directions to sample</b>"))
        self._panel_layout.addWidget(self._directions)
        self._panel_layout.addWidget(QLabel("<b>Speeds to sample</b>"))
        self._panel_layout.addWidget(self._speeds)
        self._add_note(
            "More samples are more thorough but slower. Speeds are used by the "
            "weak and strong conditions only."
        )
        self._panel_layout.addWidget(self._run_bar)
        self._panel_layout.addWidget(self._results)
        self._finish_panel()

        # The condition being computed, which is kept until it is stored.
        self._running_condition = CONDITIONS[0]
        self._run_bar.running_changed.connect(self._lock_options)
        self._on_session_changed()

    def _lock_options(self, running: bool) -> None:
        """Enable the options only while nothing is running."""
        for widget in (self._condition, self._directions, self._speeds):
            widget.setEnabled(not running)

    def _start(self) -> None:
        """Evaluate the chosen condition in the background."""
        self._running_condition = self._condition.currentText()
        self._run_bar.start(
            energy_condition_map,
            self._session.energy_tensor,
            self._session.metric,
            self._running_condition,
            self._directions.value(),
            self._speeds.value(),
        )

    def _store(self, value: object) -> None:
        """Keep the map of the condition that was evaluated."""
        self._session.add_result(condition_key(self._running_condition), value)

    def _fields(self) -> dict[str, Array]:
        """Return the maps of the conditions evaluated so far."""
        results = self._session.results
        return {
            condition_key(c): results[condition_key(c)]
            for c in CONDITIONS
            if condition_key(c) in results
        }

    def _show_summary(self) -> None:
        """List the worst value and violation count of each condition."""
        lines = []
        for name, values in self._fields().items():
            summary = summarise_condition(values)
            lines.append(
                f"<b>{name}</b><br>worst {summary.worst:.3g} {_ENERGY_UNIT}, "
                f"violated in {summary.violated} of {summary.total} cells"
            )
        self._results.setText("<br><br>".join(lines))


class ScalarsTab(_FieldJobTab):
    """Compute the expansion, shear and vorticity of the Eulerian observers.

    Parameters
    ----------
    session : Session
        The state shared with the other pages.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, "Kinematic scalars")
        self._add_note(
            "Expansion, shear and vorticity of the observers at rest in the "
            "grid. They are computed with the same derivative order as the "
            "solved energy tensor."
        )
        self._panel_layout.addWidget(self._run_bar)
        self._finish_panel()
        self._on_session_changed()

    def _start(self) -> None:
        """Compute the scalars in the background."""
        tensor = self._session.energy_tensor
        assert tensor is not None  # Run is only enabled once a tensor exists
        self._run_bar.start(scalar_fields, self._session.metric, tensor["order"])

    def _store(self, value: object) -> None:
        """Keep each of the three scalars."""
        assert isinstance(value, dict)
        for name, values in value.items():
            self._session.add_result(scalar_key(name), values)

    def _fields(self) -> dict[str, Array]:
        """Return the scalars computed so far."""
        results = self._session.results
        return {
            scalar_key(s): results[scalar_key(s)]
            for s in SCALARS
            if scalar_key(s) in results
        }


class FlowLinesTab(_JobTab):
    """Trace the lines along which momentum flows.

    Seed points are laid out in a square on a plane through the grid, and every
    line follows the momentum density from its seed in small forward steps.
    The lines are drawn over the energy density on the same plane.

    Parameters
    ----------
    session : Session
        The state shared with the other pages.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, "Momentum flow lines")
        self._view = FigureView()
        self._add_view(self._view)

        self._plane = QComboBox()
        self._plane.addItems(list(PLANES))
        self._position = QSpinBox()
        self._position_label = QLabel("<b>Plane position</b>")
        self._seeds = QSpinBox()
        self._seeds.setRange(1, 15)
        self._seeds.setValue(5)
        self._step = QDoubleSpinBox()
        self._step.setDecimals(2)
        self._step.setRange(0.01, 2.0)
        self._step.setSingleStep(0.05)
        self._step.setValue(0.25)
        self._max_steps = QSpinBox()
        self._max_steps.setRange(1, 5000)
        self._max_steps.setValue(400)
        self._time = QSpinBox()
        self._time_label = QLabel("<b>Time step</b>")
        # Number of time steps the range was last fitted to.
        self._time_steps = 0

        self._panel_layout.addWidget(QLabel("<b>Seed plane</b>"))
        self._panel_layout.addWidget(self._plane)
        self._panel_layout.addWidget(self._position_label)
        self._panel_layout.addWidget(self._position)
        self._panel_layout.addWidget(self._time_label)
        self._panel_layout.addWidget(self._time)
        self._panel_layout.addWidget(QLabel("<b>Seeds per side</b>"))
        self._panel_layout.addWidget(self._seeds)
        self._panel_layout.addWidget(QLabel("<b>Step size (grid cells)</b>"))
        self._panel_layout.addWidget(self._step)
        self._panel_layout.addWidget(QLabel("<b>Most steps per line</b>"))
        self._panel_layout.addWidget(self._max_steps)
        self._add_note(
            "Where the momentum is strongest, one step moves a line by the step "
            "size. Lines stop at the edge of the grid. More seeds and more steps "
            "take longer."
        )
        self._panel_layout.addWidget(self._run_bar)
        self._finish_panel()

        self._plane.currentIndexChanged.connect(self._plane_changed)
        self._run_bar.running_changed.connect(self._lock_options)
        self._on_session_changed()

    def _lock_options(self, running: bool) -> None:
        """Enable the options only while nothing is running."""
        for widget in (
            self._plane,
            self._position,
            self._time,
            self._seeds,
            self._step,
            self._max_steps,
        ):
            widget.setEnabled(not running)

    def _fixed_axis_points(self) -> int:
        """Return the number of grid points along the axis the plane holds fixed."""
        metric = self._session.metric
        assert metric is not None  # only called once a metric exists
        _, _, fixed = PLANES[self._plane.currentText()]
        return int(metric["tensor"][0][0].shape[fixed])

    def _set_position_range(self, keep_value: bool) -> None:
        """Let the plane move over the inside of the grid along its fixed axis.

        A line cannot be traced from a point on the edge of the grid, so the
        plane can be placed from the second point to the second to last. The
        position is a one-based grid index.
        """
        _, _, fixed = PLANES[self._plane.currentText()]
        # With fewer than three points there is no inside, so allow only the
        # smallest valid range.
        last = max(2, self._fixed_axis_points() - 1)
        old = self._position.value()
        self._position_label.setText(
            f"<b>Plane position ({AXIS_NAMES[fixed]} index)</b>"
        )
        self._position.blockSignals(True)
        self._position.setRange(2, last)
        # Start in the middle, unless the user already chose a place that is
        # still inside the grid.
        self._position.setValue(
            old if keep_value and 2 <= old <= last else (2 + last) // 2
        )
        self._position.blockSignals(False)

    def _plane_changed(self) -> None:
        """Start the plane in the middle of the newly chosen axis."""
        if self._session.metric is not None:
            self._set_position_range(keep_value=False)

    def _update_options(self) -> None:
        """Fit the position and time step controls to the current grid."""
        metric = self._session.metric
        if metric is None:
            return
        self._set_position_range(keep_value=True)
        steps = int(metric["tensor"][0][0].shape[0])
        self._time.blockSignals(True)
        self._time.setRange(1, steps)
        if steps != self._time_steps:
            # A new grid: start in the middle, as for the plane position.
            self._time.setValue((steps + 1) // 2)
            self._time_steps = steps
        self._time.blockSignals(False)
        # With a single time step there is nothing to choose.
        self._time.setVisible(steps > 1)
        self._time_label.setVisible(steps > 1)

    def _start(self) -> None:
        """Trace the lines in the background."""
        self._run_bar.start(
            momentum_flow_lines,
            self._session.energy_tensor,
            self._plane.currentText(),
            self._position.value() - 1,
            self._seeds.value(),
            self._step.value(),
            self._max_steps.value(),
            self._time.value() - 1,
        )

    def _store(self, value: object) -> None:
        """Keep the traced lines."""
        self._session.add_result(FLOW_LINES, value)

    def _has_results(self) -> bool:
        """Say whether lines have been traced for this tensor."""
        return FLOW_LINES in self._session.results

    def _show_results(self) -> None:
        """Draw the lines over the energy density, if that has been computed."""
        results = self._session.results
        figure = flow_line_figure(
            results[FLOW_LINES], self._grid_axes(), results.get(ENERGY_DENSITY)
        )
        self._view.show_figure(figure)

    def _clear_results(self) -> None:
        """Empty the viewer."""
        self._view.clear()


class EnergyDensityTab(_AnalysisTab):
    """Show the energy density that observers at rest in the grid measure.

    It is quick to compute, so it needs no Run button: it is worked out as
    soon as the tensor is solved.

    Parameters
    ----------
    session : Session
        The state shared with the other pages.
    """

    def __init__(self, session: Session) -> None:
        super().__init__(session, "Energy density")
        self._view = FieldView()
        self._add_view(self._view)
        self._status = QLabel()
        self._status.setWordWrap(True)
        self._add_note(
            "Energy density measured by observers at rest in the grid, in "
            f"{_ENERGY_UNIT}. It is computed automatically when the tensor is "
            "solved."
        )
        self._panel_layout.addWidget(self._status)
        self._finish_panel()
        self._on_session_changed()

    def _on_session_changed(self) -> None:
        """Compute the density if it is missing, and show it."""
        tensor = self._session.energy_tensor
        metric = self._session.metric
        results = self._session.results

        if tensor is None or metric is None:
            self._view.clear()
            self._status.setText("")
        elif ENERGY_DENSITY not in results:
            try:
                density = eulerian_energy_density(tensor, metric)
            except ValueError as error:
                self._view.clear()
                self._status.setText(f"Could not compute it: {error}")
                return
            # Storing it notifies every listener, including this tab, which
            # then draws it.
            self._status.setText("")
            self._session.add_result(ENERGY_DENSITY, density)
        else:
            self._view.show_fields(
                {ENERGY_DENSITY: results[ENERGY_DENSITY]},
                self._grid_axes(),
                _ENERGY_UNIT,
            )


class ResultsPage(QWidget):
    """Analyse the solved energy tensor, with one tab for each analysis.

    Parameters
    ----------
    session : Session
        The state shared with the other pages.
    """

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._tabs: list[_AnalysisTab] = [
            EnergyDensityTab(session),
            EnergyConditionsTab(session),
            ScalarsTab(session),
            FlowLinesTab(session),
        ]
        names = ("Energy density", "Energy conditions", "Scalars", "Flow lines")

        tab_widget = QTabWidget()
        for tab, name in zip(self._tabs, names, strict=True):
            tab_widget.addTab(tab, name)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>Results</h2>"))
        layout.addWidget(tab_widget, stretch=1)

    def cancel(self) -> None:
        """Stop any calculation running in any tab."""
        for tab in self._tabs:
            tab.cancel()
