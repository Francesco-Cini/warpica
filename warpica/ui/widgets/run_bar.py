"""Buttons, busy bar and status line for one job that runs in the background."""

import time
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from warpica.ui.core.jobs import BackgroundJob


class RunBar(QWidget):
    """Start a job, show that it is working and let the user cancel it.

    The page that owns the bar decides what to run. When the Run button is
    pressed the bar emits ``run_requested``, and the page answers by calling
    ``start`` with the function and its arguments. The bar then shows a busy
    indicator and the elapsed time, and reports back with ``finished``.

    Signals
    -------
    run_requested
        The user pressed the Run button.
    finished(value, seconds)
        The job succeeded. ``value`` is what the function returned.
    running_changed(running)
        A job started or ended, so the page can lock or unlock its settings.

    Parameters
    ----------
    run_text : str
        Label of the Run button, such as ``"Solve"``. It is also used in
        messages: ``"Solve failed: ..."`` and ``"Solve cancelled."``.
    busy_text : str
        What to say while working, such as ``"Solving"``.
    """

    run_requested = Signal()
    finished = Signal(object, float)
    running_changed = Signal(bool)

    def __init__(self, run_text: str, busy_text: str) -> None:
        super().__init__()
        self._run_text = run_text
        self._busy_text = busy_text

        self.run_button = QPushButton(run_text)
        self.cancel_button = QPushButton("Cancel")
        # A bar with range (0, 0) shows "busy" instead of a percentage, because
        # the calculations cannot report how far they have got.
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.status = QLabel()
        self.status.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.cancel_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(buttons)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)

        self._job: BackgroundJob | None = None
        self._started = 0.0
        # Whether the page has what the job needs. The Run button stays off
        # until it does.
        self._ready = False

        # While a job runs, check on it ten times a second.
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._check)

        self.run_button.clicked.connect(self.run_requested)
        self.cancel_button.clicked.connect(self._cancel_clicked)
        self._update_controls()

    @property
    def is_running(self) -> bool:
        """Whether a job is running now."""
        return self._job is not None

    def set_ready(self, ready: bool) -> None:
        """Say whether the page has what the job needs.

        Parameters
        ----------
        ready : bool
            If False the Run button is disabled.
        """
        self._ready = ready
        self._update_controls()

    def set_status(self, text: str) -> None:
        """Show a message under the buttons."""
        self.status.setText(text)

    def start(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Run ``function(*args, **kwargs)`` in the background.

        Parameters
        ----------
        function : callable
            A function defined at the top level of a module.
        *args, **kwargs
            Arguments for the function. They must be picklable.
        """
        if self._job is not None:
            return
        try:
            job = BackgroundJob(function, *args, **kwargs)
        except OSError as error:
            self.set_status(f"Could not start: {error}")
            return
        self._job = job
        self._started = time.monotonic()
        self.set_status(f"{self._busy_text}... 0.0 s")
        self._update_controls()
        self.running_changed.emit(True)
        self._timer.start()

    def cancel(self) -> None:
        """Stop the running job, if there is one. The status is left alone."""
        if self._job is None:
            return
        self._timer.stop()
        self._job.cancel()
        self._job = None
        self._update_controls()
        self.running_changed.emit(False)

    def _cancel_clicked(self) -> None:
        """Stop the job because the user asked."""
        self.cancel()
        self.set_status(f"{self._run_text} cancelled.")

    def _check(self) -> None:
        """See whether the job has finished, and report how it ended."""
        job = self._job
        if job is None:
            self._timer.stop()
            return
        elapsed = time.monotonic() - self._started
        outcome = job.poll()
        if outcome is None:
            self.set_status(f"{self._busy_text}... {elapsed:.1f} s")
            return

        self._timer.stop()
        self._job = None
        self._update_controls()
        self.running_changed.emit(False)
        if outcome.error is not None:
            self.set_status(f"{self._run_text} failed: {outcome.error}")
            return
        self.finished.emit(outcome.value, elapsed)

    def _update_controls(self) -> None:
        """Enable and show the controls that make sense right now."""
        running = self._job is not None
        self.run_button.setEnabled(self._ready and not running)
        self.cancel_button.setEnabled(running)
        self.progress.setVisible(running)
