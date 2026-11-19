import math
import time

import pytest

# The widget tests need the optional ui extra and a Qt platform plugin.
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from warpica.ui.core.jobs import BackgroundJob  # noqa: E402
from warpica.ui.widgets import run_bar as run_bar_module  # noqa: E402
from warpica.ui.widgets.run_bar import RunBar  # noqa: E402

TIMEOUT_MS = 60_000


@pytest.fixture
def bar(qtbot):
    run_bar = RunBar("Run", "Working")
    qtbot.addWidget(run_bar)
    run_bar.set_ready(True)
    yield run_bar
    # Never leave a background process running after a test.
    run_bar.cancel()


def use_a_job_that_never_ends(monkeypatch):
    """Make the bar start a 30 second sleep instead of the job it is given."""
    monkeypatch.setattr(
        run_bar_module,
        "BackgroundJob",
        lambda *args, **kwargs: BackgroundJob(time.sleep, 30),
    )


def test_the_run_button_stays_off_until_the_page_is_ready(qtbot):
    run_bar = RunBar("Run", "Working")
    qtbot.addWidget(run_bar)

    assert not run_bar.run_button.isEnabled()

    run_bar.set_ready(True)

    assert run_bar.run_button.isEnabled()


def test_pressing_run_asks_the_page_to_start(bar, qtbot):
    with qtbot.waitSignal(bar.run_requested, timeout=1000):
        bar.run_button.click()


def test_a_running_job_locks_run_and_shows_busy_and_unlocks_cancel(bar, monkeypatch):
    use_a_job_that_never_ends(monkeypatch)

    bar.start(math.fsum, [1.0])

    assert bar.is_running
    assert not bar.run_button.isEnabled()
    assert bar.cancel_button.isEnabled()
    assert not bar.progress.isHidden()
    assert bar.status.text() == "Working... 0.0 s"


def test_a_finished_job_reports_its_value_and_how_long_it_took(bar, qtbot):
    with qtbot.waitSignal(bar.finished, timeout=TIMEOUT_MS) as blocker:
        bar.start(math.fsum, [1.0, 2.0])

    value, seconds = blocker.args
    assert value == 3.0
    assert seconds > 0
    assert not bar.is_running
    assert bar.run_button.isEnabled()
    assert bar.progress.isHidden()


def test_a_failed_job_shows_the_error_and_reports_no_value(bar, qtbot):
    finished = []
    bar.finished.connect(lambda *args: finished.append(args))

    bar.start(int, "not a number")
    qtbot.waitUntil(lambda: not bar.is_running, timeout=TIMEOUT_MS)

    assert bar.status.text().startswith("Run failed: ValueError")
    assert finished == []
    assert bar.run_button.isEnabled()


def test_the_cancel_button_stops_the_job_and_says_so(bar, monkeypatch):
    use_a_job_that_never_ends(monkeypatch)
    bar.start(math.fsum, [1.0])

    bar.cancel_button.click()

    assert not bar.is_running
    assert bar.status.text() == "Run cancelled."
    assert bar.run_button.isEnabled()


def test_cancelling_from_code_leaves_the_status_for_the_page_to_set(bar, monkeypatch):
    use_a_job_that_never_ends(monkeypatch)
    bar.start(math.fsum, [1.0])
    bar.set_status("Something else")

    bar.cancel()

    assert not bar.is_running
    assert bar.status.text() == "Something else"


def test_cancelling_when_nothing_runs_does_nothing(bar):
    bar.set_status("Idle")

    bar.cancel()

    assert bar.status.text() == "Idle"


def test_running_changed_reports_when_a_job_starts_and_ends(bar, qtbot):
    changes = []
    bar.running_changed.connect(changes.append)

    with qtbot.waitSignal(bar.finished, timeout=TIMEOUT_MS):
        bar.start(math.fsum, [1.0])

    assert changes == [True, False]


def test_starting_again_while_running_is_ignored(bar, monkeypatch):
    use_a_job_that_never_ends(monkeypatch)
    bar.start(math.fsum, [1.0])
    job = bar._job

    bar.start(math.fsum, [2.0])

    assert bar._job is job


def test_a_process_that_cannot_start_is_reported(bar, monkeypatch):
    def cannot_start(*args, **kwargs):
        raise OSError("no more processes")

    monkeypatch.setattr(run_bar_module, "BackgroundJob", cannot_start)

    bar.start(math.fsum, [1.0])

    assert bar.status.text() == "Could not start: no more processes"
    assert not bar.is_running
    assert bar.run_button.isEnabled()


def test_the_timer_stops_if_the_job_has_already_gone(bar):
    bar._timer.start()

    bar._check()

    assert not bar._timer.isActive()
