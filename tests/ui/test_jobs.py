import math
import os
import pickle
import time

import pytest

from warpica.ui.core import worker
from warpica.ui.core.jobs import BackgroundJob


def _wait_for(job, timeout=60.0):
    """Poll a job until it ends and return its outcome."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        outcome = job.poll()
        if outcome is not None:
            return outcome
        time.sleep(0.02)
    job.cancel()
    pytest.fail("The job did not finish in time.")


def test_a_job_returns_the_value_of_its_function():
    job = BackgroundJob(math.fsum, [1.5, 2.5, 3.0])

    outcome = _wait_for(job)

    assert outcome.error is None
    assert outcome.value == 7.0


def test_keyword_arguments_reach_the_function():
    job = BackgroundJob(int, "ff", base=16)

    assert _wait_for(job).value == 255


def test_a_job_that_raises_reports_the_error_as_text():
    job = BackgroundJob(int, "not a number")

    outcome = _wait_for(job)

    assert outcome.value is None
    assert outcome.error.startswith("ValueError:")
    assert "not a number" in outcome.error


def test_a_running_job_polls_as_none_until_it_finishes():
    job = BackgroundJob(time.sleep, 30)
    try:
        assert job.poll() is None
    finally:
        job.cancel()


def test_cancelling_stops_the_job_and_says_so():
    job = BackgroundJob(time.sleep, 30)

    job.cancel()

    assert job.poll().error == "The job was cancelled."


def test_cancelling_twice_or_after_the_end_does_nothing():
    job = BackgroundJob(math.fsum, [1.0])
    outcome = _wait_for(job)

    job.cancel()
    job.cancel()

    assert job.poll() is outcome


def test_the_outcome_stays_the_same_on_every_poll():
    job = BackgroundJob(math.fsum, [1.0])

    first = _wait_for(job)

    assert job.poll() is first


def test_a_process_that_dies_is_reported_with_its_exit_code():
    job = BackgroundJob(os._exit, 3)

    outcome = _wait_for(job)

    assert outcome.value is None
    assert "stopped unexpectedly (exit code 3)" in outcome.error


def test_the_temporary_folder_is_removed_when_the_job_ends():
    job = BackgroundJob(math.fsum, [1.0])
    folder = job._folder
    assert folder.exists()

    _wait_for(job)

    assert not folder.exists()


def test_the_worker_writes_the_result_for_a_successful_job(tmp_path):
    (tmp_path / "job.pkl").write_bytes(pickle.dumps((math.fsum, ([1.0, 2.0],), {})))

    status = worker.main(str(tmp_path))

    assert status == 0
    assert pickle.loads((tmp_path / "result.pkl").read_bytes()) == 3.0


def test_the_worker_writes_the_error_for_a_failing_job(tmp_path):
    (tmp_path / "job.pkl").write_bytes(pickle.dumps((int, ("x",), {})))

    status = worker.main(str(tmp_path))

    assert status == 1
    assert (tmp_path / "error.txt").read_text(encoding="utf-8").startswith("ValueError")
