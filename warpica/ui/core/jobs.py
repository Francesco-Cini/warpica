"""Run a slow function in a separate process so the window stays responsive.

The solver is a single long call with no way to report progress or to stop
part-way. A Python thread cannot be stopped from outside, so cancelling a
thread would only hide the result while the work carried on. A separate
process can really be stopped, so that is what is used here.

The job is passed to the process through files in a temporary folder, and the
result comes back the same way. The parent only has to check from time to time
whether the process has finished, so no Qt threads or signals are involved.
"""

import pickle
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class JobOutcome:
    """How a background job ended.

    Attributes
    ----------
    value : object
        What the function returned. None if the job failed or was cancelled.
    error : str or None
        A message describing the failure, or None if the job succeeded.
    """

    value: Any = None
    error: str | None = None


class BackgroundJob:
    """Start ``function(*args, **kwargs)`` in a new process.

    The job starts as soon as the object is created. Call ``poll`` repeatedly
    to find out when it has finished, and ``cancel`` to stop it.

    Parameters
    ----------
    function : callable
        A function defined at the top level of a module, so the new process
        can find it by name.
    *args, **kwargs
        Arguments for the function. They must be picklable.
    """

    def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        self._temp = tempfile.TemporaryDirectory(prefix="warpica-")
        self._folder = Path(self._temp.name)
        (self._folder / "job.pkl").write_bytes(pickle.dumps((function, args, kwargs)))

        self._outcome: JobOutcome | None = None
        # Whatever the process prints to stderr is kept, to explain a crash.
        self._stderr = open(self._folder / "stderr.txt", "wb")  # noqa: SIM115
        self._process = subprocess.Popen(
            [sys.executable, "-m", "warpica.ui.core.worker", str(self._folder)],
            stdout=subprocess.DEVNULL,
            stderr=self._stderr,
            # On Windows, do not flash a console window for the new process.
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def poll(self) -> JobOutcome | None:
        """Check whether the job has finished, without waiting.

        Returns
        -------
        JobOutcome or None
            None while the job is still running. Once it has finished, or been
            cancelled, the same outcome is returned on every call.
        """
        if self._outcome is not None:
            return self._outcome
        exit_code = self._process.poll()
        if exit_code is None:
            return None
        self._outcome = self._read_outcome(exit_code)
        self._cleanup()
        return self._outcome

    def cancel(self) -> None:
        """Stop the job now. Does nothing if it has already ended."""
        if self._outcome is not None:
            return
        self._process.kill()
        self._process.wait()
        self._outcome = JobOutcome(error="The job was cancelled.")
        self._cleanup()

    def _read_outcome(self, exit_code: int) -> JobOutcome:
        """Collect the result or the error the process left in the folder."""
        result_file = self._folder / "result.pkl"
        error_file = self._folder / "error.txt"
        if exit_code == 0 and result_file.exists():
            return JobOutcome(value=pickle.loads(result_file.read_bytes()))
        if error_file.exists():
            return JobOutcome(error=error_file.read_text(encoding="utf-8"))

        # No result and no error message: the process was stopped from outside
        # (for example by the system running out of memory) or could not start.
        # The last line it printed to stderr usually says why.
        self._stderr.flush()
        lines = (self._folder / "stderr.txt").read_text(errors="replace").splitlines()
        detail = f": {lines[-1]}" if lines else ""
        return JobOutcome(
            error=f"The background process stopped unexpectedly "
            f"(exit code {exit_code}){detail}"
        )

    def _cleanup(self) -> None:
        """Close the log and delete the temporary folder."""
        self._stderr.close()
        self._temp.cleanup()
