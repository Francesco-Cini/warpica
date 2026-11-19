"""Entry point of the process that carries out a background job.

``BackgroundJob`` starts this module with ``python -m`` and gives it a folder.
The folder holds the job in ``job.pkl``. The worker runs it, then writes either
``result.pkl`` or, if the job raised, ``error.txt``. It is not meant to be run
by hand.
"""

import pickle
import sys
from pathlib import Path


def main(directory: str) -> int:
    """Run the job stored in ``directory`` and write its outcome there.

    Parameters
    ----------
    directory : str
        Folder created by ``BackgroundJob``.

    Returns
    -------
    int
        Exit status: 0 if the job finished, 1 if it raised an error.
    """
    folder = Path(directory)
    function, args, kwargs = pickle.loads((folder / "job.pkl").read_bytes())
    try:
        value = function(*args, **kwargs)
    except Exception as error:
        # Hand the failure to the parent as text so it can show it.
        (folder / "error.txt").write_text(
            f"{type(error).__name__}: {error}", encoding="utf-8"
        )
        return 1
    (folder / "result.pkl").write_bytes(pickle.dumps(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
