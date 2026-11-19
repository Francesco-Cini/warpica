"""The state shared by every page of the interface.

The workflow has three stages, and each one needs the one before it:

1. **Metric:** choose and build a spacetime.
2. **Solver:** compute the stress-energy tensor that produces that metric.
3. **Results:** analyse the tensor (energy conditions, scalars, ...).

``Session`` holds the product of each stage. Changing an earlier stage throws
away everything that depended on it, so a page can never show a tensor or a
result that belongs to a different metric.
"""

from collections.abc import Callable
from enum import Enum
from typing import Any

from warpica._typing import TensorDict


class Stage(Enum):
    """The stages of the workflow, in the order they are done.

    The value is the name shown in the sidebar.
    """

    METRIC = "Metric"
    SOLVER = "Solver"
    RESULTS = "Results"


class Session:
    """What has been built, solved and analysed so far.

    Attributes
    ----------
    metric : dict or None
        The metric built in the Metric stage, or None if there is none yet.
    energy_tensor : dict or None
        The stress-energy tensor from the Solver stage, or None.
    results : dict of str to object
        Analysis outputs from the Results stage, keyed by name, for example
        ``"Weak"`` for the weak energy condition.
    """

    def __init__(self) -> None:
        self.metric: TensorDict | None = None
        self.energy_tensor: TensorDict | None = None
        self.results: dict[str, Any] = {}
        # Functions to call whenever something changes, so the windows can
        # refresh themselves.
        self._listeners: list[Callable[[], None]] = []

    def subscribe(self, listener: Callable[[], None]) -> None:
        """Call ``listener()`` every time the session changes.

        Parameters
        ----------
        listener : callable
            A function that takes no arguments.
        """
        self._listeners.append(listener)

    def set_metric(self, metric: TensorDict) -> None:
        """Store a new metric and discard everything computed from the old one.

        Parameters
        ----------
        metric : dict
            The metric that was built.
        """
        self.metric = metric
        self.energy_tensor = None
        self.results = {}
        self._notify()

    def set_energy_tensor(self, energy_tensor: TensorDict) -> None:
        """Store a solved tensor and discard results computed from the old one.

        Parameters
        ----------
        energy_tensor : dict
            The stress-energy tensor of the current metric.

        Raises
        ------
        ValueError
            If there is no metric yet, because a tensor has to belong to one.
        """
        if self.metric is None:
            raise ValueError("Build a metric before storing its energy tensor.")
        self.energy_tensor = energy_tensor
        self.results = {}
        self._notify()

    def add_result(self, name: str, value: Any) -> None:
        """Store one analysis output, replacing any earlier one with that name.

        Parameters
        ----------
        name : str
            What the result is, for example ``"Weak"``.
        value : object
            The computed output.

        Raises
        ------
        ValueError
            If there is no energy tensor yet, because results are computed
            from it.
        """
        if self.energy_tensor is None:
            raise ValueError("Solve for the energy tensor before storing results.")
        self.results[name] = value
        self._notify()

    def is_unlocked(self, stage: Stage) -> bool:
        """Say whether a stage has the input it needs.

        Parameters
        ----------
        stage : Stage
            The stage to check.

        Returns
        -------
        bool
            True if the user can open this stage. The Metric stage needs
            nothing, the Solver needs a metric and the Results need a solved
            tensor.
        """
        if stage is Stage.SOLVER:
            return self.metric is not None
        if stage is Stage.RESULTS:
            return self.energy_tensor is not None
        return True

    def status(self, stage: Stage) -> str:
        """Return a short phrase describing where a stage stands.

        Parameters
        ----------
        stage : Stage
            The stage to describe.

        Returns
        -------
        str
            Text for the sidebar, such as ``"not solved"``.
        """
        if stage is Stage.METRIC:
            return "built" if self.metric is not None else "not built"
        if stage is Stage.SOLVER:
            if self.metric is None:
                return "needs a metric"
            return "solved" if self.energy_tensor is not None else "not solved"
        if self.energy_tensor is None:
            return "needs a solution"
        if not self.results:
            return "no results yet"
        return f"{len(self.results)} computed"

    def _notify(self) -> None:
        """Tell every subscriber that the session changed."""
        for listener in self._listeners:
            listener()
