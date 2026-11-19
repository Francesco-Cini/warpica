"""Catalogue of the metrics the interface can offer.

Each metric is described once here, as data: its name, the numbers a user can
change and how to build it. The window reads this description to create its
input fields, so adding a metric to the interface means adding one entry to
``METRICS`` and no window code.
"""

from collections.abc import Callable
from dataclasses import dataclass

from warpica._typing import Array, TensorDict
from warpica.metrics import metric_get_alcubierre, metric_get_minkowski


@dataclass(frozen=True)
class Parameter:
    """One number the user can change for a metric.

    Attributes
    ----------
    name : str
        Keyword the builder function receives this value under.
    label : str
        Text shown next to the input field.
    default : float
        Value the field starts with.
    minimum, maximum : float
        Smallest and largest value the field accepts.
    step : float
        Amount added or removed by one click on the field's arrows.
    """

    name: str
    label: str
    default: float
    minimum: float
    maximum: float
    step: float


@dataclass(frozen=True)
class MetricSpec:
    """Everything the interface needs to know about one metric.

    Attributes
    ----------
    label : str
        Name shown to the user.
    parameters : tuple of Parameter
        The numbers the user can change, in the order they are shown.
    build : callable
        Called as ``build(grid_size, world_centre, grid_scale, **values)`` where
        ``values`` holds one number per parameter. Returns the metric.
    """

    label: str
    parameters: tuple[Parameter, ...]
    build: Callable[..., TensorDict]

    def defaults(self) -> dict[str, float]:
        """Return the starting value of every parameter, keyed by name."""
        return {parameter.name: parameter.default for parameter in self.parameters}


def _build_alcubierre(
    grid_size: Array,
    world_centre: Array,
    grid_scale: Array,
    v: float,
    R: float,
    sigma: float,
) -> TensorDict:
    """Build the Alcubierre metric from the values the interface collects."""
    return metric_get_alcubierre(grid_size, world_centre, v, R, sigma, grid_scale)


def _build_minkowski(
    grid_size: Array, world_centre: Array, grid_scale: Array
) -> TensorDict:
    """Build flat space. It has no centre, so ``world_centre`` is not used."""
    return metric_get_minkowski(grid_size, grid_scale)


# The default grid is small (a few metres across), so the radius is limited to
# what fits comfortably inside it.
METRICS: dict[str, MetricSpec] = {
    "alcubierre": MetricSpec(
        label="Alcubierre",
        parameters=(
            Parameter("v", "Speed v (units of c)", 0.4, 0.0, 2.0, 0.1),
            Parameter("R", "Bubble radius R (m)", 2.0, 0.5, 3.0, 0.1),
            Parameter("sigma", "Wall steepness σ (1/m)", 1.0, 0.1, 5.0, 0.1),
        ),
        build=_build_alcubierre,
    ),
    "minkowski": MetricSpec(
        label="Minkowski (flat space)",
        parameters=(),
        build=_build_minkowski,
    ),
}
