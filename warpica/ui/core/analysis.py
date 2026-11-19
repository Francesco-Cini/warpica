"""The analysis calculations offered on the Results page.

The ``energy_condition_map``, ``scalar_fields``, ``eulerian_energy_density``
and ``momentum_flow_lines`` functions sit at the top level of this module so
that a background process can find them by name (see ``jobs``).
"""

from dataclasses import dataclass

import numpy as np

from warpica._typing import Array, DiffOrder, TensorDict
from warpica.analyser import (
    do_frame_transfer,
    get_energy_conditions,
    get_momentum_flow_lines,
    get_scalars,
)
from warpica.ui.core.slicing import PLANES

CONDITIONS = ("Null", "Weak", "Strong", "Dominant")
SCALARS = ("Expansion", "Shear", "Vorticity")

# Names under which these results are stored in the session.
ENERGY_DENSITY = "Energy density (Eulerian)"
FLOW_LINES = "Momentum flow lines"


def condition_key(condition: str) -> str:
    """Return the name an energy condition is stored under in the session."""
    return f"{condition} energy condition"


def scalar_key(scalar: str) -> str:
    """Return the name a kinematic scalar is stored under in the session."""
    return f"{scalar} scalar"


def energy_condition_map(
    energy_tensor: TensorDict,
    metric: TensorDict,
    condition: str,
    num_angular_vec: int,
    num_time_vec: int,
) -> Array:
    """Evaluate one energy condition over sampled observer directions.

    Parameters
    ----------
    energy_tensor : dict
        The solved stress-energy tensor.
    metric : dict
        The metric it was solved from.
    condition : {'Null', 'Weak', 'Strong', 'Dominant'}
        The condition to evaluate.
    num_angular_vec : int
        Number of spatial directions to sample.
    num_time_vec : int
        Number of speeds to sample for the timelike conditions.

    Returns
    -------
    ndarray, shape (nt, nx, ny, nz)
        The worst sampled value in each cell, in joules per cubic metre.
        Negative means the condition is violated there.
    """
    return get_energy_conditions(
        energy_tensor, metric, condition, num_angular_vec, num_time_vec
    )


def scalar_fields(metric: TensorDict, diff_order: DiffOrder) -> dict[str, Array]:
    """Compute the kinematic scalars of the Eulerian observers.

    Parameters
    ----------
    metric : dict
        The metric to analyse.
    diff_order : {'second', 'fourth'}
        Finite-difference order.

    Returns
    -------
    dict of str to ndarray
        The expansion, shear and vorticity scalars, each with shape
        (nt, nx, ny, nz).
    """
    expansion, shear, vorticity = get_scalars(metric, diff_order)
    return {"Expansion": expansion, "Shear": shear, "Vorticity": vorticity}


def eulerian_energy_density(energy_tensor: TensorDict, metric: TensorDict) -> Array:
    """Return the energy density measured by observers at rest in the grid.

    Parameters
    ----------
    energy_tensor : dict
        The solved stress-energy tensor.
    metric : dict
        The metric it was solved from.

    Returns
    -------
    ndarray, shape (nt, nx, ny, nz)
        Energy density in joules per cubic metre.
    """
    eulerian = do_frame_transfer(metric, energy_tensor, "Eulerian")
    # In the Eulerian frame, component [0][0] is the energy density.
    return np.asarray(eulerian["tensor"][0][0])


@dataclass(frozen=True)
class ConditionSummary:
    """The headline numbers of an energy condition map.

    Attributes
    ----------
    worst : float
        The lowest value in the map, or NaN if no value is finite.
    violated : int
        Number of cells where the value is negative.
    total : int
        Number of cells in the map.
    """

    worst: float
    violated: int
    total: int


def summarise_condition(values: Array) -> ConditionSummary:
    """Reduce an energy condition map to its headline numbers.

    Parameters
    ----------
    values : ndarray
        The map from ``energy_condition_map``. Negative means violated.

    Returns
    -------
    ConditionSummary
        The lowest value and how many cells are violated.
    """
    finite = values[np.isfinite(values)]
    worst = float(finite.min()) if finite.size else float("nan")
    # NaN compares false, so cells without a value are not counted as violated.
    return ConditionSummary(
        worst=worst, violated=int(np.sum(values < 0)), total=int(values.size)
    )


@dataclass(frozen=True)
class FlowLines:
    """Momentum flow lines traced from a plane of seed points.

    Attributes
    ----------
    paths : list of ndarray
        One array of shape (n_points, 3) for each line, holding the x, y and z
        coordinates of its points in metres, starting at the seed point.
    plane : {'x-y', 'x-z', 'y-z'}
        The plane the seed points lie on.
    position : int
        Zero-based index of that plane along the axis it holds fixed.
    time_index : int
        Zero-based time step the lines were traced in.
    """

    paths: list[Array]
    plane: str
    position: int
    time_index: int


def momentum_flow_lines(
    energy_tensor: TensorDict,
    plane: str,
    position: int,
    seeds_per_side: int,
    step: float,
    max_steps: int,
    time_index: int = 0,
) -> FlowLines:
    """Trace the lines along which momentum flows, from a plane of seeds.

    The seeds are spread evenly over the inside of a plane, in a square
    pattern, and every line follows the momentum density ``T^{0i}`` in small
    forward steps until it reaches the edge of the grid or runs out of steps.

    The tracer multiplies the momentum by a scale factor to turn it into a
    distance. Here that factor is chosen so that where the momentum is
    strongest a step moves exactly ``step`` grid cells, and everywhere else it
    moves less. This makes ``step`` easy to choose and independent of the
    enormous size of the momentum in SI units.

    Parameters
    ----------
    energy_tensor : dict
        The solved, contravariant stress-energy tensor of a grid built by the
        interface, whose middle is at the origin.
    plane : {'x-y', 'x-z', 'y-z'}
        The plane to place the seeds on.
    position : int
        Zero-based index of the plane along the axis it holds fixed. It must
        leave at least one grid point on each side, because lines that start
        on the edge cannot be traced.
    seeds_per_side : int
        The seeds form a square of this many by this many points.
    step : float
        Distance of one step where the momentum is strongest, in grid cells.
    max_steps : int
        Most steps to take along any one line.
    time_index : int, optional
        Zero-based time step to trace in.

    Returns
    -------
    FlowLines
        The traced lines, in metres.

    Raises
    ------
    ValueError
        If an argument is out of range, or the momentum is zero everywhere.
    """
    shape = np.shape(energy_tensor["tensor"][0][0])
    if not 0 <= time_index < shape[0]:
        raise ValueError(f"Time step must be from 0 to {shape[0] - 1}.")
    first, second, fixed = PLANES[plane]
    # Positions are one-based inside the tracer, and a seed must be strictly
    # inside the grid, so the zero-based index runs from 1 to size - 2.
    if not 1 <= position <= shape[fixed] - 2:
        raise ValueError(
            f"The seed plane must be inside the grid: index 1 to {shape[fixed] - 2}."
        )
    if seeds_per_side < 1:
        raise ValueError("There must be at least one seed per side.")

    # The tracer works on one time slice, so keep only the chosen one.
    one_step = dict(energy_tensor)
    one_step["tensor"] = [
        [
            np.asarray(energy_tensor["tensor"][i][j])[time_index : time_index + 1]
            for j in range(4)
        ]
        for i in range(4)
    ]

    # Strength of the momentum density T^{0i} at every point.
    momentum = np.stack([one_step["tensor"][0][k][0] for k in (1, 2, 3)])
    strength = np.sqrt(np.sum(momentum**2, axis=0))
    finite = strength[np.isfinite(strength)]
    strongest = float(finite.max()) if finite.size else 0.0
    if strongest == 0.0:
        raise ValueError(
            "The momentum density is zero everywhere, so there is no flow to trace."
        )

    # Seeds in one-based grid units, spread evenly over the inside of the plane.
    along_first = np.linspace(2, shape[first] - 1, seeds_per_side)
    along_second = np.linspace(2, shape[second] - 1, seeds_per_side)
    first_grid, second_grid = np.meshgrid(along_first, along_second, indexing="ij")
    seeds = np.empty((3, first_grid.size))
    seeds[first - 1] = first_grid.ravel()
    seeds[second - 1] = second_grid.ravel()
    seeds[fixed - 1] = position + 1

    paths = get_momentum_flow_lines(one_step, seeds, step, max_steps, 1.0 / strongest)

    # The tracer reports one-based grid units. Convert to metres, with the
    # middle of the grid at the origin as in ``slicing.grid_axes``.
    scaling = np.asarray(energy_tensor["scaling"], dtype=float)[1:]
    centre = (np.array(shape[1:]) + 1) / 2
    return FlowLines(
        paths=[(path - centre) * scaling for path in paths],
        plane=plane,
        position=position,
        time_index=time_index,
    )
