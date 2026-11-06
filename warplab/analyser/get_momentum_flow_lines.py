from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from warplab._typing import Array, TensorDict
from warplab.solver.utils.strcmpi import strcmpi
from warplab.solver.utils.trilinear_interp import trilinear_interp


def get_momentum_flow_lines(
    energy_tensor: TensorDict,
    start_points: Sequence[ArrayLike] | Array,
    step_size: float,
    max_steps: int,
    scale_factor: float,
) -> list[Array]:
    """Trace momentum-flow paths by explicit Euler integration.

    Parameters
    ----------
    energy_tensor : dict
        Contravariant stress-energy tensor containing 3D component fields
        or 4D fields with a singleton time axis.
    start_points : array_like, shape (3, ...)
        Starting x, y, and z coordinates in one-based grid units.
    step_size : float
        Positive integration step multiplying the sampled momentum.
    max_steps : int
        Maximum number of integration steps, at least zero.
    scale_factor : float
        Multiplier applied to each momentum component.

    Returns
    -------
    paths : list of ndarrays
        One array of shape (n_steps + 1, 3) per seed. Paths stop at the
        interpolation boundary or a NaN position.
    """
    if not strcmpi(energy_tensor["index"], "contravariant"):
        raise ValueError(
            "Energy tensor for momentum flowlines should be contravariant."
        )

    if max_steps < 0 or step_size <= 0:
        raise ValueError("max_steps must be nonnegative and step_size positive")

    def spatial_field(component):
        field = np.asarray(component)
        if field.ndim == 4 and field.shape[0] == 1:
            field = field[0]
        if field.ndim != 3:
            raise ValueError("Select one time slice before tracing momentum flow")
        return field * scale_factor

    # Momentum density components T^0x, T^0y, T^0z: the vector field whose flow
    # lines are traced. scale_factor rescales them.
    x_mom = spatial_field(energy_tensor["tensor"][0][1])
    y_mom = spatial_field(energy_tensor["tensor"][0][2])
    z_mom = spatial_field(energy_tensor["tensor"][0][3])

    starting_points_x = np.ravel(start_points[0])
    starting_points_y = np.ravel(start_points[1])
    starting_points_z = np.ravel(start_points[2])

    # One path per starting point. Positions are in one-based grid coordinates.
    paths: list[Any] = [None for _ in range(max(starting_points_x.shape))]

    for j in range(max(starting_points_x.shape)):
        pos = np.zeros((max_steps + 1, 3))

        pos[0, :] = np.array(
            [starting_points_x[j], starting_points_y[j], starting_points_z[j]]
        )

        # Keep the interpolation stencil inside the one-based spatial grid.
        count = 1
        # Follow the field with forward Euler steps: move by step_size times the
        # momentum interpolated at the current position. Stop at a NaN or near the
        # grid edge.
        for i in range(max_steps):
            if (
                np.sum(np.isnan(pos[i, :])) > 0
                or (np.floor(pos[i, 0]) <= 1 or np.ceil(pos[i, 0]) >= x_mom.shape[0])
                or (np.floor(pos[i, 1]) <= 1 or np.ceil(pos[i, 1]) >= x_mom.shape[1])
                or (np.floor(pos[i, 2]) <= 1 or np.ceil(pos[i, 2]) >= x_mom.shape[2])
            ):
                break

            x_momentum = trilinear_interp(x_mom, pos[i, :])
            y_momentum = trilinear_interp(y_mom, pos[i, :])
            z_momentum = trilinear_interp(z_mom, pos[i, :])

            pos[i + 1, 0] = pos[i, 0] + x_momentum * step_size
            pos[i + 1, 1] = pos[i, 1] + y_momentum * step_size
            pos[i + 1, 2] = pos[i, 2] + z_momentum * step_size

            count = i + 2

        paths[j] = pos[:count, :]

    return paths
