from collections.abc import Callable, Sequence
from datetime import datetime

from warpica._typing import Array, GridSize, TensorDict, Vector
from warpica.metrics.set_minkowski_three_plus_one import set_minkowski_three_plus_one
from warpica.metrics.three_plus_one_builder import three_plus_one_builder


def custom_metric(
    grid_size: GridSize,
    world_centre: Vector,
    grid_scaling: Vector,
    alpha_function: Callable[[float, float, float, float], float],
    beta_function: Callable[[float, float, float, float], Vector],
    gamma_function: Callable[
        [float, float, float, float], Sequence[Sequence[float]] | Array
    ],
) -> TensorDict:
    """Build a metric from lapse, covariant shift, and spatial-metric callbacks.

    Parameters
    ----------
    grid_size : array_like of int, shape (4,)
        Number of samples along the (t, x, y, z) axes.
    world_centre : array_like, shape (4,)
        Coordinate offsets in (t, x, y, z) order. Grid coordinates are
        (index + 1) * spacing - world_centre.
    grid_scaling : array_like, shape (4,)
        Grid spacings in (t, x, y, z) order.
    alpha_function : callable
        Function of (t, x, y, z) returning the scalar lapse.
    beta_function : callable
        Function of (t, x, y, z) returning three covariant shift components.
    gamma_function : callable
        Function of (t, x, y, z) returning a symmetric 3 by 3 spatial metric.
        Only the upper triangle is read.

    Returns
    -------
    metric : dict
        Cartesian covariant metric, grid scaling, and identifying metadata.
        Components use tensor[mu][nu] with shape (nt, nx, ny, nz).
    """
    metric: TensorDict = {}

    metric["type"] = "metric"
    metric["name"] = "Custom Metric"
    metric["scaling"] = grid_scaling
    metric["coords"] = "cartesian"
    metric["index"] = "covariant"
    metric["date"] = datetime.now()

    # Start from flat space and overwrite every cell with the user's functions.
    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    for h in range(grid_size[0]):
        for i in range(grid_size[1]):
            for j in range(grid_size[2]):
                for k in range(grid_size[3]):
                    # Physical coordinates of this cell, with the same (index + 1) *
                    # spacing - world_centre convention as the built-in generators.
                    t = (h + 1) * grid_scaling[0] - world_centre[0]
                    x = (i + 1) * grid_scaling[1] - world_centre[1]
                    y = (j + 1) * grid_scaling[2] - world_centre[2]
                    z = (k + 1) * grid_scaling[3] - world_centre[3]

                    alpha_field = alpha_function(t, x, y, z)
                    beta_field = beta_function(t, x, y, z)
                    gamma_field = gamma_function(t, x, y, z)

                    alpha[h, i, j, k] = alpha_field

                    beta[0][h, i, j, k] = beta_field[0]
                    beta[1][h, i, j, k] = beta_field[1]
                    beta[2][h, i, j, k] = beta_field[2]

                    gamma[0][0][h, i, j, k] = gamma_field[0][0]
                    gamma[1][1][h, i, j, k] = gamma_field[1][1]
                    gamma[2][2][h, i, j, k] = gamma_field[2][2]
                    gamma[0][1][h, i, j, k] = gamma_field[0][1]
                    gamma[0][2][h, i, j, k] = gamma_field[0][2]
                    gamma[1][2][h, i, j, k] = gamma_field[1][2]

                    # Only the upper triangle of the spatial metric is read from the
                    # user's function; mirror it to keep gamma symmetric.
                    gamma[1][0][h, i, j, k] = gamma[0][1][h, i, j, k]
                    gamma[2][0][h, i, j, k] = gamma[0][2][h, i, j, k]
                    gamma[2][1][h, i, j, k] = gamma[1][2][h, i, j, k]

    metric["tensor"] = three_plus_one_builder(alpha, beta, gamma)

    return metric
