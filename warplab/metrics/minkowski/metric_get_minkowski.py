from datetime import datetime

import numpy as np

from warplab._typing import GridSize, TensorDict, Vector


def metric_get_minkowski(
    grid_size: GridSize,
    grid_scaling: Vector | None = None,
) -> TensorDict:
    """Build the Minkowski metric.

    Parameters
    ----------
    grid_size : array_like of int, shape (4,)
        Number of samples along the (t, x, y, z) axes.
    grid_scaling : array_like, shape (4,), optional
        Grid spacings in (t, x, y, z) order, in seconds and metres.
        Defaults to unit spacing.

    Returns
    -------
    metric : dict
        Cartesian covariant metric, grid scaling, and identifying metadata.
        Components use tensor[mu][nu] with shape (nt, nx, ny, nz).

    Notes
    -----
    The metric uses signature (-, +, +, +).
    """
    if grid_scaling is None:
        grid_scaling = np.array([1, 1, 1, 1])

    metric: TensorDict = {}

    metric["type"] = "metric"
    metric["name"] = "Minkowski"
    metric["scaling"] = grid_scaling
    metric["coords"] = "cartesian"
    metric["index"] = "covariant"
    metric["date"] = datetime.now()

    # Flat space diag(-1, 1, 1, 1) on the grid; unlike the other generators this one
    # builds the (4, 4, t, x, y, z) array directly.
    metric["tensor"] = np.zeros((4, 4, *grid_size))

    metric["tensor"][0][0] = -np.ones(grid_size)

    metric["tensor"][1][1] = np.ones(grid_size)
    metric["tensor"][2][2] = np.ones(grid_size)
    metric["tensor"][3][3] = np.ones(grid_size)

    metric["tensor"][0][1] = np.zeros(grid_size)
    metric["tensor"][1][0] = np.zeros(grid_size)
    metric["tensor"][0][2] = np.zeros(grid_size)
    metric["tensor"][2][0] = np.zeros(grid_size)
    metric["tensor"][1][2] = np.zeros(grid_size)
    metric["tensor"][2][1] = np.zeros(grid_size)
    metric["tensor"][1][3] = np.zeros(grid_size)
    metric["tensor"][3][1] = np.zeros(grid_size)
    metric["tensor"][2][3] = np.zeros(grid_size)
    metric["tensor"][3][2] = np.zeros(grid_size)
    metric["tensor"][0][3] = np.zeros(grid_size)
    metric["tensor"][3][0] = np.zeros(grid_size)

    return metric
