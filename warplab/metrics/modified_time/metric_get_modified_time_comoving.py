from datetime import date as _date

import numpy as np

from warplab.metrics.set_minkowski import set_minkowski
from warplab.metrics.utils.shape_function_alcubierre import shape_function_alcubierre


def metric_get_modified_time_comoving(
    grid_size, world_centre, v, R, sigma, A, grid_scaling=None
):
    """Build the modified-time metric in comoving coordinates.

    Parameters
    ----------
    grid_size : array_like of int, shape (4,)
        Number of samples along the (t, x, y, z) axes.
    world_centre : array_like, shape (4,)
        Coordinate offsets in (t, x, y, z) order. Grid coordinates are
        (index + 1) * spacing - world_centre.
    v : float
        Bubble speed parameter in units of the speed of light.
    R : float
        Bubble radius in spatial coordinate units.
    sigma : float
        Wall steepness, in inverse spatial coordinate units.
    A : float
        Lapse modification factor; the interior lapse approaches 1 / A.
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
    The comoving construction requires a single time slice.
    """
    if grid_scaling is None:
        grid_scaling = np.array([1, 1, 1, 1])

    if grid_size[0] > 1:
        raise Exception(
            "The time grid is greater than 1, only a size of 1 can be used in comoving"
        )

    metric = {}

    metric["params"] = {}

    metric["params"]["grid_size"] = grid_size
    metric["params"]["world_centre"] = world_centre
    metric["params"]["velocity"] = v
    metric["params"]["R"] = R
    metric["params"]["sigma"] = sigma
    metric["params"]["A"] = A

    metric["type"] = "metric"
    metric["frame"] = "comoving"
    metric["name"] = "Modified Time Comoving"
    metric["scaling"] = grid_scaling
    metric["coords"] = "cartesian"
    metric["index"] = "covariant"
    metric["date"] = _date.today().isoformat()

    metric["tensor"] = set_minkowski(grid_size)

    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):
                x = (i + 1) * grid_scaling[1] - world_centre[1]
                y = (j + 1) * grid_scaling[2] - world_centre[2]
                z = (k + 1) * grid_scaling[3] - world_centre[3]

                for t in range(grid_size[0]):
                    xs = 0

                    r = (((x - xs) ** 2) + (y**2) + (z**2)) ** (1 / 2)

                    fs = shape_function_alcubierre(r, R, sigma)

                    metric["tensor"][0][1][t, i, j, k] = v * (1 - fs)
                    metric["tensor"][1][0][t, i, j, k] = metric["tensor"][0][1][
                        t, i, j, k
                    ]

                    metric["tensor"][0][0][t, i, j, k] = -(((1 - fs) + fs / A) ** 2) + (
                        (v * (1 - fs)) ** 2
                    )

    return metric
