from datetime import date as _date

import numpy as np

from warpica._typing import GridSize, TensorDict, Vector
from warpica.metrics.set_minkowski_three_plus_one import set_minkowski_three_plus_one
from warpica.metrics.three_plus_one_builder import three_plus_one_builder
from warpica.metrics.utils.shape_function_alcubierre import shape_function_alcubierre


def metric_get_alcubierre_comoving(
    grid_size: GridSize,
    world_centre: Vector,
    v: float,
    R: float,
    sigma: float,
    grid_scale: Vector | None = None,
) -> TensorDict:
    """Build the Alcubierre metric in comoving coordinates.

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
    grid_scale : array_like, shape (4,), optional
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
    if grid_scale is None:
        grid_scale = np.array([1, 1, 1, 1])

    if grid_size[0] > 1:
        raise ValueError(
            f"Comoving Alcubierre requires one time slice, got {grid_size[0]}"
        )

    metric: TensorDict = {}

    metric["params"] = {}

    metric["params"]["grid_size"] = grid_size
    metric["params"]["world_centre"] = world_centre
    metric["params"]["velocity"] = v
    metric["params"]["R"] = R
    metric["params"]["sigma"] = sigma

    metric["type"] = "metric"
    metric["name"] = "Alcubierre Comoving"
    metric["scaling"] = grid_scale
    metric["coords"] = "cartesian"
    metric["index"] = "covariant"
    metric["date"] = _date.today().isoformat()

    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    # Single time slice: in the bubble's own frame the metric does not depend on
    # time.
    t = 0
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):
                x = (i + 1) * grid_scale[1] - world_centre[1]
                y = (j + 1) * grid_scale[2] - world_centre[2]
                z = (k + 1) * grid_scale[3] - world_centre[3]

                r = ((x**2) + (y**2) + (z**2)) ** (1 / 2)

                fs = shape_function_alcubierre(r, R, sigma)

                # Bubble frame: the centre sits at the origin. The shift is 0 inside (fs
                # ~ 1, the bubble is at rest) and v far away, where space streams past
                # it.
                beta[0][t, i, j, k] = v * (1 - fs)

    metric["tensor"] = three_plus_one_builder(alpha, beta, gamma)

    return metric
