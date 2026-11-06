from datetime import date as _date

import numpy as np

from warplab._typing import GridSize, TensorDict, Vector
from warplab.metrics.set_minkowski_three_plus_one import set_minkowski_three_plus_one
from warplab.metrics.three_plus_one_builder import three_plus_one_builder
from warplab.units.universal_constants.c import c


def metric_get_lentz(
    grid_size: GridSize,
    world_centre: Vector,
    v: float,
    scale: float | None,
    grid_scale: Vector | None = None,
) -> TensorDict:
    """Build the Lentz metric.

    Parameters
    ----------
    grid_size : array_like of int, shape (4,)
        Number of samples along the (t, x, y, z) axes.
    world_centre : array_like, shape (4,)
        Coordinate offsets in (t, x, y, z) order. Grid coordinates are
        (index + 1) * spacing - world_centre.
    v : float
        Bubble speed parameter in units of the speed of light.
    scale : float or None
        Size of the piecewise Lentz regions. None uses the largest spatial
        grid dimension divided by seven.
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
    """
    if scale is None:
        # Default region size: one seventh of the largest spatial grid dimension.
        # Note this is a sample count, not a length in metres.
        scale = np.max(np.asarray(grid_size)[1:4] / 7)

    if grid_scale is None:
        grid_scale = np.array([1, 1, 1, 1])

    metric: TensorDict = {}

    metric["params"] = {}

    metric["params"]["grid_size"] = grid_size
    metric["params"]["world_centre"] = world_centre
    metric["params"]["velocity"] = v

    metric["type"] = "metric"
    metric["name"] = "Lentz"
    metric["scaling"] = grid_scale
    metric["coords"] = "cartesian"
    metric["index"] = "covariant"
    metric["date"] = _date.today().isoformat()

    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):
                x = (i + 1) * grid_scale[1] - world_centre[1]
                y = (j + 1) * grid_scale[2] - world_centre[2]

                for t in range(grid_size[0]):
                    xs = ((t + 1) * grid_scale[0] - world_centre[0]) * v * c()

                    xp = x - xs

                    WFX, WFY = get_warp_factor_by_region(xp, y, scale)

                    # The shift has x and y components only; the Lentz construction does
                    # not depend on z.
                    beta[0][t, i, j, k] = -WFX * v
                    beta[1][t, i, j, k] = WFY * v

    metric["tensor"] = three_plus_one_builder(alpha, beta, gamma)

    return metric


def get_warp_factor_by_region(x_in, y_in, size_scale):
    """Evaluate the piecewise Lentz shift factors in the x-y plane.

    Parameters
    ----------
    x_in, y_in : float
        Coordinates relative to the bubble centre.
    size_scale : float
        Spatial scale defining region boundaries.

    Returns
    -------
    factor_x, factor_y : float
        Dimensionless shift factors; factor_y changes sign across y = 0.
    """
    # Piecewise-constant warp factors. Each branch below tests one polygonal region
    # of the (x, y) plane, measured in units of size_scale, and sets constant
    # factors (WFX, WFY) for the shift. y is folded to |y| so one set of regions
    # serves both halves; the sign of y_in is restored for WFY at the end.
    x = x_in
    y = np.abs(y_in)

    WFX = 0
    WFY = 0

    if (x >= size_scale and x <= 2 * size_scale) and (x - size_scale >= y):
        WFX = -2
        WFY = 0

    elif (
        (x > size_scale and x <= 2 * size_scale)
        and (x - size_scale <= y)
        and (-y + 3 * size_scale >= x)
    ):
        WFX = -1
        WFY = 1

    elif (x > 0 and x <= size_scale) and (x + size_scale > y) and (-y + size_scale < x):
        WFX = 0
        WFY = 1

    elif (
        (x > 0 and x <= size_scale)
        and (x + size_scale <= y)
        and (-y + 3 * size_scale >= x)
    ):
        WFX = -0.5
        WFY = 0.5

    elif (
        (x > -size_scale and x <= 0)
        and (-x + size_scale < y)
        and (-y + 3 * size_scale >= -x)
    ):
        WFX = 0.5
        WFY = 0.5

    elif (
        (x > -size_scale and x <= 0)
        and (x + size_scale <= y)
        and (-y + size_scale >= x)
    ):
        WFX = 1
        WFY = 0

    elif (x >= -size_scale and x <= size_scale) and (x + size_scale > y):
        WFX = 1
        WFY = 0

    WFY = np.sign(y_in) * WFY

    return WFX, WFY
