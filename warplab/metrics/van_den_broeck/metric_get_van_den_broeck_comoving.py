from datetime import date as _date

import numpy as np

from warplab._typing import GridSize, TensorDict, Vector
from warplab.metrics.set_minkowski import set_minkowski
from warplab.metrics.utils.shape_function_alcubierre import shape_function_alcubierre


def metric_get_van_den_broeck_comoving(
    grid_size: GridSize,
    world_centre: Vector,
    v: float,
    R_1: float,
    sigma_1: float,
    R_2: float,
    sigma_2: float,
    A: float,
    grid_scale: Vector | None = None,
) -> TensorDict:
    """Build the Van den Broeck metric in comoving coordinates.

    Parameters
    ----------
    grid_size : array_like of int, shape (4,)
        Number of samples along the (t, x, y, z) axes.
    world_centre : array_like, shape (4,)
        Coordinate offsets in (t, x, y, z) order. Grid coordinates are
        (index + 1) * spacing - world_centre.
    v : float
        Bubble speed parameter in units of the speed of light.
    R_1 : float
        Radius of the inner spatial-expansion profile.
    sigma_1 : float
        Steepness of the inner profile.
    R_2 : float
        Radius of the outer shift profile.
    sigma_2 : float
        Steepness of the outer profile.
    A : float
        Amplitude of the spatial-expansion profile.
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
    The effective velocity parameter is v * (1 + A)**2.
    """
    if grid_scale is None:
        grid_scale = np.array([1, 1, 1, 1])

    if grid_size[0] > 1:
        raise ValueError(
            f"Comoving Van Den Broeck requires one time slice, got {grid_size[0]}"
        )

    metric: TensorDict = {}

    metric["params"] = {}

    metric["params"]["grid_size"] = grid_size
    metric["params"]["world_centre"] = world_centre
    metric["params"]["velocity"] = v * ((1 + A) ** 2)
    metric["params"]["R_1"] = R_1
    metric["params"]["sigma_1"] = sigma_1
    metric["params"]["R_2"] = R_2
    metric["params"]["sigma_2"] = sigma_2
    metric["params"]["A"] = A

    metric["type"] = "metric"
    metric["name"] = "Van Den Broeck Comoving"
    metric["scaling"] = grid_scale
    metric["coords"] = "cartesian"
    metric["index"] = "covariant"
    metric["date"] = _date.today().isoformat()

    metric["tensor"] = set_minkowski(grid_size)

    t = 0

    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):
                x = (i + 1) * grid_scale[1] - world_centre[1]
                y = (j + 1) * grid_scale[2] - world_centre[2]
                z = (k + 1) * grid_scale[3] - world_centre[3]

                r = np.sqrt((x**2) + (y**2) + (z**2))

                # B is a conformal factor that scales the spatial metric: B = 1 far away
                # and 1 + A inside the radius R_1. f_s is the Alcubierre-style shift
                # profile (radius R_2, sharpness sigma_2) times the speed v.
                B = 1 + shape_function_alcubierre(r, R_1, sigma_1) * A

                f_s = shape_function_alcubierre(r, R_2, sigma_2) * v

                # Comoving frame, centre at the origin: spatial metric B^2 times the
                # identity, shift beta^x = v - f_s with lapse 1, so g_0x = B^2 (v - f_s)
                # and g_00 = -(1 - B^2 (v - f_s)^2).
                metric["tensor"][1][1][t, i, j, k] = B**2
                metric["tensor"][2][2][t, i, j, k] = B**2
                metric["tensor"][3][3][t, i, j, k] = B**2

                metric["tensor"][0][1][t, i, j, k] = (B**2) * (v - f_s)
                metric["tensor"][1][0][t, i, j, k] = metric["tensor"][0][1][t, i, j, k]

                metric["tensor"][0][0][t, i, j, k] = -(1 - (B**2) * ((v - f_s) ** 2))

    return metric
