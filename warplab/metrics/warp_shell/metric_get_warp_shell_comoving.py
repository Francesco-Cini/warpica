"""WarpFactory spherical shell construction; see docs/testing.md for limitations."""

import numpy as np

from warplab.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warplab.units.universal_constants.c import c
from warplab.units.universal_constants.G import G


def _cumtrapz(y, x):
    return np.concatenate(([0.0], np.cumsum((y[1:] + y[:-1]) * np.diff(x) / 2)))


def _smooth(values, span):
    # MATLAB smooth's centered moving average, with shrinking endpoint windows.
    width = int(span * len(values)) if 0 < span < 1 else int(span)
    width = min(width, len(values))
    width = max(1, width - (width % 2 == 0))
    idx = np.arange(len(values))
    radius = np.minimum(np.minimum(idx, len(values) - 1 - idx), width // 2)
    sums = np.concatenate(([0.0], np.cumsum(values)))
    return (sums[idx + radius + 1] - sums[idx - radius]) / (2 * radius + 1)


def metric_get_warp_shell_comoving(
    grid_size,
    world_centre,
    m,
    R_1,
    R_2,
    R_buff=0,
    sigma=0,
    smooth_factor=1,
    v_warp=0,
    do_warp=False,
    grid_scaling=None,
):
    """Build a static spherical shell with an optional x-directed shift.

    Parameters
    ----------
    grid_size : array_like of int, shape (4,)
        Grid shape (nt, nx, ny, nz), requiring nt = 1.
    world_centre : array_like, shape (4,)
        Coordinate offsets; spatial coordinates use (index + 1) * spacing
        minus the corresponding offset.
    m : float
        Nonnegative shell mass in kilograms.
    R_1, R_2 : float
        Inner and outer shell radii in metres, with 0 <= R_1 < R_2.
    R_buff : float or None, optional
        Buffer at each end of the shift transition; default is zero.
    sigma : float or None, optional
        Nonnegative sigmoid sharpness; default is zero.
    smooth_factor : float or None, optional
        Moving-average span, or fraction of radial samples when between
        zero and one. Defaults to one.
    v_warp : float or None, optional
        Dimensionless x-directed shift amplitude; default is zero.
    do_warp : bool, optional
        Include the shift when true; defaults to false.
    grid_scaling : array_like, shape (4,), optional
        Grid spacings in seconds and metres; defaults to unit spacing.

    Returns
    -------
    metric : dict
        Covariant Cartesian metric with radial profiles stored in params.

    Notes
    -----
    Radial profiles use 100,000 samples and linear interpolation. The shell
    must satisfy the constant-density pressure model's compactness limit
    and the smoothed profile must contain no coordinate horizon.
    """
    R_buff = 0 if R_buff is None else R_buff
    sigma = 0 if sigma is None else sigma
    smooth_factor = 1 if smooth_factor is None else smooth_factor
    v_warp = 0 if v_warp is None else v_warp
    if grid_size[0] != 1:
        raise ValueError("Comoving warp shell requires one time slice")
    if not (m >= 0 and 0 <= R_1 < R_2 and 0 <= 2 * R_buff < R_2 - R_1):
        raise ValueError("Require nonnegative mass, 0 <= R_1 < R_2, and a valid buffer")
    if smooth_factor < 0 or sigma < 0:
        raise ValueError("Smoothing and sigmoid sharpness must be nonnegative")
    metric = metric_get_minkowski(grid_size, grid_scaling)
    metric["name"] = "Comoving Warp Shell"
    xyz = np.meshgrid(
        *[
            (np.arange(grid_size[i]) + 1) * metric["scaling"][i] - world_centre[i]
            for i in range(1, 4)
        ],
        indexing="ij",
    )
    radius = np.sqrt(sum(x * x for x in xyz))
    r = np.linspace(0, 1.2 * max(float(radius.max()), R_2), 100_000)
    rho = m / (4 * np.pi / 3 * (R_2**3 - R_1**3)) * ((r > R_1) & (r < R_2))
    mass = _cumtrapz(4 * np.pi * rho * r * r, r)
    rs = 2 * G() * mass[-1] / c() ** 2
    if rs >= 8 * R_2 / 9:
        raise ValueError(
            "Shell exceeds the constant-density pressure model's compactness limit"
        )
    pressure = np.zeros_like(r)
    inside = r < R_2
    a = R_2 * np.sqrt(R_2 - rs)
    b = np.sqrt(R_2**3 - rs * r[inside] ** 2)
    pressure[inside] = c() ** 2 * rho[inside] * (a - b) / (b - 3 * a)
    params = {"rho": rho.copy(), "P": pressure.copy(), "rVec": r}
    for _ in range(4):
        rho = _smooth(rho, 1.79 * smooth_factor)
        pressure = _smooth(pressure, smooth_factor)
    mass = _cumtrapz(4 * np.pi * rho * r * r, r)
    compactness = np.zeros_like(r)
    compactness[1:] = 2 * G() * mass[1:] / r[1:] / c() ** 2
    if np.any(compactness >= 1):
        raise ValueError("Shell profile contains a coordinate horizon")
    B = 1 / (1 - compactness)
    dalpha = np.zeros_like(r)
    dalpha[1:] = (
        G() * mass[1:] / c() ** 2
        + 4 * np.pi * G() * r[1:] ** 3 * pressure[1:] / c() ** 4
    ) / (r[1:] ** 2 * (1 - compactness[1:]))
    alpha = _cumtrapz(dalpha, r)
    alpha += 0.5 * np.log1p(-compactness[-1]) - alpha[-1]
    A = -np.exp(2 * alpha)
    shift = np.ones_like(r)
    low, high = R_1 + R_buff, R_2 - R_buff
    shift[r >= high] = 0
    transition = (r > low) & (r < high)
    exponent = (
        (high - low)
        * (sigma + 2)
        / 2
        * (1 / (r[transition] - high) + 1 / (r[transition] - low))
    )
    shift[transition] = 1 / (1 + np.exp(-np.clip(exponent, -700, 700)))
    for _ in range(2):
        shift = _smooth(shift, smooth_factor)
    metric["tensor"][0, 0, 0] = np.interp(radius, r, A)
    radial = np.interp(radius, r, B) - 1
    direction = [
        np.divide(x, radius, out=np.zeros_like(x), where=radius != 0) for x in xyz
    ]
    for i in range(3):
        for j in range(i, 3):
            metric["tensor"][i + 1, j + 1, 0] = (i == j) + radial * direction[
                i
            ] * direction[j]
            metric["tensor"][j + 1, i + 1, 0] = metric["tensor"][i + 1, j + 1, 0]
    if do_warp:
        metric["tensor"][0, 1, 0] = -v_warp * np.interp(radius, r, shift)
        metric["tensor"][1, 0] = metric["tensor"][0, 1]
    params.update(rhosmooth=rho, Psmooth=pressure, M=mass, A=A, B=B)
    metric["params"] = params
    return metric
