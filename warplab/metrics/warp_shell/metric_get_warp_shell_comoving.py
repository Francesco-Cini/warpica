"""WarpFactory spherical shell construction; see docs/testing.md for limitations."""

import numpy as np

from warplab._typing import GridSize, TensorDict, Vector
from warplab.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warplab.units.universal_constants.c import c
from warplab.units.universal_constants.G import G


# Cumulative integral of y(x) by the trapezoid rule, starting from 0.
def _cumtrapz(y, x):
    return np.concatenate(([0.0], np.cumsum((y[1:] + y[:-1]) * np.diff(x) / 2)))


def _radial_limit(radius, R_2):
    """Return the largest sampled radius, covering the grid and the shell."""
    return 1.2 * max(float(radius.max()), R_2)


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
    grid_size: GridSize,
    world_centre: Vector,
    m: float,
    R_1: float,
    R_2: float,
    R_buff: float | None = 0,
    sigma: float | None = 0,
    smooth_factor: float | None = 1,
    v_warp: float | None = 0,
    do_warp: bool = False,
    grid_scaling: Vector | None = None,
) -> TensorDict:
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
        raise ValueError(
            f"Comoving warp shell requires one time slice, got {grid_size[0]}"
        )
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
    # Build the radial profile on a fine 1-D grid (100,000 points out to 1.2 times
    # the largest radius needed), solve for the metric functions on it, and
    # interpolate onto the 3-D grid at the end. `radius` is the distance of each 3-D
    # grid point from the centre.
    radius = np.sqrt(sum(x * x for x in xyz))
    r = np.linspace(0, _radial_limit(radius, R_2), 100_000)
    # Uniform mass density: total mass m spread evenly over the shell R_1 < r < R_2.
    rho = m / (4 * np.pi / 3 * (R_2**3 - R_1**3)) * ((r > R_1) & (r < R_2))
    mass = _cumtrapz(4 * np.pi * rho * r * r, r)
    # Schwarzschild radius of the total mass. A uniform sphere cannot be in static
    # equilibrium once r_s reaches 8/9 of its radius (Buchdahl's limit).
    rs = 2 * G() * mass[-1] / c() ** 2
    if rs >= 8 * R_2 / 9:
        raise ValueError(
            "Shell exceeds the constant-density pressure model's compactness limit"
        )
    # Pressure that holds the shell up, from the constant-density (interior
    # Schwarzschild) formula P = rho c^2 (a - b) / (b - 3a), where a and b are
    # defined below. Zero outside r = R_2.
    pressure = np.zeros_like(r)
    inside = r < R_2
    a = R_2 * np.sqrt(R_2 - rs)
    b = np.sqrt(R_2**3 - rs * r[inside] ** 2)
    pressure[inside] = c() ** 2 * rho[inside] * (a - b) / (b - 3 * a)
    # Keep the unsmoothed profiles for the returned params.
    params = {"rho": rho.copy(), "P": pressure.copy(), "rVec": r}
    # Smooth the sharp-edged profiles with four passes of a moving average (the
    # density window is 1.79 times the pressure window). smooth_factor counts
    # radial samples.
    for _ in range(4):
        rho = _smooth(rho, 1.79 * smooth_factor)
        pressure = _smooth(pressure, smooth_factor)
    # Enclosed mass m(r) of the smoothed density, and the compactness
    # 2 G m(r) / (r c^2). Compactness >= 1 would be a horizon, which is rejected.
    mass = _cumtrapz(4 * np.pi * rho * r * r, r)
    compactness = np.zeros_like(r)
    compactness[1:] = 2 * G() * mass[1:] / r[1:] / c() ** 2
    if np.any(compactness >= 1):
        raise ValueError("Shell profile contains a coordinate horizon")
    # Radial spatial metric component g_rr = 1 / (1 - 2 G m(r) / (r c^2)).
    B = 1 / (1 - compactness)
    # Gradient of the log of the lapse from the hydrostatic (TOV) equation, using
    # the enclosed mass and the pressure. Integrating gives alpha(r), fixed so that
    # it matches the exterior Schwarzschild value at the outer edge of the grid.
    dalpha = np.zeros_like(r)
    dalpha[1:] = (
        G() * mass[1:] / c() ** 2
        + 4 * np.pi * G() * r[1:] ** 3 * pressure[1:] / c() ** 4
    ) / (r[1:] ** 2 * (1 - compactness[1:]))
    alpha = _cumtrapz(dalpha, r)
    alpha += 0.5 * np.log1p(-compactness[-1]) - alpha[-1]
    # Time-time metric component g_tt = -exp(2 alpha) (negative: signature -+++).
    A = -np.exp(2 * alpha)
    # Shift profile: 1 inside R_1 + R_buff, 0 outside R_2 - R_buff, and a smooth
    # logistic transition between. The exponent diverges at both ends of the
    # transition, so the profile reaches exactly 1 and 0 there.
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
    # Interpolate the radial functions onto the 3-D grid (linear interpolation).
    metric["tensor"][0, 0, 0] = np.interp(radius, r, A)
    # Spatial metric g_ij = delta_ij + (B - 1) n_i n_j, where n is the unit radial
    # vector (taken as 0 at the centre to avoid dividing by zero).
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
    # Optional warp: add a shift along x, g_0x = -v_warp * shift(r).
    if do_warp:
        metric["tensor"][0, 1, 0] = -v_warp * np.interp(radius, r, shift)
        metric["tensor"][1, 0] = metric["tensor"][0, 1]
    params.update(rhosmooth=rho, Psmooth=pressure, M=mass, A=A, B=B)
    metric["params"] = params
    return metric
