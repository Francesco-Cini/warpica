import numpy as np

from warplab._typing import GridSize, TensorDict, Vector
from warplab.metrics.minkowski.metric_get_minkowski import metric_get_minkowski


def metric_get_schwarzschild(
    grid_size: GridSize,
    world_centre: Vector,
    r_s: float,
    grid_scaling: Vector | None = None,
) -> TensorDict:
    """Build the Schwarzschild metric.

    Parameters
    ----------
    grid_size : array_like of int, shape (4,)
        Number of samples along the (t, x, y, z) axes.
    world_centre : array_like, shape (4,)
        Coordinate offsets in (t, x, y, z) order. Grid coordinates are
        (index + 1) * spacing - world_centre.
    r_s : float
        Schwarzschild radius in spatial coordinate units.
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
    The metric uses signature (-, +, +, +) and requires a single time slice.
    For nonzero r_s, the grid must exclude the origin and coordinate horizon.
    """
    if grid_size[0] != 1:
        raise ValueError(f"Schwarzschild requires one time slice, got {grid_size[0]}")
    if r_s < 0:
        raise ValueError("r_s must be nonnegative")
    metric = metric_get_minkowski(grid_size, grid_scaling)
    metric.update(
        name="Schwarzschild",
        params={"r_s": r_s, "grid_size": grid_size, "world_centre": world_centre},
    )
    if r_s == 0:
        return metric
    xyz = np.meshgrid(
        *[
            (np.arange(grid_size[i]) + 1) * metric["scaling"][i] - world_centre[i]
            for i in range(1, 4)
        ],
        indexing="ij",
    )
    r = np.sqrt(sum(x * x for x in xyz))
    if np.any(r == 0) or np.any(np.isclose(r, r_s, rtol=1e-12, atol=0)):
        raise ValueError(
            "Grid intersects the Schwarzschild origin or coordinate horizon"
        )
    metric["tensor"][0, 0, 0] = -(1 - r_s / r)
    for i in range(3):
        for j in range(i, 3):
            metric["tensor"][i + 1, j + 1, 0] = (i == j) + r_s * xyz[i] * xyz[j] / (
                r * r * (r - r_s)
            )
            metric["tensor"][j + 1, i + 1, 0] = metric["tensor"][i + 1, j + 1, 0]
    return metric
