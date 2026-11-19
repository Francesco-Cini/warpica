"""Helpers for looking at a field on the (t, x, y, z) grid through a flat slice."""

import numpy as np

from warpica._typing import Array, Vector

AXIS_NAMES = ("t", "x", "y", "z")

# For each plane the user can pick: the axis drawn horizontally, the axis drawn
# vertically, and the axis held fixed. Axes are numbered as in the grid:
# 0 = t, 1 = x, 2 = y, 3 = z.
PLANES: dict[str, tuple[int, int, int]] = {
    "x-y": (1, 2, 3),
    "x-z": (1, 3, 2),
    "y-z": (2, 3, 1),
}


def grid_axes(shape: tuple[int, ...], scaling: Vector) -> tuple[Array, ...]:
    """Return the coordinate of every grid point along t, x, y and z.

    The middle of each axis is at zero. This matches the grid the Metric page
    builds, where a point's coordinate is ``(index + 1) * spacing - centre``
    and the centre is ``(size + 1) / 2 * spacing``.

    Parameters
    ----------
    shape : tuple of int
        Number of points along (t, x, y, z).
    scaling : array_like, shape (4,)
        Spacing along (t, x, y, z): seconds, then metres.

    Returns
    -------
    tuple of ndarray
        One coordinate array per axis, in the order (t, x, y, z).
    """
    return tuple(
        (np.arange(points) + 1 - (points + 1) / 2) * spacing
        for points, spacing in zip(shape, scaling, strict=True)
    )


def axis_extent(coordinates: Array) -> tuple[float, float]:
    """Return the outer edges of a row of grid cells, for drawing.

    ``imshow`` wants the edges of the first and last cell, but the coordinates
    are the cell centres, so each end is pushed out by half a cell.

    Parameters
    ----------
    coordinates : ndarray
        Cell-centre coordinates along one axis, in increasing order.

    Returns
    -------
    tuple of float
        Lower and upper edge.
    """
    # A single point has no spacing, so give it a cell one unit wide.
    half = (coordinates[1] - coordinates[0]) / 2 if len(coordinates) > 1 else 0.5
    return float(coordinates[0] - half), float(coordinates[-1] + half)


def take_slice(field: Array, plane: str, position: int, time_index: int = 0) -> Array:
    """Cut a flat slice out of a four-dimensional field.

    Parameters
    ----------
    field : ndarray, shape (nt, nx, ny, nz)
        The field to cut.
    plane : {'x-y', 'x-z', 'y-z'}
        Which two axes the slice spans.
    position : int
        Zero-based index along the axis that is held fixed.
    time_index : int, optional
        Zero-based index along t.

    Returns
    -------
    ndarray, shape (n_first, n_second)
        The slice, indexed first by the horizontal axis and then by the
        vertical one.
    """
    _, _, fixed = PLANES[plane]
    index: list[int | slice] = [slice(None)] * 4
    index[0] = time_index
    index[fixed] = position
    return np.asarray(field)[tuple(index)]
