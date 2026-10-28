import numpy as np


def get_slice_data(plane, slice_centre, tensor):
    """Convert two fixed one-based axes and locations to NumPy indices.

    Parameters
    ----------
    plane : array_like of int, shape (2,)
        Distinct fixed axes: 1=t, 2=x, 3=y, 4=z.
    slice_centre : array_like, shape (2,), or None
        One-based locations, rounded to nearest cells. None uses middle cells.
    tensor : dict
        Tensor dictionary providing the component grid shape.

    Returns
    -------
    indices : tuple
        Four indices: two integers and two full slices.
    """
    shape = np.shape(tensor["tensor"][0][0])
    plane = np.asarray(plane)
    if (
        plane.shape != (2,)
        or not np.isin(plane, [1, 2, 3, 4]).all()
        or plane[0] == plane[1]
    ):
        raise ValueError("Select two distinct axes from 1=t, 2=x, 3=y, 4=z")
    axes = plane.astype(int) - 1
    if slice_centre is None:
        locations = [(shape[axis] + 1) // 2 for axis in axes]
    else:
        values = np.asarray(slice_centre)
        if values.shape != (2,) or not np.isfinite(values).all():
            raise ValueError("Provide two finite slice locations")
        locations = np.floor(values + 0.5).astype(int)
    indices = [slice(None)] * 4
    for axis, location in zip(axes, locations):
        if not 1 <= location <= shape[axis]:
            raise ValueError("Slice location is outside the grid")
        indices[axis] = int(location) - 1
    return tuple(indices)
