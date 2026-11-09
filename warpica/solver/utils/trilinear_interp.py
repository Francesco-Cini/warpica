import numpy as np


def trilinear_interp(f, x):
    """Interpolate a 3D field at one-based grid coordinates.

    Parameters
    ----------
    f : array_like, ndim=3
        Scalar field on a regular spatial grid.
    x : array_like, shape (3,)
        One-based coordinates within the field, including its boundary.

    Returns
    -------
    value : scalar
        Trilinearly interpolated field value.
    """
    f = np.asarray(f)
    # Positions are one-based grid coordinates (1 is the first sample); shift to
    # zero-based array indices.
    point = np.asarray(x, dtype=float) - 1
    if f.ndim != 3 or point.shape != (3,):
        raise ValueError("Expected a 3D field and three coordinates")
    if (
        not np.isfinite(point).all()
        or np.any(point < 0)
        or np.any(point > np.array(f.shape) - 1)
    ):
        raise ValueError("Interpolation point is outside the field")
    # Weighted average of the 8 samples around the point: along each axis a corner
    # gets weight `fraction` on the upper side and `1 - fraction` on the lower.
    lower = np.floor(point).astype(int)
    upper = np.minimum(lower + 1, np.array(f.shape) - 1)
    fraction = point - lower
    result = 0.0
    for i in range(2):
        for j in range(2):
            for k in range(2):
                bits = np.array([i, j, k])
                index = np.where(bits, upper, lower)
                weight = np.prod(np.where(bits, fraction, 1 - fraction))
                result += weight * f[tuple(index)]
    return result
