_STENCIL_SIZE = {"second": 3, "fourth": 5}
_AXES = ("time", "x", "y", "z")


def verify_grid_size(shape, diff_order):
    """Raise if a grid axis is too short for the finite-difference order.

    Parameters
    ----------
    shape : sequence of int, shape (4,)
        Grid size in (t, x, y, z) order.
    diff_order : {'second', 'fourth'}
        Finite-difference accuracy order, compared case-insensitively. Any
        other value is left for the caller to reject.

    Raises
    ------
    ValueError
        If an axis has more than one sample but fewer than the stencil needs
        (3 for second order, 5 for fourth order).

    Notes
    -----
    The derivative functions return zeros along an axis that is too short,
    which would silently drop the time dependence or a spatial gradient. An
    axis of exactly one sample is allowed: nothing varies along it, so zero is
    the correct derivative for static, comoving and homogeneous fields.
    """
    required = _STENCIL_SIZE.get(str(diff_order).lower())
    if required is None:
        return
    for axis, size in zip(_AXES, shape):
        if 1 < size < required:
            remedies = ["add samples", "use a single sample if nothing varies"]
            if str(diff_order).lower() == "fourth" and size >= _STENCIL_SIZE["second"]:
                remedies.append("use diff_order='second'")
            raise ValueError(
                f"diff_order={diff_order!r} needs at least {required} samples "
                f"along the {axis} axis, or exactly 1 (got {size}); "
                + ", ".join(remedies[:-1])
                + f" or {remedies[-1]}"
            )
