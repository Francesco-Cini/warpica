def label_cartesian_axis(plane):
    """Return the labels of the two unfixed Cartesian axes.

    Parameters
    ----------
    plane : sequence of int
        Two distinct fixed axes using the one-based range 1 through 4.

    Returns
    -------
    labels : tuple of str
        Remaining coordinate labels in (t, x, y, z) order.
    """
    if (
        len(plane) != 2
        or len(set(plane)) != 2
        or any(p not in (1, 2, 3, 4) for p in plane)
    ):
        raise ValueError("Select two distinct axes between 1 and 4")
    return tuple(
        label for i, label in enumerate(("t", "x", "y", "z"), 1) if i not in plane
    )
