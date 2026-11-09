from array_api_compat import array_namespace


def tensor_cell_2_array(tensor):
    """Stack tensor components into trailing matrix axes.

    Parameters
    ----------
    tensor : dict
        Tensor containing 4 by 4 component arrays of a common grid shape.

    Returns
    -------
    array : array
        Shape (*grid_shape, 4, 4), preserving the input array backend.
    """
    components = tensor["tensor"]
    flat_components = tuple(components[i][j] for i in range(4) for j in range(4))
    xp = array_namespace(*flat_components)

    rows = tuple(xp.stack(tuple(components[i]), axis=-1) for i in range(4))
    return xp.stack(rows, axis=-2)
