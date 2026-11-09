from warpica.solver.utils.second_order.ricci_t_2 import _ricci_tensor


def ricci_t_mem_2(g_u, g_l, delta):
    """Compute covariant Ricci components with second-order differences.

    Parameters
    ----------
    g_u : sequence of sequences of arrays
        Contravariant metric components on the same grid.
    g_l : sequence of sequences of arrays
        Covariant metric components on the (t, x, y, z) grid.
    delta : array_like, shape (4,)
        Grid spacings in (t, x, y, z) order.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.

    Notes
    -----
    This memory-saving variant recomputes second derivatives. Like the
    MATLAB variant, it expects the first spacing in length units (ct).
    Multiply delta[0] by c() when comparing with ricci_t_2.
    """
    return _ricci_tensor(g_u, g_l, delta, cache_second=False, time_is_seconds=False)
