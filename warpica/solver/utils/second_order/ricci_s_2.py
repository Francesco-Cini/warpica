def ricci_s_2(R_munu, g_u):
    """Contract the Ricci tensor with the inverse metric.

    Parameters
    ----------
    R_munu : sequence of sequences of arrays
        Covariant Ricci tensor components.
    g_u : sequence of sequences of arrays
        Contravariant metric components on the same grid.

    Returns
    -------
    R : array
        Ricci scalar on the input grid.
    """
    R = 0

    for mu in range(4):
        for nu in range(4):
            R = R + g_u[mu][nu] * R_munu[mu][nu]

    return R
