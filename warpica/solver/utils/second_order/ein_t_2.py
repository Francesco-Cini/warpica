def ein_t_2(R_munu, R, g_l):
    """Form the covariant Einstein tensor from Ricci curvature.

    Parameters
    ----------
    R_munu : sequence of sequences of arrays
        Covariant Ricci tensor components.
    R : array
        Ricci scalar on the input grid.
    g_l : sequence of sequences of arrays
        Covariant metric components on the (t, x, y, z) grid.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.
    """
    E = [[None for _ in range(4)] for _ in range(4)]

    for mu in range(4):
        for nu in range(4):
            E[mu][nu] = R_munu[mu][nu] - 0.5 * g_l[mu][nu] * R

    return E
