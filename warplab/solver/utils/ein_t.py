def ein_t(R_munu, R, gl):
    """Form the covariant Einstein tensor from Ricci curvature.

    Parameters
    ----------
    R_munu : sequence of sequences of arrays
        Covariant Ricci tensor components.
    R : array
        Ricci scalar on the input grid.
    gl : sequence of sequences of arrays
        Covariant metric components on the (t, x, y, z) grid.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.
    """
    E = [[None for _ in range(4)] for _ in range(4)]

    for mu in range(4):
        for nu in range(4):
            E[mu][nu] = R_munu[mu][nu] - (1 / 2) * gl[mu][nu] * R

    return E
