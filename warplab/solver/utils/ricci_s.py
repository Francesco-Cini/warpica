def ricci_s(R_munu, gu):
    """Contract the Ricci tensor with the inverse metric.

    Parameters
    ----------
    R_munu : sequence of sequences of arrays
        Covariant Ricci tensor components.
    gu : sequence of sequences of arrays
        Contravariant metric components on the same grid.

    Returns
    -------
    R : array
        Ricci scalar on the input grid.
    """
    R = 0

    for mu in range(4):
        for nu in range(4):
            R = R + gu[mu][nu] * R_munu[mu][nu]

    return R
