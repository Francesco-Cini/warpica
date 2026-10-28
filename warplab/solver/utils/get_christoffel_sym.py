def get_christoffel_sym(g_u, diff_1_gl, i, k, l):
    """Contract metric derivatives to obtain one Christoffel component.

    Parameters
    ----------
    g_u : sequence of sequences of arrays
        Contravariant metric components on the same grid.
    diff_1_gl : nested sequence of arrays
        Metric derivatives indexed [mu][nu][derivative_axis].
    i, k, l : int
        Zero-based indices of Gamma^i_kl.

    Returns
    -------
    gamma : array
        Selected Christoffel component on the grid.
    """
    gamma = 0
    for m in range(4):
        gamma = gamma + (1 / 2) * g_u[i][m] * (
            diff_1_gl[m][k][l] + diff_1_gl[m][l][k] - diff_1_gl[k][l][m]
        )

    return gamma
