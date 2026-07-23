def get_christoffel_sym(g_u, diff_1_gl, i, k, l):
    
    gamma = 0
    for m in range(4):
        gamma = gamma + (1 / 2) * g_u[i][m] * (diff_1_gl[m][k][l] + diff_1_gl[m][l][k] - diff_1_gl[k][l][m])

    return gamma
