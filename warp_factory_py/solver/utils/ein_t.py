def ein_t(R_munu, R, gl):

    E = [[None for _ in range(4)] for _ in range(4)]

    for mu in range(4):
        for nu in range(4):
            E[mu][nu] = R_munu[mu][nu] - (1 / 2) * gl[mu][nu] * R

    return E 