def ricci_s(R_munu, gu):
    
    R = 0

    for mu in range(4):
        for nu in range(4):
            R = R + gu[mu][nu] * R_munu[mu][nu]
    
    return R