from warp_factory_py.solver.utils.take_finite_difference_1 import take_finite_difference_1
from warp_factory_py.solver.utils.get_christoffel_sym import get_christoffel_sym

def cov_div(g_l, g_u, vec_u, vec_d, idx_div, idx_vec, delta, stair_sel):

    diff_1_gl = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]

    s = g_l[0][0].shape

    for i in range(4):
        for j in range(4):
            if i == 2 and j == 2 and s[1] == 1:
                phi_phi_flag = 1
            else:
                phi_phi_flag = 0
            
            for k in range(4):
                diff_1_gl[i][j][k] = take_finite_difference_1(g_l[i][j], k, delta, phi_phi_flag)

    if stair_sel == 0:
        cd_vec = take_finite_difference_1(vec_d[idx_vec], idx_div, delta, 0)
        
        for i in range(4):
            gamma = get_christoffel_sym(g_u, diff_1_gl, i, idx_vec, idx_div)
            cd_vec = cd_vec - gamma * vec_d[i]

    elif stair_sel == 1:
        cd_vec = take_finite_difference_1(vec_u[idx_vec], idx_div, delta, 0)

        for i in range(4):
            gamma = get_christoffel_sym(g_u, diff_1_gl, idx_vec, idx_div, i)
            cd_vec = cd_vec + gamma * vec_u[i]

    else:
        raise Exception("Invalid variance selected")
    
    return cd_vec
