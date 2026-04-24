import numpy as np

from warp_factory_py.units.universal_constants.c import c

from warp_factory_py.solver.utils.take_finite_difference_1 import take_finite_difference_1
from warp_factory_py.solver.utils.take_finite_difference_2 import take_finite_difference_2

def ricci_t(gu, gl, delta):

    s = gl[0][0].shape

    R_munu = [[None for _ in range(4)] for _ in range(4)]

    diff_1_gl = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)] 
    diff_2_gl = [[[[None for _ in range(4)] for _ in range(4)] for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            phi_phi_flag = 0 

            for k in range(4):
                diff_1_gl[i][j][k] = take_finite_difference_1(gl[i][j], k, delta, phi_phi_flag)
                if k == 1:
                    diff_1_gl[i][j][k] = (1 / c()) * diff_1_gl[i][j][k]

                for n in range(k,4):
                    diff_2_gl[i][j][k][n] = take_finite_difference_2(gl[i][j], k, n, delta, phi_phi_flag)

                    if (n == 1 and k != 1) or (n != 1 and k == 1):
                        diff_2_gl[i][j][k][n] = (1 / c()) * diff_2_gl[i][j][k][n]

                    if k != n:
                        diff_2_gl[i][j][n][k] = diff_2_gl[i][j][k][n]
                
    for k in range(4):
        diff_1_gl[1][0][k] = diff_1_gl[0][1][k]
        diff_1_gl[2][0][k] = diff_1_gl[0][2][k]
        diff_1_gl[2][1][k] = diff_1_gl[1][2][k]
        diff_1_gl[3][0][k] = diff_1_gl[0][3][k]
        diff_1_gl[3][1][k] = diff_1_gl[1][3][k]
        diff_1_gl[3][2][k] = diff_1_gl[2][3][k]
        
        for n in range(4):
            diff_2_gl[1][0][k][n] = diff_2_gl[0][1][k][n]
            diff_2_gl[2][0][k][n] = diff_2_gl[0][2][k][n]
            diff_2_gl[2][1][k][n] = diff_2_gl[1][2][k][n]
            diff_2_gl[3][0][k][n] = diff_2_gl[0][3][k][n]
            diff_2_gl[3][1][k][n] = diff_2_gl[1][3][k][n]
            diff_2_gl[3][2][k][n] = diff_2_gl[2][3][k][n]

    for i in range(4):
        for j in range(4):

            R_munu_temp = np.zeros(s)

            for a in range(4):
                for b in range(4):
                    R_munu_temp_2 = np.zeros(s)

                    R_munu_temp_2 = R_munu_temp_2 - (diff_2_gl[i][j][a][b] + diff_2_gl[a][b][i][j] - diff_2_gl[i][b][j][a] - diff_2_gl[j][b][i][a])

                    for r in range(4):

                        R_munu_temp_3 = np.zeros(s)
                        R_munu_temp_4 = np.zeros(s)
                        R_munu_temp_5 = np.zeros(s)

                        for d in range(4):

                            R_munu_temp_3 = R_munu_temp_3 + diff_1_gl[b][d][j] * gu[r][d]
                            R_munu_temp_4 = R_munu_temp_4 + (diff_1_gl[j][d][b] - diff_1_gl[j][b][d]) * gu[r][d]

                            R_munu_temp_5 = R_munu_temp_5 - (diff_1_gl[b][d][a] + diff_1_gl[b][d][a] - diff_1_gl[a][b][d]) * gu[r][d]

                        R_munu_temp_2 = R_munu_temp_2 + R_munu_temp_4 * diff_1_gl[i][r][a] + (1 / 2) * (R_munu_temp_3 * diff_1_gl[a][r][i] + R_munu_temp_5 * (diff_1_gl[j][r][i] + diff_1_gl[i][r][j] - diff_1_gl[j][i][r]))

                    R_munu_temp = R_munu_temp + gu[a][b] * R_munu_temp_2

            R_munu[i][j] = (1 / 2) * R_munu_temp

    R_munu[1][0] = R_munu[0][1]
    R_munu[2][0] = R_munu[0][2]
    R_munu[2][1] = R_munu[1][2]
    R_munu[3][0] = R_munu[0][3]
    R_munu[3][1] = R_munu[1][3]
    R_munu[3][2] = R_munu[2][3]

    return R_munu