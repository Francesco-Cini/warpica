import numpy as np

from warp_factory_py.metrics.three_plus_one_decomposer import three_plus_one_decomposer
from warp_factory_py.analyser.change_tensor_index import change_tensor_index
from warp_factory_py.solver.utils.cov_div import cov_div
from warp_factory_py.solver.utils.c4_inv import c4_inv
from warp_factory_py.analyser.utils.get_trace import get_trace

def get_scalars(metric):

    shape = metric["tensor"][0][0].shape
    array_metric_tensor = np.empty((*shape, 4, 4))

    for i in range(4):
        for j in range(4):
            array_metric_tensor[:, :, :, :, i, j] = metric["tensor"][i][j]

    alpha, _, _, beta_up, _ = three_plus_one_decomposer(metric)

    array_beta = np.empty((*shape, 3))

    for i in range(3):
        array_beta[:, :, :, :, i] = beta_up[i]

    s = shape
    u_up = np.zeros((*s, 4))
    u_down = np.zeros((*s, 4))

    for t in range(s[0]):
        for i in range(s[1]):
            for j in range(s[2]):
                for k in range(s[3]):
                    u_up[t, i, j, k, :] = 1 / alpha[t, i , j, k] * np.array([1, -array_beta[t, i, j, k, 0], -array_beta[t, i, j, k, 1], -array_beta[t, i, j, k, 2]])
                    u_down[t, i, j, k, :] = (array_metric_tensor[t, i, j, k, :, :] @ u_up[t, i, j, k, :]) 

    u_up_cell = [None for _ in range(4)]
    u_down_cell = [None for _ in range(4)]

    for i in range(4):
        u_up_cell[i] = u_up[:, :, :, :, i]
        u_down_cell[i] = u_down[:, :, :, :, i]

    del_u = [[None for _ in range(4)] for _ in range(4)]

    metric = change_tensor_index(metric, "covariant")

    delta = np.asarray(metric.get('scaling', np.ones(4)))

    for i in range(4):
        for j in range(4):
            del_u[i][j] = cov_div(metric['tensor'], c4_inv(metric['tensor']), u_up_cell, u_down_cell, i, j, delta, 0)

    P_mix = [[None for _ in range(4)] for _ in range(4)]
    P = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            if i == j:
                k_delta = 1
            else:
                k_delta = 0
            
            P_mix[i][j] = k_delta + u_up_cell[i] * u_down_cell[j]
            P[i][j] = metric['tensor'][i][j] + u_down_cell[i] * u_down_cell[j]

    theta = {}
    theta['index'] = "covariant"
    theta['type'] = "tensor"
    theta['tensor'] = [[None for _ in range(4)] for _ in range(4)]

    omega = {}
    omega['index'] = "covariant"
    omega['type'] = "tensor"
    omega['tensor'] = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            theta['tensor'][i][j] = np.zeros(metric['tensor'][0][0].shape)
            omega['tensor'][i][j] = np.zeros(metric['tensor'][0][0].shape)
            for a in range(4):
                for b in range(4):
                    theta['tensor'][i][j] = theta['tensor'][i][j] + P_mix[a][i] * P_mix[b][j] * (1 / 2) * (del_u[a][b] + del_u[b][a])
                    omega['tensor'][i][j] = omega['tensor'][i][j] + P_mix[a][i] * P_mix[b][j] * (1 / 2) * (del_u[a][b] - del_u[b][a])

    theta_trace = get_trace(theta, metric)

    omega_up = change_tensor_index(omega, "contravariant", metric)
    omega_trace = np.zeros(metric['tensor'][0][0].shape)

    for mu in range(4):
        for nu in range(4):
            omega_trace = omega_trace + (1 / 2) * omega_up['tensor'][mu][nu] * omega['tensor'][mu][nu]

    shear = {}
    shear['index'] = "covariant"
    shear['type'] = "tensor"
    shear['tensor'] = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            shear['tensor'][i][j] = theta['tensor'][i][j] - theta_trace * (1 / 3) * P[i][j]

    shear_up = change_tensor_index(shear, "contravariant", metric)
    sigma_2 = np.zeros(metric['tensor'][0][0].shape)

    for i in range(4):
        for j in range(4):
            sigma_2 = sigma_2 + (1 / 2) * shear['tensor'][i][j] * shear_up['tensor'][i][j]

    shear_scalar = sigma_2
    expansion_scalar = theta_trace
    vorticity_scalar = omega_trace

    return expansion_scalar, shear_scalar, vorticity_scalar
