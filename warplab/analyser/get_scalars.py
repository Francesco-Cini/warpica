from typing import Any

import numpy as np

from warplab._typing import Array, DiffOrder, TensorDict
from warplab.analyser.change_tensor_index import change_tensor_index
from warplab.analyser.utils.get_trace import get_trace
from warplab.metrics.three_plus_one_decomposer import three_plus_one_decomposer
from warplab.solver.utils.c4_inv import c4_inv
from warplab.solver.utils.cov_div import cov_div
from warplab.solver.utils.verify_grid_size import verify_grid_size


def get_scalars(
    metric: TensorDict,
    diff_order: DiffOrder = "fourth",
) -> tuple[Array, Array, Array]:
    """Compute kinematic scalars for the Eulerian observer field.

    Parameters
    ----------
    metric : dict
        Metric metadata and components indexed as tensor[mu][nu], each
        with shape (nt, nx, ny, nz).
    diff_order : {'second', 'fourth'}, optional
        Finite-difference accuracy order. Default is 'fourth'.

    Returns
    -------
    expansion_scalar : ndarray
        Trace of the projected symmetric velocity gradient.
    shear_scalar : ndarray
        One half of the shear tensor contracted with itself.
    vorticity_scalar : ndarray
        One half of the vorticity tensor contracted with itself.
        All outputs have shape (nt, nx, ny, nz).

    Raises
    ------
    ValueError
        If an axis has two or more samples but too few for the stencil
        (5 for fourth order, 3 for second order). One sample is accepted.
    """
    metric = change_tensor_index(metric, "covariant")

    shape = metric["tensor"][0][0].shape
    verify_grid_size(shape, diff_order)
    array_metric_tensor = np.empty((*shape, 4, 4))

    for i in range(4):
        for j in range(4):
            array_metric_tensor[:, :, :, :, i, j] = metric["tensor"][i][j]

    # 3+1 split of the metric: read off the lapse alpha and shift vector beta^i.
    alpha, _, _, beta_up, _ = three_plus_one_decomposer(metric)

    array_beta = np.empty((*shape, 3))

    for i in range(3):
        array_beta[:, :, :, :, i] = beta_up[i]

    s = shape
    # Eulerian observer 4-velocity in every cell: u^mu = (1, -beta^i) / alpha
    # (u_up) and its lowered form u_mu = g_mu_nu u^nu (u_down).
    u_up = np.zeros((*s, 4))
    u_down = np.zeros((*s, 4))

    for t in range(s[0]):
        for i in range(s[1]):
            for j in range(s[2]):
                for k in range(s[3]):
                    u_up[t, i, j, k, :] = (
                        1
                        / alpha[t, i, j, k]
                        * np.array(
                            [
                                1,
                                -array_beta[t, i, j, k, 0],
                                -array_beta[t, i, j, k, 1],
                                -array_beta[t, i, j, k, 2],
                            ]
                        )
                    )
                    u_down[t, i, j, k, :] = (
                        array_metric_tensor[t, i, j, k, :, :] @ u_up[t, i, j, k, :]
                    )

    u_up_cell: list[Any] = [None for _ in range(4)]
    u_down_cell: list[Any] = [None for _ in range(4)]

    for i in range(4):
        u_up_cell[i] = u_up[:, :, :, :, i]
        u_down_cell[i] = u_down[:, :, :, :, i]

    del_u: list[list[Any]] = [[None for _ in range(4)] for _ in range(4)]

    metric = change_tensor_index(metric, "covariant")

    delta = np.asarray(metric.get("scaling", np.ones(4)))

    inverse_metric = c4_inv(metric["tensor"])
    for i in range(4):
        for j in range(4):
            del_u[i][j] = cov_div(
                metric["tensor"],
                inverse_metric,
                u_up_cell,
                u_down_cell,
                i,
                j,
                delta,
                0,
                diff_order,
            )

    # Project onto the spatial hypersurface orthogonal to the observer velocity.
    P_mix: list[list[Any]] = [[None for _ in range(4)] for _ in range(4)]
    P: list[list[Any]] = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            if i == j:
                k_delta = 1
            else:
                k_delta = 0

            P_mix[i][j] = k_delta + u_up_cell[i] * u_down_cell[j]
            P[i][j] = metric["tensor"][i][j] + u_down_cell[i] * u_down_cell[j]

    theta: TensorDict = {}
    theta["index"] = "covariant"
    theta["type"] = "tensor"
    theta["tensor"] = [[None for _ in range(4)] for _ in range(4)]

    omega: TensorDict = {}
    omega["index"] = "covariant"
    omega["type"] = "tensor"
    omega["tensor"] = [[None for _ in range(4)] for _ in range(4)]

    # Symmetric and antisymmetric projected gradients give expansion and rotation.
    for i in range(4):
        for j in range(4):
            theta["tensor"][i][j] = np.zeros(metric["tensor"][0][0].shape)
            omega["tensor"][i][j] = np.zeros(metric["tensor"][0][0].shape)
            for a in range(4):
                for b in range(4):
                    theta["tensor"][i][j] = theta["tensor"][i][j] + P_mix[a][i] * P_mix[
                        b
                    ][j] * (1 / 2) * (del_u[a][b] + del_u[b][a])
                    omega["tensor"][i][j] = omega["tensor"][i][j] + P_mix[a][i] * P_mix[
                        b
                    ][j] * (1 / 2) * (del_u[a][b] - del_u[b][a])

    # Expansion scalar: the trace of theta, the fractional rate of volume change.
    theta_trace = get_trace(theta, metric)

    omega_up = change_tensor_index(omega, "contravariant", metric)
    # Vorticity scalar: (1/2) omega_mu_nu omega^mu_nu, the squared magnitude
    # of the rotation tensor (a non-negative scalar, not a vector).
    omega_trace = np.zeros(metric["tensor"][0][0].shape)

    for mu in range(4):
        for nu in range(4):
            omega_trace = (
                omega_trace
                + (1 / 2) * omega_up["tensor"][mu][nu] * omega["tensor"][mu][nu]
            )

    # Removing one third of the expansion leaves the spatial, trace-free shear.
    shear: TensorDict = {}
    shear["index"] = "covariant"
    shear["type"] = "tensor"
    shear["tensor"] = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            shear["tensor"][i][j] = (
                theta["tensor"][i][j] - theta_trace * (1 / 3) * P[i][j]
            )

    shear_up = change_tensor_index(shear, "contravariant", metric)
    # Shear scalar: (1/2) sigma_mu_nu sigma^mu_nu, likewise the squared
    # magnitude of the shear tensor.
    sigma_2 = np.zeros(metric["tensor"][0][0].shape)

    for i in range(4):
        for j in range(4):
            sigma_2 = (
                sigma_2 + (1 / 2) * shear["tensor"][i][j] * shear_up["tensor"][i][j]
            )

    shear_scalar = sigma_2
    expansion_scalar = theta_trace
    vorticity_scalar = omega_trace

    return expansion_scalar, shear_scalar, vorticity_scalar
