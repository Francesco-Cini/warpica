from collections.abc import Sequence
from typing import Any

import numpy as np

from warplab._typing import Array
from warplab.solver.utils.c3_inv import c3_inv


def three_plus_one_builder(
    alpha: Array,
    beta: Sequence[Array],
    gamma: Sequence[Sequence[Array]],
) -> list[list[Array]]:
    """Assemble a covariant spacetime metric from its 3+1 fields.

    Parameters
    ----------
    alpha : ndarray
        Lapse field, with shape (nt, nx, ny, nz).
    beta : list of arrays
        Three covariant shift fields.
    gamma : list of lists of arrays
        Symmetric 3 by 3 covariant spatial-metric fields.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.

    Notes
    -----
    The input shift is covariant. Its index is raised with the inverse
    spatial metric when computing the time-time component.
    """
    gamma_up = c3_inv(gamma)

    s = alpha.shape

    # Raise the covariant shift before forming beta_i * beta^i.
    beta_up: list[Any] = [None for _ in range(3)]

    for i in range(3):
        beta_up[i] = np.zeros(s)
        for j in range(3):
            beta_up[i] = beta_up[i] + gamma_up[i][j] * beta[j]

    metric_tensor: list[list[Any]] = [[None for _ in range(4)] for _ in range(4)]

    # Assemble the 3+1 form of the metric:
    #   g_00 = -alpha^2 + beta_i beta^i,   g_0i = beta_i,   g_ij = gamma_ij,
    # where beta is the covariant shift (indices down) and gamma the spatial metric.
    metric_tensor[0][0] = -(alpha**2)
    for i in range(3):
        metric_tensor[0][0] += beta_up[i] * beta[i]

    for i in range(1, 4):
        metric_tensor[0][i] = beta[i - 1]
        metric_tensor[i][0] = metric_tensor[0][i]

    for i in range(1, 4):
        for j in range(1, 4):
            metric_tensor[i][j] = gamma[i - 1][j - 1]

    return metric_tensor
