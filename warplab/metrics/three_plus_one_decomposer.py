from typing import Any

import numpy as np

from warplab._typing import Array, TensorDict
from warplab.solver.utils.c3_inv import c3_inv


def three_plus_one_decomposer(
    metric: TensorDict,
) -> tuple[Array, list[Array], list[list[Array]], list[Array], list[list[Array]]]:
    """Decompose a spacetime metric into lapse, shift, and spatial fields.

    Parameters
    ----------
    metric : dict
        Metric metadata and components indexed as tensor[mu][nu], each
        with shape (nt, nx, ny, nz).

    Returns
    -------
    alpha : ndarray
        Lapse field, with shape (nt, nx, ny, nz).
    beta_down : list of arrays
        Three covariant shift components.
    gamma_down : list of lists of arrays
        Covariant spatial metric.
    beta_up : list of arrays
        Three contravariant shift components.
    gamma_up : list of lists of arrays
        Inverse spatial metric.
    """
    # Imported here because analyser imports metrics; a module-level import is circular.
    from warplab.analyser.change_tensor_index import change_tensor_index

    metric = change_tensor_index(metric, "covariant")

    beta_down = [metric["tensor"][0][1], metric["tensor"][0][2], metric["tensor"][0][3]]

    gamma_down = [
        [metric["tensor"][1][1], metric["tensor"][1][2], metric["tensor"][1][3]],
        [metric["tensor"][2][1], metric["tensor"][2][2], metric["tensor"][2][3]],
        [metric["tensor"][3][1], metric["tensor"][3][2], metric["tensor"][3][3]],
    ]

    gamma_up = c3_inv(gamma_down)

    s = metric["tensor"][0][0].shape

    beta_up: list[Any] = [None for _ in range(3)]
    for i in range(3):
        beta_up[i] = np.zeros(s)
        for j in range(3):
            beta_up[i] = beta_up[i] + gamma_up[i][j] * beta_down[j]

    alpha = np.sqrt(
        beta_up[0] * beta_down[0]
        + beta_up[1] * beta_down[1]
        + beta_up[2] * beta_down[2]
        - metric["tensor"][0][0]
    )

    return alpha, beta_down, gamma_down, beta_up, gamma_up
