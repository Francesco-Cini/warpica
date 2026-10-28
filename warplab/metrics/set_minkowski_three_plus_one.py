import numpy as np


def set_minkowski_three_plus_one(grid_size):
    """Allocate the lapse, shift, and spatial metric for flat spacetime.

    Parameters
    ----------
    grid_size : array_like of int, shape (4,)
        Number of samples along the (t, x, y, z) axes.

    Returns
    -------
    alpha : ndarray
        Lapse field, with shape (nt, nx, ny, nz).
    beta : list of arrays
        Three covariant shift fields.
    gamma : list of lists of arrays
        Symmetric 3 by 3 covariant spatial-metric fields.
    """
    alpha = np.ones(grid_size)

    beta = [np.zeros(grid_size) for _ in range(3)]

    gamma = [[None for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            if i == j:
                gamma[i][j] = np.ones(grid_size)
            else:
                gamma[i][j] = np.zeros(grid_size)

    return alpha, beta, gamma
