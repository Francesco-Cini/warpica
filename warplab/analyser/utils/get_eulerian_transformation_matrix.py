import numpy as np


# NumPy returns NaN or inf for a degenerate metric; the checks at the end turn
# that into one clear error, so keep the intermediate warnings out of the way.
@np.errstate(divide="ignore", invalid="ignore")
def get_eulerian_transformation_matrix(g, coords=None):
    """Construct an orthonormal Eulerian basis from a covariant metric.

    Parameters
    ----------
    g : array_like
        Symmetric metric with shape (4, 4) or (nt, nx, ny, nz, 4, 4).
    coords : str or None, optional
        Compatibility argument; the construction assumes Cartesian coordinates.

    Returns
    -------
    M : ndarray
        Basis matrix with the same shape as g; M.T @ g @ M is the
        Minkowski metric at each grid cell.

    Raises
    ------
    ValueError
        If g has an unsupported shape, is not symmetric, or has no Eulerian
        observers (it is singular or its t = const surfaces are not spacelike).
    """
    g = np.asarray(g)

    if g.ndim == 2 and g.shape == (4, 4):
        if not np.allclose(g, g.T):
            raise ValueError("g is not symmetric")

        M = np.zeros((4, 4), dtype=np.result_type(g, float))

        factor_0 = g[3, 3]

        factor_1 = -(g[2, 3] ** 2) + g[2, 2] * factor_0

        factor_2 = (
            2 * g[1, 2] * g[1, 3] * g[2, 3]
            - g[3, 3] * g[1, 2] ** 2
            - g[2, 2] * g[1, 3] ** 2
            + g[1, 1] * factor_1
        )

        factor_3 = (
            -2 * g[3, 3] * g[0, 1] * g[0, 2] * g[1, 2]
            + 2 * g[0, 2] * g[0, 3] * g[1, 2] * g[1, 3]
            + 2 * g[0, 1] * g[0, 2] * g[1, 3] * g[2, 3]
            + 2 * g[0, 1] * g[0, 3] * g[1, 2] * g[2, 3]
            - g[0, 1] ** 2 * g[2, 3] ** 2
            - g[0, 2] ** 2 * g[1, 3] ** 2
            - g[0, 3] ** 2 * g[1, 2] ** 2
            + g[2, 2] * (-2 * g[0, 1] * g[0, 3] * g[1, 3] + g[3, 3] * g[0, 1] ** 2)
            + g[1, 1]
            * (
                -2 * g[0, 2] * g[0, 3] * g[2, 3]
                + g[3, 3] * g[0, 2] ** 2
                + g[2, 2] * g[0, 3] ** 2
            )
            - g[0, 0] * factor_2
        )

        M[0, 0] = np.sqrt(factor_2 / factor_3)

        M[1, 0] = (
            g[0, 1] * g[2, 3] ** 2
            + g[0, 2] * g[1, 2] * g[3, 3]
            - g[0, 2] * g[1, 3] * g[2, 3]
            - g[0, 3] * g[1, 2] * g[2, 3]
            + g[0, 3] * g[1, 3] * g[2, 2]
            - g[0, 1] * g[2, 2] * g[3, 3]
        ) / np.sqrt(factor_2 * factor_3)

        M[2, 0] = (
            g[0, 2] * g[1, 3] ** 2
            - g[0, 3] * g[1, 2] * g[1, 3]
            + g[0, 1] * g[1, 2] * g[3, 3]
            - g[0, 1] * g[1, 3] * g[2, 3]
            - g[0, 2] * g[1, 1] * g[3, 3]
            + g[0, 3] * g[1, 1] * g[2, 3]
        ) / np.sqrt(factor_2 * factor_3)

        M[3, 0] = (
            g[0, 3] * g[1, 2] ** 2
            - g[0, 2] * g[1, 2] * g[1, 3]
            - g[0, 1] * g[1, 2] * g[2, 3]
            + g[0, 1] * g[1, 3] * g[2, 2]
            + g[0, 2] * g[1, 1] * g[2, 3]
            - g[0, 3] * g[1, 1] * g[2, 2]
        ) / np.sqrt(factor_2 * factor_3)

        M[1, 1] = np.sqrt(factor_1 / factor_2)

        M[2, 1] = (g[1, 3] * g[2, 3] - g[1, 2] * g[3, 3]) / np.sqrt(factor_1 * factor_2)

        M[3, 1] = (g[1, 2] * g[2, 3] - g[1, 3] * g[2, 2]) / np.sqrt(factor_1 * factor_2)

        M[2, 2] = np.sqrt(factor_0 / factor_1)

        M[3, 2] = -g[2, 3] / np.sqrt(factor_0 * factor_1)

        M[3, 3] = np.sqrt(1 / factor_0)

    elif g.ndim == 6 and g.shape[-2:] == (4, 4):
        M = np.zeros_like(g, dtype=np.result_type(g, float))

        factor_0 = g[..., 3, 3]

        factor_1 = -(g[..., 2, 3] ** 2) + g[..., 2, 2] * factor_0

        factor_2 = (
            2 * g[..., 1, 2] * g[..., 1, 3] * g[..., 2, 3]
            - g[..., 3, 3] * g[..., 1, 2] ** 2
            - g[..., 2, 2] * g[..., 1, 3] ** 2
            + g[..., 1, 1] * factor_1
        )

        factor_3 = (
            -2 * g[..., 3, 3] * g[..., 0, 1] * g[..., 0, 2] * g[..., 1, 2]
            + 2 * g[..., 0, 2] * g[..., 0, 3] * g[..., 1, 2] * g[..., 1, 3]
            + 2 * g[..., 0, 1] * g[..., 0, 2] * g[..., 1, 3] * g[..., 2, 3]
            + 2 * g[..., 0, 1] * g[..., 0, 3] * g[..., 1, 2] * g[..., 2, 3]
            - g[..., 0, 1] ** 2 * g[..., 2, 3] ** 2
            - g[..., 0, 2] ** 2 * g[..., 1, 3] ** 2
            - g[..., 0, 3] ** 2 * g[..., 1, 2] ** 2
            + g[..., 2, 2]
            * (
                -2 * g[..., 0, 1] * g[..., 0, 3] * g[..., 1, 3]
                + g[..., 3, 3] * g[..., 0, 1] ** 2
            )
            + g[..., 1, 1]
            * (
                -2 * g[..., 0, 2] * g[..., 0, 3] * g[..., 2, 3]
                + g[..., 3, 3] * g[..., 0, 2] ** 2
                + g[..., 2, 2] * g[..., 0, 3] ** 2
            )
            - g[..., 0, 0] * factor_2
        )

        M[..., 0, 0] = np.sqrt(factor_2 / factor_3)

        M[..., 1, 0] = (
            g[..., 0, 1] * g[..., 2, 3] ** 2
            + g[..., 0, 2] * g[..., 1, 2] * g[..., 3, 3]
            - g[..., 0, 2] * g[..., 1, 3] * g[..., 2, 3]
            - g[..., 0, 3] * g[..., 1, 2] * g[..., 2, 3]
            + g[..., 0, 3] * g[..., 1, 3] * g[..., 2, 2]
            - g[..., 0, 1] * g[..., 2, 2] * g[..., 3, 3]
        ) / np.sqrt(factor_2 * factor_3)

        M[..., 2, 0] = (
            g[..., 0, 2] * g[..., 1, 3] ** 2
            - g[..., 0, 3] * g[..., 1, 2] * g[..., 1, 3]
            + g[..., 0, 1] * g[..., 1, 2] * g[..., 3, 3]
            - g[..., 0, 1] * g[..., 1, 3] * g[..., 2, 3]
            - g[..., 0, 2] * g[..., 1, 1] * g[..., 3, 3]
            + g[..., 0, 3] * g[..., 1, 1] * g[..., 2, 3]
        ) / np.sqrt(factor_2 * factor_3)

        M[..., 3, 0] = (
            g[..., 0, 3] * g[..., 1, 2] ** 2
            - g[..., 0, 2] * g[..., 1, 2] * g[..., 1, 3]
            - g[..., 0, 1] * g[..., 1, 2] * g[..., 2, 3]
            + g[..., 0, 1] * g[..., 1, 3] * g[..., 2, 2]
            + g[..., 0, 2] * g[..., 1, 1] * g[..., 2, 3]
            - g[..., 0, 3] * g[..., 1, 1] * g[..., 2, 2]
        ) / np.sqrt(factor_2 * factor_3)

        M[..., 1, 1] = np.sqrt(factor_1 / factor_2)

        M[..., 2, 1] = (
            g[..., 1, 3] * g[..., 2, 3] - g[..., 1, 2] * g[..., 3, 3]
        ) / np.sqrt(factor_1 * factor_2)

        M[..., 3, 1] = (
            g[..., 1, 2] * g[..., 2, 3] - g[..., 1, 3] * g[..., 2, 2]
        ) / np.sqrt(factor_1 * factor_2)

        M[..., 2, 2] = np.sqrt(factor_0 / factor_1)

        M[..., 3, 2] = -g[..., 2, 3] / np.sqrt(factor_0 * factor_1)

        M[..., 3, 3] = np.sqrt(1 / factor_0)

    else:
        raise ValueError(
            f"Unrecognised matrix size {g.shape}; "
            "expected (4, 4) or (nt, nx, ny, nz, 4, 4)"
        )

    if not np.isfinite(M).all():
        raise ValueError(
            "Eulerian transformation is not finite: the metric is singular, or its "
            "t = const surfaces are not spacelike so no Eulerian observers exist"
        )

    return M
