import numpy as np


def get_eulerian_transformation_matrix(g, coords=None):

    g = np.asarray(g)

    if g.ndim == 2:

        assert np.allclose(g, g.T), "g is Not Symmetric"

        M = np.zeros((4, 4), dtype=np.result_type(g, float))

        factor_0 = g[3, 3]

        factor_1 = -g[2, 3]**2 + g[2, 2] * factor_0

        factor_2 = (
            2 * g[1, 2] * g[1, 3] * g[2, 3]
            - g[3, 3] * g[1, 2]**2
            - g[2, 2] * g[1, 3]**2
            + g[1, 1] * factor_1
        )

        factor_3 = (
            -2 * g[3, 3] * g[0, 1] * g[0, 2] * g[1, 2]
            + 2 * g[0, 2] * g[0, 3] * g[1, 2] * g[1, 3]
            + 2 * g[0, 1] * g[0, 2] * g[1, 3] * g[2, 3]
            + 2 * g[0, 1] * g[0, 3] * g[1, 2] * g[2, 3]
            - g[0, 1]**2 * g[2, 3]**2
            - g[0, 2]**2 * g[1, 3]**2
            - g[0, 3]**2 * g[1, 2]**2
            + g[2, 2] * (
                -2 * g[0, 1] * g[0, 3] * g[1, 3]
                + g[3, 3] * g[0, 1]**2
            )
            + g[1, 1] * (
                -2 * g[0, 2] * g[0, 3] * g[2, 3]
                + g[3, 3] * g[0, 2]**2
                + g[2, 2] * g[0, 3]**2
            )
            - g[0, 0] * factor_2
        )

        M[0, 0] = np.sqrt(factor_2 / factor_3)

        M[1, 0] = (
            g[0, 1] * g[2, 3]**2
            + g[0, 2] * g[1, 2] * g[3, 3]
            - g[0, 2] * g[1, 3] * g[2, 3]
            - g[0, 3] * g[1, 2] * g[2, 3]
            + g[0, 3] * g[1, 3] * g[2, 2]
            - g[0, 1] * g[2, 2] * g[3, 3]
        ) / np.sqrt(factor_2 * factor_3)

        M[2, 0] = (
            g[0, 2] * g[1, 3]**2
            - g[0, 3] * g[1, 2] * g[1, 3]
            + g[0, 1] * g[1, 2] * g[3, 3]
            - g[0, 1] * g[1, 3] * g[2, 3]
            - g[0, 2] * g[1, 1] * g[3, 3]
            + g[0, 3] * g[1, 1] * g[2, 3]
        ) / np.sqrt(factor_2 * factor_3)

        M[3, 0] = (
            g[0, 3] * g[1, 2]**2
            - g[0, 2] * g[1, 2] * g[1, 3]
            - g[0, 1] * g[1, 2] * g[2, 3]
            + g[0, 1] * g[1, 3] * g[2, 2]
            + g[0, 2] * g[1, 1] * g[2, 3]
            - g[0, 3] * g[1, 1] * g[2, 2]
        ) / np.sqrt(factor_2 * factor_3)

        M[1, 1] = np.sqrt(factor_1 / factor_2)

        M[2, 1] = (
            g[1, 3] * g[2, 3]
            - g[1, 2] * g[3, 3]
        ) / np.sqrt(factor_1 * factor_2)

        M[3, 1] = (
            g[1, 2] * g[2, 3]
            - g[1, 3] * g[2, 2]
        ) / np.sqrt(factor_1 * factor_2)

        M[2, 2] = np.sqrt(factor_0 / factor_1)

        M[3, 2] = -g[2, 3] / np.sqrt(factor_0 * factor_1)

        M[3, 3] = np.sqrt(1 / factor_0)

    elif g.ndim == 6:

        M = np.zeros_like(g, dtype=np.result_type(g, float))

        factor_0 = g[..., 3, 3]

        factor_1 = -g[..., 2, 3]**2 + g[..., 2, 2] * factor_0

        factor_2 = (
            2 * g[..., 1, 2] * g[..., 1, 3] * g[..., 2, 3]
            - g[..., 3, 3] * g[..., 1, 2]**2
            - g[..., 2, 2] * g[..., 1, 3]**2
            + g[..., 1, 1] * factor_1
        )

        factor_3 = (
            -2 * g[..., 3, 3] * g[..., 0, 1] * g[..., 0, 2] * g[..., 1, 2]
            + 2 * g[..., 0, 2] * g[..., 0, 3] * g[..., 1, 2] * g[..., 1, 3]
            + 2 * g[..., 0, 1] * g[..., 0, 2] * g[..., 1, 3] * g[..., 2, 3]
            + 2 * g[..., 0, 1] * g[..., 0, 3] * g[..., 1, 2] * g[..., 2, 3]
            - g[..., 0, 1]**2 * g[..., 2, 3]**2
            - g[..., 0, 2]**2 * g[..., 1, 3]**2
            - g[..., 0, 3]**2 * g[..., 1, 2]**2
            + g[..., 2, 2] * (
                -2 * g[..., 0, 1] * g[..., 0, 3] * g[..., 1, 3]
                + g[..., 3, 3] * g[..., 0, 1]**2
            )
            + g[..., 1, 1] * (
                -2 * g[..., 0, 2] * g[..., 0, 3] * g[..., 2, 3]
                + g[..., 3, 3] * g[..., 0, 2]**2
                + g[..., 2, 2] * g[..., 0, 3]**2
            )
            - g[..., 0, 0] * factor_2
        )

        M[..., 0, 0] = np.sqrt(factor_2 / factor_3)

        M[..., 1, 0] = (
            g[..., 0, 1] * g[..., 2, 3]**2
            + g[..., 0, 2] * g[..., 1, 2] * g[..., 3, 3]
            - g[..., 0, 2] * g[..., 1, 3] * g[..., 2, 3]
            - g[..., 0, 3] * g[..., 1, 2] * g[..., 2, 3]
            + g[..., 0, 3] * g[..., 1, 3] * g[..., 2, 2]
            - g[..., 0, 1] * g[..., 2, 2] * g[..., 3, 3]
        ) / np.sqrt(factor_2 * factor_3)

        M[..., 2, 0] = (
            g[..., 0, 2] * g[..., 1, 3]**2
            - g[..., 0, 3] * g[..., 1, 2] * g[..., 1, 3]
            + g[..., 0, 1] * g[..., 1, 2] * g[..., 3, 3]
            - g[..., 0, 1] * g[..., 1, 3] * g[..., 2, 3]
            - g[..., 0, 2] * g[..., 1, 1] * g[..., 3, 3]
            + g[..., 0, 3] * g[..., 1, 1] * g[..., 2, 3]
        ) / np.sqrt(factor_2 * factor_3)

        M[..., 3, 0] = (
            g[..., 0, 3] * g[..., 1, 2]**2
            - g[..., 0, 2] * g[..., 1, 2] * g[..., 1, 3]
            - g[..., 0, 1] * g[..., 1, 2] * g[..., 2, 3]
            + g[..., 0, 1] * g[..., 1, 3] * g[..., 2, 2]
            + g[..., 0, 2] * g[..., 1, 1] * g[..., 2, 3]
            - g[..., 0, 3] * g[..., 1, 1] * g[..., 2, 2]
        ) / np.sqrt(factor_2 * factor_3)

        M[..., 1, 1] = np.sqrt(factor_1 / factor_2)

        M[..., 2, 1] = (
            g[..., 1, 3] * g[..., 2, 3]
            - g[..., 1, 2] * g[..., 3, 3]
        ) / np.sqrt(factor_1 * factor_2)

        M[..., 3, 1] = (
            g[..., 1, 2] * g[..., 2, 3]
            - g[..., 1, 3] * g[..., 2, 2]
        ) / np.sqrt(factor_1 * factor_2)

        M[..., 2, 2] = np.sqrt(factor_0 / factor_1)

        M[..., 3, 2] = -g[..., 2, 3] / np.sqrt(factor_0 * factor_1)

        M[..., 3, 3] = np.sqrt(1 / factor_0)

    else:
        raise Exception("Unrecognised matrix size")

    if np.isinf(M).any():
        raise Exception(
            "Eulerian Transformation is Infinite - Numerical Precision Insufficient"
        )

    if not np.isreal(M).all():
        raise Exception(
            "Eulerian Transformation is imaginary - Numerical Precision Insufficient"
        )

    return M