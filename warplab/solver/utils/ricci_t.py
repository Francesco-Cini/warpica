from array_api_compat import array_namespace

from warplab.solver.utils.ricci_contractions import raised_derivatives
from warplab.solver.utils.take_finite_difference_1 import take_finite_difference_1
from warplab.solver.utils.take_finite_difference_2 import take_finite_difference_2
from warplab.units.universal_constants.c import c


def ricci_t(gu, gl, delta):
    """Compute covariant Ricci components with fourth-order differences.

    Parameters
    ----------
    gu : sequence of sequences of arrays
        Contravariant metric components on the same grid.
    gl : sequence of sequences of arrays
        Covariant metric components on the (t, x, y, z) grid.
    delta : array_like, shape (4,)
        Grid spacings in (t, x, y, z) order.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.

    Notes
    -----
    Time spacing is in seconds; time derivatives are converted to the ct basis.
    """
    xp = array_namespace(gl[0][0])

    R_munu = [[None for _ in range(4)] for _ in range(4)]

    diff_1_gl = [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]
    diff_2_gl = [
        [[[None for _ in range(4)] for _ in range(4)] for _ in range(4)]
        for _ in range(4)
    ]

    # Cache only independent components; metric and mixed-partial symmetry
    # supply the remaining entries without additional finite differences.
    for i in range(4):
        for j in range(i, 4):
            for k in range(4):
                diff_1_gl[i][j][k] = take_finite_difference_1(gl[i][j], k, delta)
                # The stored time coordinate is seconds; curvature uses ct.
                if k == 0:
                    diff_1_gl[i][j][k] = (1 / c()) * diff_1_gl[i][j][k]

                for n in range(k, 4):
                    diff_2_gl[i][j][k][n] = take_finite_difference_2(
                        gl[i][j], k, n, delta
                    )

                    if (n == 0 and k != 0) or (n != 0 and k == 0):
                        diff_2_gl[i][j][k][n] = (1 / c()) * diff_2_gl[i][j][k][n]

                    if k == 0 and n == 0:
                        diff_2_gl[i][j][k][n] = diff_2_gl[i][j][k][n] / c() ** 2

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

    raised, skew, connection = raised_derivatives(gu, diff_1_gl)

    for i in range(4):
        for j in range(i, 4):
            R_munu_temp = xp.zeros_like(gl[0][0])

            for a in range(4):
                for b in range(4):
                    R_munu_temp_2 = xp.zeros_like(gl[0][0])

                    R_munu_temp_2 = R_munu_temp_2 - (
                        diff_2_gl[i][j][a][b]
                        + diff_2_gl[a][b][i][j]
                        - diff_2_gl[i][b][j][a]
                        - diff_2_gl[j][b][i][a]
                    )

                    for r in range(4):
                        R_munu_temp_3 = raised[j][b][r]
                        R_munu_temp_4 = skew[j][b][r]
                        R_munu_temp_5 = -connection[a][b][r]

                        R_munu_temp_2 = (
                            R_munu_temp_2
                            + R_munu_temp_4 * diff_1_gl[i][r][a]
                            + (1 / 2)
                            * (
                                R_munu_temp_3 * diff_1_gl[a][r][i]
                                + R_munu_temp_5
                                * (
                                    diff_1_gl[j][r][i]
                                    + diff_1_gl[i][r][j]
                                    - diff_1_gl[j][i][r]
                                )
                            )
                        )

                    R_munu_temp = R_munu_temp + gu[a][b] * R_munu_temp_2

            R_munu[i][j] = (1 / 2) * R_munu_temp

    R_munu[1][0] = R_munu[0][1]
    R_munu[2][0] = R_munu[0][2]
    R_munu[2][1] = R_munu[1][2]
    R_munu[3][0] = R_munu[0][3]
    R_munu[3][1] = R_munu[1][3]
    R_munu[3][2] = R_munu[2][3]

    return R_munu
