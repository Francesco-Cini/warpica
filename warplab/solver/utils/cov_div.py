from warplab.solver.utils.get_christoffel_sym import get_christoffel_sym
from warplab.solver.utils.second_order.take_finite_difference_1_2 import (
    take_finite_difference_1_2,
)
from warplab.solver.utils.take_finite_difference_1 import take_finite_difference_1
from warplab.units.universal_constants.c import c


def cov_div(
    g_l, g_u, vec_u, vec_d, idx_div, idx_vec, delta, stair_sel, diff_order="fourth"
):
    """Compute one component of a vector covariant derivative.

    Parameters
    ----------
    g_l : sequence of sequences of arrays
        Covariant metric components on the (t, x, y, z) grid.
    g_u : sequence of sequences of arrays
        Contravariant metric components on the same grid.
    vec_u, vec_d : sequence of arrays
        Contravariant and covariant components of the same four-vector.
    idx_div : int
        Zero-based differentiation axis.
    idx_vec : int
        Zero-based vector component.
    delta : array_like, shape (4,)
        Grid spacings in (t, x, y, z) order.
    stair_sel : {0, 1}
        Select covariant (0) or contravariant (1) vector components.
    diff_order : {'second', 'fourth'}, optional
        Finite-difference accuracy order. Default is 'fourth'.

    Returns
    -------
    result : array
        Selected covariant derivative on the grid.

    Notes
    -----
    Time spacings are in seconds; time derivatives are converted to ct.
    """
    diff_order = str(diff_order).lower()
    if diff_order not in ("second", "fourth"):
        raise ValueError("diff_order must be second or fourth")
    derivative = (
        take_finite_difference_1_2
        if diff_order == "second"
        else take_finite_difference_1
    )

    # Partial derivative along one axis. Time is stored in seconds but the curvature
    # uses ct, so the time derivative is divided by c.
    def partial(field, axis):
        """Differentiate `field` along `axis`, with time in the ct basis."""
        value = derivative(field, axis, delta)
        return value / c() if axis == 0 else value

    # diff[i][j][k] = derivative of g_ij along axis k, for the Christoffel symbols.
    diff = [
        [[partial(g_l[i][j], k) for k in range(4)] for j in range(4)] for i in range(4)
    ]
    # Covariant derivative along axis idx_div of component idx_vec:
    #   covariant vector:     d v_k  - Gamma^i_kd v_i
    #   contravariant vector: d v^k  + Gamma^k_di v^i,  summed over i.
    if stair_sel == 0:
        result = partial(vec_d[idx_vec], idx_div)
        for i in range(4):
            result = (
                result - get_christoffel_sym(g_u, diff, i, idx_vec, idx_div) * vec_d[i]
            )
    elif stair_sel == 1:
        result = partial(vec_u[idx_vec], idx_div)
        for i in range(4):
            result = (
                result + get_christoffel_sym(g_u, diff, idx_vec, idx_div, i) * vec_u[i]
            )
    else:
        raise ValueError("Invalid variance selected")
    return result
