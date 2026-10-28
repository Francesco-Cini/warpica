import numpy as np

from warplab.solver.utils.second_order.c4_inv_2 import c4_inv_2
from warplab.solver.utils.second_order.ein_e_2 import ein_e_2
from warplab.solver.utils.second_order.ein_t_2 import ein_t_2
from warplab.solver.utils.second_order.ricci_s_2 import ricci_s_2
from warplab.solver.utils.second_order.ricci_t_2 import ricci_t_2


def met_2_den_2(
    metric_tensor,
    delta=None,
    units=None,
):
    """Compute contravariant stress-energy from covariant metric fields.

    Parameters
    ----------
    metric_tensor : sequence of sequences of arrays
        Covariant metric components on the (t, x, y, z) grid.
    delta : array_like, shape (4,), optional
        Grid spacings in (t, x, y, z) order; None selects unit spacing.
    units : array_like, shape (3,), optional
        Length, mass, and time unit factors; None selects SI units.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.
    """
    if delta is None:
        delta = np.array([1, 1, 1, 1])

    if units is None:
        units = np.array([1, 1, 1])

    g_l = metric_tensor
    g_u = c4_inv_2(g_l)

    R_munu = ricci_t_2(g_u, g_l, delta)

    R = ricci_s_2(R_munu, g_u)

    E = ein_t_2(R_munu, R, g_l)

    energy_density = ein_e_2(E, g_u, units)

    return energy_density
