import numpy as np

from warplab.solver.utils.c4_inv import c4_inv
from warplab.solver.utils.ein_e import ein_e
from warplab.solver.utils.ein_t import ein_t
from warplab.solver.utils.ricci_s import ricci_s
from warplab.solver.utils.ricci_t import ricci_t


def met_2_den(gl, delta):
    """Compute contravariant stress-energy from covariant metric fields.

    Parameters
    ----------
    gl : sequence of sequences of arrays
        Covariant metric components on the (t, x, y, z) grid.
    delta : array_like, shape (4,), optional
        Grid spacings in (t, x, y, z) order; None selects unit spacing.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.
    """
    if delta is None:
        delta = np.array([1, 1, 1, 1])

    gu = c4_inv(gl)

    R_munu = ricci_t(gu, gl, delta)

    R = ricci_s(R_munu, gu)

    E = ein_t(R_munu, R, gl)

    energy_density = ein_e(E, gu)

    return energy_density
