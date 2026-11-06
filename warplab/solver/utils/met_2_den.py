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

    # 1. Inverse metric g^mu_nu, needed to raise indices and in the curvature.
    gu = c4_inv(gl)

    # 2. Ricci tensor R_mu_nu from first and second derivatives of the metric.
    R_munu = ricci_t(gu, gl, delta)

    # 3. Ricci scalar R = g^mu_nu R_mu_nu.
    R = ricci_s(R_munu, gu)

    # 4. Einstein tensor G_mu_nu = R_mu_nu - (1/2) g_mu_nu R (named E here).
    E = ein_t(R_munu, R, gl)

    # 5. Einstein's equation G = (8 pi G / c^4) T, solved for T and with both
    # indices raised: contravariant stress-energy T^mu_nu in SI units.
    energy_density = ein_e(E, gu)

    return energy_density
