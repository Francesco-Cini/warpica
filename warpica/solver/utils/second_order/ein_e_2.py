import numpy as np

from warpica.units.universal_constants.c import c
from warpica.units.universal_constants.G import G


def ein_e_2(E, g_u, units):
    """Convert the Einstein tensor to contravariant stress-energy.

    Parameters
    ----------
    E : sequence of sequences of arrays
        Covariant Einstein tensor components.
    g_u : sequence of sequences of arrays
        Contravariant metric components on the same grid.
    units : array_like, shape (3,)
        Length, mass, and time unit factors; use ones for SI units.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.

    Notes
    -----
    G and c come from `warpica.units`, as in the fourth-order path, and are
    scaled by the unit factors.
    """
    # G and c in the requested units. With the SI default units = [1, 1, 1] these are
    # exactly the values the fourth-order path uses.
    grav = G() * units[2] ** 2 * units[1] / units[0] ** 3
    light = c() * units[2] / units[0]

    en_den_ = [[None for _ in range(4)] for _ in range(4)]

    for mu in range(4):
        for nu in range(4):
            en_den_[mu][nu] = light**4 / (8 * np.pi * grav) * E[mu][nu]

    en_den = [[None for _ in range(4)] for _ in range(4)]

    for mu in range(4):
        for nu in range(4):
            en_den[mu][nu] = 0

            for alpha in range(4):
                for beta in range(4):
                    en_den[mu][nu] = (
                        en_den[mu][nu]
                        + en_den_[alpha][beta] * g_u[alpha][mu] * g_u[beta][nu]
                    )

    return en_den
