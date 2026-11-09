from math import pi

from warpica.units.universal_constants.c import c
from warpica.units.universal_constants.G import G


def ein_e(E, gu):
    """Convert the Einstein tensor to contravariant stress-energy.

    Parameters
    ----------
    E : sequence of sequences of arrays
        Covariant Einstein tensor components.
    gu : sequence of sequences of arrays
        Contravariant metric components on the same grid.

    Returns
    -------
    tensor : list of lists of arrays
        A 4 by 4 component table; each component has the input grid shape.
    """
    # Einstein's equation G_mu_nu = (8 pi G / c^4) T_mu_nu, solved for T:
    # scale each covariant component by c^4 / (8 pi G).
    en_den_ = [[None for _ in range(4)] for _ in range(4)]

    for mu in range(4):
        for nu in range(4):
            en_den_[mu][nu] = (c() ** 4) / (8 * pi * G()) * E[mu][nu]

    # Raise both indices: T^mu_nu = g^alpha_mu g^beta_nu T_alpha_beta,
    # summing over alpha and beta.
    en_den = [[None for _ in range(4)] for _ in range(4)]

    for mu in range(4):
        for nu in range(4):
            en_den[mu][nu] = 0

            for alpha in range(4):
                for beta in range(4):
                    en_den[mu][nu] = (
                        en_den[mu][nu]
                        + en_den_[alpha][beta] * gu[alpha][mu] * gu[beta][nu]
                    )

    return en_den
