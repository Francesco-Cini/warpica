import numpy as np

from warp_factory_py.units.universal_constants.c import c
from warp_factory_py.units.universal_constants.G import G

def ein_e(E, gu):

    en_den_ = [[None for _ in range(4)] for _ in range(4)]

    for mu in range(4):
        for nu in range(4):

            en_den_[mu][nu] = (c() ** 4) / (8 * np.pi * G()) * E[mu][nu]
    
    en_den = [[None for _ in range(4)] for _ in range(4)]

    for mu in range(4):
        for nu in range(4):

            en_den[mu][nu] = 0
            
            for alpha in range(4):
                for beta in range(4):
                    en_den[mu][nu] = en_den[mu][nu] + en_den_[alpha][beta] * gu[alpha][mu] * gu[beta][nu]

    return en_den 