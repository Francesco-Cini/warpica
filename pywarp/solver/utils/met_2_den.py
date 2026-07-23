import numpy as np

from warp_factory_py.solver.utils.c4_inv import c4_inv
from warp_factory_py.solver.utils.ricci_t import ricci_t
from warp_factory_py.solver.utils.ricci_s import ricci_s
from warp_factory_py.solver.utils.ein_t import ein_t
from warp_factory_py.solver.utils.ein_e import ein_e

def met_2_den(gl, delta):
    
    if delta is None:
        delta = np.array([1, 1, 1, 1])
    
    gu = c4_inv(gl)

    R_munu = ricci_t(gu, gl, delta)

    R = ricci_s(R_munu, gu)

    E = ein_t(R_munu, R, gl)

    energy_density = ein_e(E, gu)

    return energy_density