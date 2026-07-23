import numpy as np
from datetime import date as _date

from warp_factory_py.metrics.set_minkowski_three_plus_one import set_minkowski_three_plus_one
from warp_factory_py.metrics.utils.shape_function_alcubierre import shape_function_alcubierre 
from warp_factory_py.units.universal_constants.c import c 
from warp_factory_py.metrics.three_plus_one_builder import three_plus_one_builder

def metric_get_alcubierre_comoving(
        grid_size: np.ndarray,
        world_centre: np.ndarray, 
        v: float, 
        R: float, 
        sigma: float, 
        grid_scale: np.ndarray | None
        ) -> dict:
    """
    Parameters
    ----------

    Returns
    -------
    """

    if grid_scale is None:
        grid_scale = np.array([1, 1, 1, 1])

    if grid_size[0] > 1:
        raise Exception("The time grid is greater than 1, only a size of 1 can be used in the comoving")
    
    metric = {}

    metric['params'] = {}

    metric['params']['grid_size'] = grid_size
    metric['params']['world_centre'] = world_centre
    metric['params']['velocity'] = v 
    metric['params']['R'] = R 
    metric['params']['sigma'] = sigma 

    metric['type'] = "metric"
    metric['name'] = "Alcubierre Comoving"
    metric['scaling'] = grid_scale
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['data'] = _date.today().isoformat()

    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)


    t = 1
    for i in range(grid_size[1]):
        for j in range (grid_size[2]):
            for k in range (grid_size[3]):
                
                x = i * grid_scale[1] - world_centre[1]
                y = j * grid_scale[2] - world_centre[2]
                z = k * grid_scale[3] - world_centre[3]

                r = ((x ** 2) + (y ** 2) + (z ** 2)) ** (1 / 2)

                fs = shape_function_alcubierre(r, R, sigma)

                beta[0][t, i, j, k] = v * (1 - fs)
    
    metric['tensor'] = three_plus_one_builder(alpha, beta, gamma)

    return metric