import numpy as np
from datetime import date as _date

from warp_factory_py.metrics.set_minkowski import set_minkowski
from warp_factory_py.units.universal_constants.c import c 
from warp_factory_py.metrics.utils.shape_function_alcubierre import shape_function_alcubierre 

def metric_get_modified_time(grid_size, world_centre, v, R, sigma, A, grid_scaling):

    if grid_scaling is None:
        grid_scaling = np.array([1, 1, 1, 1])

    metric = {}

    metric['params'] = {}

    metric['params']['grid_size'] = grid_size
    metric['params']['world_centre'] = world_centre
    metric['params']['velocity'] = v 
    metric['params']['R'] = R 
    metric['params']['sigma'] = sigma  
    metric['params']['A'] = A 

    metric['type'] = "metric"
    metric['frame'] = "comoving"
    metric['name'] = "Modified Time"
    metric['scaling'] = grid_scaling
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = _date.today().isoformat()

    metric['tensor'] = set_minkowski(grid_size)

    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):

                x = i * grid_scaling[0] - world_centre[0]
                y = j * grid_scaling[1] - world_centre[1]
                z = k * grid_scaling[2] - world_centre[2]

                for t in range(grid_size[0]):

                    xs = (t * grid_scaling[0] - world_centre[0]) * v * c()

                    r = (((x - xs) ** 2) + (y ** 2) + (z ** 2)) ** (1 / 2)

                    fs = shape_function_alcubierre(r, R, sigma)

                    metric['tensor'][0][1][t, i, j, k] = - v * fs
                    metric['tensor'][1][0][t, i, j, k] = metric['tensor'][0][1][t, i, j, k]

                    metric['tensor'][0][0][t, i, j, k] = - (((1 - fs) + fs / A) ** 2) + ((fs * v) ** 2)

    return metric