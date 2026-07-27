import numpy as np
from datetime import date as _date

from pywarp.metrics.set_minkowski import set_minkowski
from pywarp.units.universal_constants.c import c
from pywarp.metrics.utils.shape_function_alcubierre import shape_function_alcubierre

def metric_get_van_den_broeck(grid_size, world_centre, v, R_1, sigma_1, R_2, sigma_2, A, grid_scale):

    if grid_scale is None:
        grid_scale = np.array([1, 1, 1, 1])

    metric = {}

    metric['params'] = {}

    metric['params']['grid_size'] = grid_size
    metric['params']['world_centre'] = world_centre
    metric['params']['velocity'] = v * ((1 + A) ** 2)
    metric['params']['R_1'] = R_1 
    metric['params']['sigma_1'] = sigma_1
    metric['params']['R_2'] = R_2
    metric['params']['sigma_2'] = sigma_2
    metric['params']['A'] = A 

    metric['type'] = "metric"
    metric['name'] = "Van Den Broeck"
    metric['scaling'] = grid_scale
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = _date.today().isoformat()

    metric['tensor'] = set_minkowski(grid_size)
 
    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):

                x = i * grid_scale[1] - world_centre[1]
                y = j * grid_scale[2] - world_centre[2]
                z = k * grid_scale[3] - world_centre[3]

                for t in range(grid_size[0]):

                    x_s = (t * grid_scale[0] - world_centre[0]) * v * ((1 + A) ** 2) * c()

                    r = np.sqrt(((x - x_s) ** 2) + (y ** 2) + (z ** 2))

                    B = 1 + shape_function_alcubierre(r, R_1, sigma_1) * A 

                    f_s = shape_function_alcubierre(r, R_2, sigma_2) * v 

                    metric['tensor'][1][1][t, i, j, k] = B ** 2
                    metric['tensor'][2][2][t, i, j, k] = B ** 2
                    metric['tensor'][3][3][t, i, j, k] = B ** 2

                    metric['tensor'][0][1][t, i, j, k] = - (B ** 2) * f_s
                    metric['tensor'][1][0][t, i, j, k] = metric['tensor'][0][1][t, i, j, k]

                    metric['tensor'][0][0][t, i, j, k] = - (1 - (B ** 2) * (f_s ** 2))

    return metric
