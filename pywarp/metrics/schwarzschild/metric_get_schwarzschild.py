import numpy as np
from datetime import date as _date

from pywarp.metrics.set_minkowski import set_minkowski

def metric_get_schwarzschild(grid_size, world_centre, r_s, grid_scaling):

    if grid_scaling is None:
        grid_scaling = np.array([1, 1, 1, 1])

    if grid_size[0] > 1:
        raise Exception("The time grid is greater than 1, only a size of 1 can be used for the Schwarzschild solution.")

    metric = {}

    metric['params'] = {}

    metric['params']['grid_size'] = grid_size
    metric['params']['world_centre'] = world_centre
    metric['params']['r_s'] = r_s

    metric['type'] = "metric"
    metric['frame'] = "comoving"
    metric['name'] = "Schwarzschild"
    metric['scaling'] = grid_scaling
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] =  _date.today().isoformat()

    metric['tensor'] = set_minkowski(grid_size)

    epsilon = 0.0000000001

    t = 0

    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):

                x = i * grid_scaling[1] - world_centre[1]
                y = j * grid_scaling[2] - world_centre[2]
                z = k * grid_scaling[3] - world_centre[3]

                r = np.sqrt((x ** 2) + (y ** 2) + (z ** 2)) + epsilon

                metric['tensor'][0][0][t, i, j, k] = - (1 - r_s / r)
                metric['tensor'][1][1][t, i, j, k] = (((x ** 2) / (1 - r_s / r)) + (y ** 2) + (z ** 2)) / (r ** 2)
                metric['tensor'][2][2][t, i, j, k] = ((x ** 2) + ((y ** 2) / (1 - r_s / r)) + (z ** 2)) / (r ** 2)
                metric['tensor'][3][3][t, i, j, k] = ((x ** 2) + (y ** 2) + ((z ** 2) / (1 - r_s / r))) / (r ** 2)
                
                metric['tensor'][1][2][t, i, j, k] = r_s / ((r ** 3) - (r ** 2) * r_s) * x * y
                metric['tensor'][2][1][t, i, j, k] = metric['tensor'][1][2][t, i, j, k]

                metric['tensor'][1][3][t, i, j, k] = r_s / ((r ** 3) - (r ** 2) * r_s) * x * z
                metric['tensor'][3][1][t, i, j, k] = metric['tensor'][1][3][t, i, j, k]

                metric['tensor'][2][3][t, i, j, k] = r_s / ((r ** 3) - (r ** 2) * r_s) * y * z
                metric['tensor'][3][2][t, i, j, k] = metric['tensor'][2][3][t, i, j, k]


    return metric 
