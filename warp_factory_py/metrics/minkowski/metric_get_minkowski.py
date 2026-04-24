import numpy as np 
from datetime import datetime 

def metric_get_minkowski(grid_size, grid_scaling = None):

    if grid_scaling is None:
        grid_scaling = np.array([1, 1, 1, 1])

    metric = {}

    metric['type'] = "metric"
    metric['name'] = "Minkowski"
    metric['scaling'] = grid_scaling
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = datetime.now()

    metric['tensor'] = np.zeros((4, 4, *grid_size))

    metric['tensor'][0][0] = - np.ones(grid_size)

    metric['tensor'][1][1] = np.ones(grid_size)
    metric['tensor'][2][2] = np.ones(grid_size)
    metric['tensor'][3][3] = np.ones(grid_size)

    metric['tensor'][0][1] = np.zeros(grid_size)
    metric['tensor'][1][0] = np.zeros(grid_size)
    metric['tensor'][0][2] = np.zeros(grid_size)
    metric['tensor'][2][0] = np.zeros(grid_size)
    metric['tensor'][1][2] = np.zeros(grid_size)
    metric['tensor'][2][1] = np.zeros(grid_size)
    metric['tensor'][1][3] = np.zeros(grid_size)
    metric['tensor'][3][1] = np.zeros(grid_size)
    metric['tensor'][2][3] = np.zeros(grid_size)
    metric['tensor'][3][2] = np.zeros(grid_size)
    metric['tensor'][0][3] = np.zeros(grid_size)
    metric['tensor'][3][0] = np.zeros(grid_size)

    return metric 