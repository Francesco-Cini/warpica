import numpy as np
from datetime import date as _date

from warp_factory_py.metrics.set_minkowski_three_plus_one import set_minkowski_three_plus_one
from warp_factory_py.units.universal_constants.c import c 
from warp_factory_py.metrics.three_plus_one_builder import three_plus_one_builder

def metric_get_lentz_comoving(
        grid_size: np.ndarray, 
        world_centre: np.ndarray, 
        v: float, 
        scale, 
        grid_scale: np.ndarray | None
        ) -> dict:

    if scale is None:
        scale = np.max(grid_size[1:4] / 7)

    if grid_scale is None:
        grid_scale = np.array([1, 1, 1, 1])

    if grid_size[0] > 1:
        raise Exception("The time grid is greater than 1, only a size of 1 can be used for Lentz")
    
    metric = {}

    metric['params'] = {}

    metric['params']['grid_size'] = grid_size
    metric['params']['world_centre'] = world_centre
    metric['params']['velocity'] = v 

    metric['type'] = "metric"
    metric['name'] = "Lentz Comoving"
    metric['scaling'] = grid_scale
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = _date.today().isoformat()

    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    t = 1 

    for i in range(grid_size[1]):
        for j in range(grid_size[2]):
            for k in range(grid_size[3]):

                x = i * grid_scale[1] - world_centre[1]
                y = j * grid_scale[2] - world_centre[2]

                WFX, WFY = get_warp_factor_by_region(x, y, scale)

                beta[0][t, i, j, k] = v * (1 - WFX)
                beta[1][t, i, j, k] = v * WFY
    
    metric['tensor'] = three_plus_one_builder(alpha, beta, gamma)

    return metric

def get_warp_factor_by_region(x_in, y_in, size_scale):

    x = x_in
    y = np.abs(y_in)
    WFX = 0
    WFY = 0

    if (x >= size_scale and x <= 2 * size_scale) and (x - size_scale >= y):
        WFX = -2
        WFY = 0
    elif (x > size_scale and x <= 2 * size_scale) and (x - size_scale <= y) and (-y + 3 * size_scale >= x):
        WFX = -1
        WFY =  1
    elif (x > 0 and x <= size_scale) and (x + size_scale > y) and (-y + size_scale < x):
        WFX = 0
        WFY =  1
    elif (x > 0 and x <= size_scale) and (x + size_scale <= y) and (-y + 3 * size_scale >= x):
        WFX = -0.5
        WFY =  0.5
    elif (x > -size_scale and x <= 0) and (-x + size_scale < y) and (-y + 3 * size_scale >= -x):
        WFX = 0.5
        WFY = 0.5
    elif (x > -size_scale and x <= 0) and (x + size_scale <= y) and (-y + size_scale >= x):
        WFX = 1
        WFY = 0
    elif (x >= -size_scale and x <= size_scale) and (x + size_scale > y):
        WFX = 1
        WFY = 0
    
    WFY = np.sign(y_in) * WFY 

    return WFX, WFY