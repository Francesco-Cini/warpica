import numpy as np
from datetime import date as _date
from warp_factory_py.metrics.set_minkowski_three_plus_one import set_minkowski_three_plus_one
from warp_factory_py.metrics.utils.shape_function_alcubierre import shape_function_alcubierre 
from warp_factory_py.units.universal_constants.c import c 
from warp_factory_py.metrics.three_plus_one_builder import three_plus_one_builder

def metric_get_alcubierre(grid_size,world_centre,v,R,sigma,grid_scale = None):

    """METRICGET_ALCUBIERRE: Builds the Alcubierre metric

    INPUTS: 
    grid_size - 1x4 array. world size in [t, x, y, z], double type.
    
    worldCenter - 1x4 array. world center location in [t, x, y, z], double type.

    v - speed of the warp drive in factors of c, along the x direction, double type.

    R - radius of the warp bubble, double type.

    sigma - thickness parameter of the bubble, double type.

    grid_scale - scaling of the grid in [t, x, y, z]. double type.

    OUTPUTS: 
    metric - metric struct object. """

    # Handle default input arguments

    if grid_scale is None:
        grid_scale = np.array([1,1,1,1])

    # Assign parameters to metric structure

    metric = {}

    metric['params'] = {}

    metric['params']['grid_size'] = grid_size
    metric['params']['world_centre'] = world_centre
    metric['params']['velocity'] = v
    metric['params']['R'] = R
    metric['params']['sigma'] = sigma

    # Assign quantities to metric structure

    metric['type'] = "metric"
    metric['name'] = 'Alcubierre'
    metric['scaling'] = grid_scale
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = _date.today().isoformat()

    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    Nt, Nx, Ny, Nz = grid_size
    dt, dx, dy, dz = grid_scale

    for i in range(Nx):
        for j in range (Ny):
            for k in range(Nz):

                x = (i + 1) * dx - world_centre[1]
                y = (j + 1) * dy - world_centre[2]
                z = (k + 1) * dz - world_centre[3]

                for t in range(Nt):

                    # Determine the x offset of the center of the bubble, centered in time
                    xs = ((t + 1) * dt - world_centre[0]) * v * c()

                    # Find the radius from the center of the bubble
                    r = ((x - xs) ** 2 + y ** 2 + z ** 2) ** 0.5

                    # Find shape function at this point in r
                    fs = shape_function_alcubierre(r, R, sigma)

                    # Add Alcubierre modification to shift vector along x
                    beta[0][t, i, j, k] = -v * fs

    metric['tensor'] = three_plus_one_builder(alpha, beta, gamma)
    return metric
