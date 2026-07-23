import numpy as np
from datetime import date as _date

def metric_get_warp_shell_comoving(grid_size, world_centre, m, R_1, R_2, R_buff, sigma, smooth_factor, v_warp, do_warp, grid_scaling):

    if R_buff is None:
        R_buff = 0

    if sigma is None:
        sigma = 0

    if smooth_factor is None:
        smooth_factor = 0

    if v_warp is None:
        v_warp = 0

    if do_warp is None:
        do_warp = 0

    if grid_scaling is None:
        grid_scaling = np.array([1, 1, 1, 1])

    metric = {}

    metric['type'] = "metric"
    metric['name'] = "Comoving Warp Shell"
    metric['scaling'] = grid_scaling
    metric['coords'] = "cartesian" 
    metric['index'] = "covariant" 
    metric['date'] = _date.today().isoformat()

    world_size = np.sqrt(((grid_size[1] * grid_scaling[1] - world_centre[1]) ** 2) + ((grid_size[2] * grid_scaling[2] - world_centre[2]) ** 2) + ((grid_size[3] * grid_scaling[3] - world_centre[3]) ** 2))
    r_sample_res = 10e+5
    r_sample = np.linspace(0, world_size * 1.2, r_sample_res)

    rho = np.zeros(1, r_sample.size) + (m / ((4 / 3) * np.pi * ((R_2 ** 3) - (R_1 ** 3)))) * ((r_sample > R_1) & (r_sample < R_2))

    metric['params'] = {}

    metric['params']['rho'] = rho 

    max_R = np.argmin(np.diff(rho > 0))
    max_R = r_sample(max_R)

    return metric