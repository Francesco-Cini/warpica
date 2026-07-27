import numpy as np 
from datetime import datetime 

from pywarp.metrics.set_minkowski_three_plus_one import set_minkowski_three_plus_one
from pywarp.metrics.three_plus_one_builder import three_plus_one_builder


def custom_metric(grid_size, world_centre, grid_scaling, alpha_function, beta_function, gamma_function):

    metric = {}

    metric['type'] = "metric"
    metric['name'] = "Custom Metric"
    metric['scaling'] = grid_scaling
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = datetime.now()

    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    for h in range(grid_size[0]):
        for i in range(grid_size[1]):
            for j  in range(grid_size[2]):
                for k in range(grid_size[3]):
                    t = h * grid_scaling[0] - world_centre[0]
                    x = i * grid_scaling[1] - world_centre[1]
                    y = j * grid_scaling[2] - world_centre[2]
                    z = k * grid_scaling[3] - world_centre[3]

                    alpha_field = alpha_function(t, x, y ,z)
                    beta_field = beta_function(t, x, y, z)
                    gamma_field = gamma_function(t, x, y, z)

                    alpha[h, i, j, k] = alpha_field

                    beta[0][h, i, j, k] = beta_field[0]
                    beta[1][h, i, j, k] = beta_field[1]
                    beta[2][h, i, j, k] = beta_field[2]

                    gamma[0][0][h, i, j, k] = gamma_field[0][0]
                    gamma[1][1][h, i, j, k] = gamma_field[1][1]
                    gamma[2][2][h, i, j, k] = gamma_field[2][2]
                    gamma[0][1][h, i, j, k] = gamma_field[0][1]
                    gamma[0][2][h, i, j, k] = gamma_field[0][2]
                    gamma[1][2][h, i, j, k] = gamma_field[1][2]

                    gamma[1][0][h, i, j, k] = gamma[0][1][h, i, j, k]
                    gamma[2][0][h, i, j, k] = gamma[0][2][h, i, j, k]
                    gamma[2][1][h, i, j, k] = gamma[1][2][h, i, j, k]
    
    metric['tensor'] = three_plus_one_builder(alpha, beta, gamma)

    return metric
