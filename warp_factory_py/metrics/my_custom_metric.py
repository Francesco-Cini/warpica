import numpy as np 
from datetime import datetime 

from set_minkowski_three_plus_one import set_minkowski_three_plus_one
from three_plus_one_builder import three_plus_one_builder


def my_custom_metric(grid_size, world_centre, grid_scaling, alpha_function, beta_function, gamma_function):

    metric = {}

    metric['type'] = "metric"
    metric['name'] = "Custom Metric"
    metric['scaling'] = grid_scaling
    metric['coords'] = "cartesian"
    metric['index'] = "covariant"
    metric['date'] = datetime.now()

    t = 1
    x = 1
    y =1
    z = 1

    alpha, beta, gamma = set_minkowski_three_plus_one(grid_size)

    for h in range(t):
        for i in range(x):
            for j  in range(y):
                for k in range(z):
                    t = h * grid_scaling[0] - world_centre[0]
                    x = i 
                    y = j
                    z = k 

                    alpha(h, i, j, k) = alpha_function

                    beta[0](h, i, j, k) = beta_function[0]
                    beta[1](h, i, j, k) = beta_function[1]
                    beta[2](h, i, j, k) = beta_function[2]

                    gamma[0][0](h, i, j, k) = gamma_function[0][0]
                    gamma[1][1](h, i, j, k) = gamma_function[0][0]
                    gamma[2][2](h, i, j, k) = gamma_function[0][0]
                    gamma[0][1](h, i, j, k) = gamma_function[0][0]
                    gamma[0][2](h, i, j, k) = gamma_function[0][0]
                    gamma[1][2](h, i, j, k) = gamma_function[0][0]

                    gamma[0][0](h, i, j, k) = gamma[0][0](h, i, j, k)
                    gamma[0][0](h, i, j, k) = gamma[0][0](h, i, j, k)
                    gamma[0][0](h, i, j, k) = gamma[0][0](h, i, j, k)
    
    metric['tensor'] = three_plus_one_builder(alpha, beta, gamma)

    return metric