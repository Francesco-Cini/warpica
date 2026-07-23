import numpy as np

def set_minkowski(grid_size):

    metric = [[None for _ in range(4)] for _ in range(4)]

    metric[0][0] = -np.ones(grid_size)

    metric[1][1] = np.ones(grid_size)
    metric[2][2] = np.ones(grid_size)
    metric[3][3] = np.ones(grid_size)

    metric[0][1] = np.zeros(grid_size)
    metric[1][0] = np.zeros(grid_size)
    metric[0][2] = np.zeros(grid_size)
    metric[2][0] = np.zeros(grid_size)
    metric[1][2] = np.zeros(grid_size)
    metric[2][1] = np.zeros(grid_size)
    metric[1][3] = np.zeros(grid_size)
    metric[3][1] = np.zeros(grid_size)
    metric[2][3] = np.zeros(grid_size)
    metric[3][2] = np.zeros(grid_size)
    metric[0][3] = np.zeros(grid_size)
    metric[3][0] = np.zeros(grid_size)

    return metric