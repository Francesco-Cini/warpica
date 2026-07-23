import numpy as np

def set_minkowski_three_plus_one(grid_size):
    """ SETMINKOWSKI: Returns the 3+1 format for flat space

    INPUTS:
    grid_size - World size in [t,x,y,z]

    OUTPUTS:
    alpha - Lapse rate 4D array

    beta - Shift vector, 1x3 cell of 4D arrays
 
    gamma - Spatial terms, 3x3 cell of 4D arrays.

    """
    alpha = np.ones(grid_size)

    beta = [np.zeros(grid_size) for _ in range(3)]

    gamma = [[None for _ in range(3)] for _ in range(3)]

    for i in range(3):
        for j in range(3):
            if i == j:
                gamma[i][j] = np.ones(grid_size)
            else:
                gamma[i][j] = np.zeros(grid_size)

    return alpha, beta, gamma