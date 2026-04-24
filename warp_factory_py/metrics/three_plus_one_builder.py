import numpy as np
from warp_factory_py.solver.utils.c3_inv import c3_inv

def three_plus_one_builder(alpha, beta, gamma):

    ## Three Plus One Builder: Builds the metric given input 3+1 components of alpha, beta, and gamma

    # INPUTS:
    #alpha - (TxXxYxZ) lapse rate map across spacetime
    #beta - {3}x(TxXxYxZ) (covariant assumed) shift vector map across spacetime
    #gamma - {3x3}x(TxXxYxZ) (covariant assumed) spatial term map map across spacetime

    ##

    # Set up Spacial Components  

    gamma_up = c3_inv(gamma)

    # Find grid size

    s = alpha.shape

    # Calculate beta_i

    beta_up = [None for _ in range(3)]

    for i in range(3):
        beta_up[i] = np.zeros(s)
        for j in range(3):
            beta_up[i] = beta_up[i] + gamma_up[i][j] * beta[j]

    # Create time-time component

    metric_tensor = [[None for _ in range(4)] for _ in range(4)]

    metric_tensor[0][0] = -alpha ** 2
    for i in range(3):
        metric_tensor[0][0] += beta_up[i] * beta[i]

    # Create time-space components

    for i in range(1,4):
        metric_tensor[0][i] = beta[i-1]
        metric_tensor[i][0] = metric_tensor[0][i]

    # Create space-space components

    for i in range(1,4):
        for j in range(1,4):
            metric_tensor[i][j] = gamma[i-1][j-1]

    return metric_tensor