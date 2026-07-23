import numpy as np

from warp_factory_py.solver.utils.strcmpi import strcmpi
from warp_factory_py.solver.utils.trilinear_interp import trilinear_interp

def get_momentum_flow_lines(energy_tensor, start_points, step_size, max_steps, scale_factor):

    if not strcmpi(energy_tensor['index'], "contravariant"):
        raise Exception("Energy tensor for momentum flowlines should be contravariant.")
    
    x_mom = np.squeeze(energy_tensor['tensor'][0][1]) * scale_factor
    y_mom = np.squeeze(energy_tensor['tensor'][0][2]) * scale_factor
    z_mom = np.squeeze(energy_tensor['tensor'][0][3]) * scale_factor

    starting_points_x = np.ravel(start_points[0])
    starting_points_y = np.ravel(start_points[1])
    starting_points_z = np.ravel(start_points[2])

    paths = [None for _ in range(max(starting_points_x.shape))]

    for j in range(max(starting_points_x.shape)):
        pos = np.zeros((max_steps + 1, 3))

        pos[0, :] = np.array([starting_points_x[j], starting_points_y[j], starting_points_z[j]])

        for i in range(max_steps):
            if (
                np.sum(np.isnan(pos[i, :])) > 0
                or (np.floor(pos[i, 0]) <= 1 or np.ceil(pos[i, 0]) >= x_mom.shape[0])
                or (np.floor(pos[i, 1]) <= 1 or np.ceil(pos[i, 1]) >= x_mom.shape[1])
                or (np.floor(pos[i, 2]) <= 1 or np.ceil(pos[i, 2]) >= x_mom.shape[2])
            ):
                break 

            x_momentum = trilinear_interp(x_mom, pos[i, :])
            y_momentum = trilinear_interp(y_mom, pos[i, :])
            z_momentum = trilinear_interp(z_mom, pos[i, :])
            
            pos[i+1, 0] = pos[i, 0] + x_momentum * step_size
            pos[i+1, 1] = pos[i, 1] + y_momentum * step_size
            pos[i+1, 2] = pos[i, 2] + z_momentum * step_size

        paths[j] = pos[0:i, :]

    return paths
