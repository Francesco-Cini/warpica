import numpy as np
import copy

from warp_factory_py.analyser.change_tensor_index import change_tensor_index
from warp_factory_py.solver.verify_tensor import verify_tensor
from warp_factory_py.solver.utils.strcmpi import strcmpi
from warp_factory_py.solver.utils.is_field import is_field
from warp_factory_py.solver.utils.tensor_cell_2_array import tensor_cell_2_array
from warp_factory_py.analyser.utils.get_eulerian_transformation_matrix import get_eulerian_transformation_matrix

def do_frame_transfer(metric, energy_tensor, frame):

    transformed_energy_tensor = copy.deepcopy(energy_tensor)
    transformed_energy_tensor['tensor'] = [[None for _ in range(4)] for _ in range(4)]

    if not verify_tensor(metric, 1):
        raise Exception("Metric is not verified. Please verify metric using verify_tensor(metric).")
    
    if not verify_tensor(energy_tensor, 1):
        raise Exception("Stress-energy is not verified. Please veify Stress-energy tensor using verify_tensor(energy_tensor).")
    
    if strcmpi(frame, "Eulerian") and not (is_field(energy_tensor, 'frame') and strcmpi(energy_tensor['frame'], "Eulerian")):
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric)

        array_energy_tensor = tensor_cell_2_array(energy_tensor)
        array_metric_tensor = tensor_cell_2_array(metric)

        M = get_eulerian_transformation_matrix(array_metric_tensor, metric['coords'])

        transformed_temp_tensor = {}

        transformed_temp_tensor['tensor'] = np.matmul(np.matmul(np.swapaxes(M, -1, -2), array_energy_tensor), M)

        z = energy_tensor['tensor'][0][0].shape

        for i in range(4):
            for j in range(4):
                transformed_energy_tensor['tensor'][i][j] = transformed_temp_tensor['tensor'][..., i, j].reshape(z)

        for i in range(2,4):
            transformed_energy_tensor['tensor'][0][i] = -transformed_energy_tensor['tensor'][0][i]
            transformed_energy_tensor['tensor'][i][0] = -transformed_energy_tensor['tensor'][i][0]

        transformed_energy_tensor['frame'] = "Eulerian"
        transformed_energy_tensor['index'] = "contravariant"

    else:
        raise Warning("Frame not found")

    return transformed_energy_tensor
