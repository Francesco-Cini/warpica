import numpy as np

from warp_factory_py.solver.utils.strcmpi import strcmpi
from warp_factory_py.solver.verify_tensor import verify_tensor
from warp_factory_py.analyser.do_frame_transfer import do_frame_transfer
from warp_factory_py.analyser.utils.generate_uniform_field import generate_uniform_field
from warp_factory_py.analyser.change_tensor_index import change_tensor_index
from warp_factory_py.metrics.minkowski.metric_get_minkowski import metric_get_minkowski
from warp_factory_py.analyser.utils.get_inner_product import get_inner_product
from warp_factory_py.analyser.utils.get_trace import get_trace

def get_energy_conditions(energy_tensor, metric, condition, num_angular_vec, num_time_vec, return_vec):

    # Handle default input arguments
    if num_angular_vec is None:
        num_angular_vec = 100
    
    if num_time_vec is None: 
        num_time_vec = 10

    if return_vec is None:
        return_vec = 0


    # Check if correct corrections input
    if not (strcmpi(condition, "Null") or strcmpi(condition, "Weak") or strcmpi(condition, "Dominant") or strcmpi(condition, "Strong")):
        raise Exception('Incorrect energy condition input, use either: "Null", "Weak", "Dominant", "Strong"')
    
    # Return warning for any coordinate system no cartesian
    if not strcmpi(metric['coords'], "cartesian"):
        raise Warning('Evaluation not verified for coordinate systems other than Cartesian!')
    
    # Check tensor formats are correct
    if not verify_tensor(metric, 1):
        raise Exception("Metric is not verified. Please verify metric using verify_tensor(metric)")
    
    if not verify_tensor(energy_tensor, 1):
        raise Exception("Stress-energy is not verified. Please verify stress-eenergy using verify_tensor(EnergyTensor)")

    # Get size of spacetime 
    a, b, c, d = metric['tensor'][0][0].shape

    # Convert energy tensor into the local inertial frame if not eulerian
    energy_tensor = do_frame_transfer(metric, energy_tensor, "Eulerian")

    # -------------------
    # Build Vector Fields
    # -------------------
    if strcmpi(condition, "Null") or strcmpi(condition, "Dominant"):
        type = "nulllike"
    
    elif strcmpi(condition, "Weak") or strcmpi(condition, "Strong"):
        type = "timelike"

    vec_field = generate_uniform_field(type, num_angular_vec, num_time_vec)

    # Declare variables to be determined in theeval of energy conditions
    map_array = np.full((a, b, c, d), np.nan)
    if return_vec == 1:
        vec = np.zeros((a, b, c, d, num_angular_vec, num_time_vec))

    # ----------------------
    # Find energy conditions
    # ----------------------

    # Null energy condition
    if strcmpi(condition, "Null"):
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric)

        for ii in range(num_angular_vec):
            temp = np.zeros((a, b, c, d))
        
            for mu in range(4):
                for nu in range(4):
                    temp = temp + energy_tensor['tensor'][mu][nu] * vec_field[mu][ii] * vec_field[nu][ii]

            map_array = np.minimum(map_array, temp)

            if return_vec == 1:
                vec[:, :, :, :, ii] = temp

    # Weak energy condition
    elif strcmpi(condition, "Weak"):
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric)

        for jj in range(num_time_vec):
            for ii in range(num_angular_vec):
                temp = np.zeros((a, b, c, d))

                for mu in range(4):
                    for nu in range(4):
                        temp = temp + energy_tensor['tensor'][mu][nu] * vec_field[mu][ii][jj] * vec_field[nu][ii][jj]

                map_array = np.minimum(map_array, temp)

                if return_vec == 1:
                    vec[:, :, :, :, ii, jj] = temp
        
    # Dominant energy condition
    elif strcmpi(condition, "Dominant"):
        metric_minkowski = metric_get_minkowski(np.array([a, b, c, d]))
        metric_minkowski = change_tensor_index(metric_minkowski, "covariant")

        energy_tensor = change_tensor_index(energy_tensor, "mixedupdown", metric_minkowski)

        for ii in range(num_angular_vec):
            temp = np.zeros((a, b, c, d, 4))

            for mu in range(4):
                for nu in range(4):
                    temp[:, :, :, :, mu] = temp[:, :, :, :, mu] - energy_tensor['tensor'][mu][nu] * vec_field[nu, ii]

            # Wrap into vector dict
            vector = {}
            vector['field'] = np.array([temp[:, :, :, :, 0], temp[:, :, :, :, 1], temp[:, :, :, :, 2], temp[:, :, :, :, 3]])
            vector['index'] = "contravariant"
            vector['type'] = "4-vector"

            # Find inner product to determine if timelike or null
            diff = get_inner_product(vector, vector, metric_minkowski)
            diff = np.sign(diff) * np.sqrt(np.abs(diff))
            
            map_array = np.maximum(map_array, diff)
            
            if return_vec == 1:
                vec[:, :, :, :, ii] = diff

        # Flip sign of the dominant energy condition to better align with evaluations of 
        # other conditions (i.e. negative is violating)

        map_array = - map_array
        if return_vec == 1:
            vec = - vec

    # Strong energy condition
    elif strcmpi(condition, "Strong"):
        # Build minkowski reference metric
        metric_minkowski = metric_get_minkowski(np.array([a, b, c, d]))
        metric_minkowski = change_tensor_index(metric_minkowski, "covariant")

        # Make sure the energy tensor is covariant
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric_minkowski)

        e_trace = get_trace(energy_tensor, metric_minkowski)

        for jj in range(num_time_vec):
            for ii in range(num_angular_vec):
                temp = np.zeros((a, b, c, d))

                for mu in range(4):
                    for nu in range(4):
                        temp = temp + (energy_tensor['tensor'][mu][nu] - 0.5 * e_trace * metric_minkowski['tensor'][mu][nu] * vec_field[mu][ii][jj] * vec_field[nu][ii][jj])
        
                map_array = np.minimum(map_array, temp)
                if return_vec == 1:
                    vec[:, :, :, :, ii, jj] = temp

    else:
        raise Exception("Unrecognised input energy condition, use either: 'Null', 'Weak', 'Strong', 'Dominant'")

    # If return_vec is set, return both the vec (other places in code) and the vector_field
    if return_vec == 1:
        vector_field_out = vec_field
    else:
        vec = []
        vector_field_out = []
    
    if return_vec == 1:
        return map_array, vec, vector_field_out

    return map_array
