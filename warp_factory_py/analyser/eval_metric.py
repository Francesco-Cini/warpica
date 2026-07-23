from warp_factory_py.solver.get_energy_tensor import get_energy_tensor
from warp_factory_py.analyser.do_frame_transfer import do_frame_transfer
from warp_factory_py.analyser.get_energy_conditions import get_energy_conditions

def eval_metric(metric, keep_positive, num_angular_vec, num_time_vec):

    # Handle default input arguments
    if keep_positive is None:
        keep_positive = 1

    if num_angular_vec is None:
        num_angular_vec = 100

    if num_time_vec is None:
        num_time_vec = 10

    # Metric output
    output = {}

    output['metric'] = metric
    
    #Energy tensor outputs
    output['energy_tensor'] = get_energy_tensor(metric)
    output['energy_tensor_eulerian'] = do_frame_transfer(metric, output['energy_tensor'], "Eulerian")

    #Energy condition outputs

    output['null'] = get_energy_conditions(output['energy_tensor'], metric, "Null", num_angular_vec, num_time_vec, 0)
    output['weak'] = get_energy_conditions(output['energy_tensor'], metric, "Weak", num_angular_vec, num_time_vec, 0)
    output['strong'] = get_energy_conditions(output['energy_tensor'], metric, "Strong", num_angular_vec, num_time_vec, 0)
    output['dominant'] = get_energy_conditions(output['energy_tensor'], metric, "Dominant", num_angular_vec, num_time_vec, 0)

    if not keep_positive:
        output['null'][output['null'] > 0] = 0
        output['weak'][output['weak'] > 0] = 0
        output['strong'][output['strong'] > 0] = 0
        output['dominant'][output['dominant'] > 0] = 0

    output['expansion'], output['shear'], output['vorticity'] = get_scalars(metric)

    return output
