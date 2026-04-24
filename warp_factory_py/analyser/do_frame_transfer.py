from solver.verify_tensor import verify_tensor

def do_frame_transfer(metric, energy_tensor, frame, try_gpu):

    if try_gpu is None:
        try_gpu = 0

    transformed_energy_tensor = energy_tensor
    transformed_energy_tensor['tensor'] = 

    if not verify_tensor(metric, 1):