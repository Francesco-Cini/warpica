import numpy as np

from warp_factory_py.solver.verify_tensor import verify_tensor
from warp_factory_py.solver.utils.strcmpi import strcmpi
from warp_factory_py.solver.utils.c4_inv import c4_inv

def get_inner_product(vec_a, vec_b, metric):

    if not verify_tensor(metric, 1):
        raise Exception("Metric is not verified. Please verifty metric using verify_tensor(metric).")
    
    s = metric['tensor'][0][0].shape
    innerprod = np.zeros(s)

    if not strcmpi(vec_a['index'], vec_b['index']):
        for mu in range(4):
            for nu in range(4):
                innerprod = innerprod + vec_a['field'][mu] * vec_b['field'][nu]

    elif strcmpi(vec_a['index'], vec_b['index']):
        if strcmpi(vec_a['index'], metric['index']):
            metric['tensor'] = c4_inv(metric['tensor'])

        for mu in range(4):
            for nu in range(4):
                innerprod = innerprod + vec_a['field'][mu] * vec_b['field'][nu] * metric['tensor'][mu][nu]

    return innerprod