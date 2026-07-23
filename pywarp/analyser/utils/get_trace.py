import numpy as np

from warp_factory_py.solver.verify_tensor import verify_tensor
from warp_factory_py.solver.utils.strcmpi import strcmpi
from warp_factory_py.solver.utils.c4_inv import c4_inv

def get_trace(tensor, metric):

    if not verify_tensor(metric, 1):
        raise Exception("Metric is not verified. Please verify metric using verify_tensor(metric).")
    
    trace = np.zeros(metric['tensor'][0][0].shape)

    if strcmpi(tensor['index'], metric['index']):
        metric['tensor'] = c4_inv(metric['tensor'])

    for a in range(4):
        for b in range(4):
            trace = trace + metric['tensor'][a][b] * tensor['tensor'][a][b]

    return trace