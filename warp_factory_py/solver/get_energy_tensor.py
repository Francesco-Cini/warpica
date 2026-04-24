from datetime import datetime 

from warp_factory_py.solver.verify_tensor import verify_tensor
from warp_factory_py.solver.utils.strcmpi import strcmpi
from warp_factory_py.analyser.utils.change_tensor_index import change_tensor_index
from warp_factory_py.solver.utils.met_2_den import met_2_den

def get_energy_tensor(metric, diff_order):

    if diff_order is None:
        diff_order = 'fourth'
    
    if not verify_tensor(metric, 1): 
        raise Exception("Metric is not verified. Please verify metric using verify_tensor(metric).")
    
    if not strcmpi(metric['index'], "covariant"):
        metric = change_tensor_index(metric, "covariant")
        print(f"Changed metric from %s index to %s index\n", {metric["index"]}, "covariant")
    

    if strcmpi(diff_order, 'fourth'):
        energy_tensor = met_2_den(metric['tensor'], metric['scaling'])

    elif strcmpi(diff_order, 'second'):
        energy_tensor = met_2_den_2(metric['tensor'], metric['scaling'])

    else:
        raise Exception("Order Flag Not Specified Correctly. Options: 'fourth' or 'second'")
        
    energy = {}
        
    energy['type'] = "Stress-Energy" 
    energy['tensor'] = energy_tensor
    energy['coords'] = metric['coords']
    energy['index'] = "contravariant"
    energy['order'] = diff_order
    energy['name'] = metric['name']
    energy['date'] = datetime.now()

    return energy 