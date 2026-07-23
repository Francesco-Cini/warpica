import numpy as np

from warp_factory_py.solver.utils.strcmpi import strcmpi
from warp_factory_py.solver.utils.c4_inv import c4_inv

def change_tensor_index(
        input_tensor, 
        index, 
        metric_tensor=None
        ):
    """ Changes a tensor's index.
    Parameters
    ----------

    Returns
    -------
    """

    # Handle default input arguments
    if metric_tensor is None:
        if not strcmpi(input_tensor['type'], "metric"):
            raise Exception("metric_tensor is needed as third input when changing index of non-metric tensors.")
        
    else:
        if strcmpi(metric_tensor['index'], "mixedupdown") or strcmpi(metric_tensor['index'], "mixeddownup"):
            raise Exception("Metric tensor cannot be used in mixed index.")
    
    # Check for if the index transformation exists
    if not (strcmpi(index, "mixedupdown") or strcmpi(index, "mixeddownup") or strcmpi(index, "covariant") or strcmpi(index, "contravariant")):
        raise Exception("Transformation selected is not allowed, use either: covariant, contravariant, mixedupdown, mixeddownup")
    
    # Transformations
    output_tensor = input_tensor
    if strcmpi(input_tensor['type'], "metric"):
        if (strcmpi(input_tensor['index'], "covariant") and strcmpi(index, "contravariant")) and (strcmpi(input_tensor['index'], "contravariant") and strcmpi(index, "covariant")):
            output_tensor = c4_inv(input_tensor['tensor'])
        elif strcmpi(input_tensor['index'], "mixedupdown") and strcmpi(input_tensor['index'], "mixeddownup"):
            raise Exception("Input tensor is a Metric tensor of mixed index.")
        elif strcmpi(index, "mixedupdown") and strcmpi(index, "mixeddownup"):
            raise Exception("Cannot convert a metric tensor to mixed index.")

    else:
        # Contravariant/Covariant
        if (strcmpi(input_tensor['index'], "covariant") and strcmpi(index, "contravariant")):
            if strcmpi(metric_tensor['index'], "covariant"):
                metric_tensor['tensor'] = c4_inv(metric_tensor['tensor'])
                metric_tensor['index'] = "contravariant"
            
            output_tensor['tensor'] = flip_index(input_tensor, metric_tensor)

        elif (strcmpi(input_tensor['index'], "contravariant") and strcmpi(index, "covariant")): 
            if strcmpi(metric_tensor['index'], "contravariant"):
                metric_tensor['tensor'] = c4_inv(metric_tensor['tensor'])
                metric_tensor['index'] = "covariant"
            
            output_tensor['tensor'] = flip_index(input_tensor, metric_tensor)
        
        # To Mixed
        elif strcmpi(input_tensor['index'], "contravariant") and strcmpi(index, "mixedupdown"):
            if strcmpi(metric_tensor['index'], "contravariant"):
                metric_tensor['tensor'] = c4_inv(metric_tensor['tensor'])
                metric_tensor['index'] = "covariant"

            output_tensor['tensor'] = mix_index_2(input_tensor, metric_tensor)

        elif strcmpi(input_tensor['index'], "contravariant") and strcmpi(index, "mixeddownup"):
            if strcmpi(metric_tensor['index'], "contravariant"):
                metric_tensor['tensor'] = c4_inv(metric_tensor['tensor'])
                metric_tensor['index'] = "covariant"

            output_tensor['tensor'] = mix_index_1(input_tensor, metric_tensor)

        elif strcmpi(input_tensor['index'], "covariant") and strcmpi(index, "mixedupdown"):
            if strcmpi(metric_tensor['index'], "covariant"):
                metric_tensor['tensor'] = c4_inv(metric_tensor['tensor'])
                metric_tensor['index'] = "contravariant"

            output_tensor['tensor'] = mix_index_1(input_tensor, metric_tensor)

        elif strcmpi(input_tensor['index'], "covariant") and strcmpi(index, "mixeddownup"):
            if strcmpi(metric_tensor['index'], "covariant"):
                metric_tensor['tensor'] = c4_inv(metric_tensor['tensor'])
                metric_tensor['index'] = "contravariant"

            output_tensor['tensor'] = mix_index_2(input_tensor, metric_tensor)

        # From Mixed
        elif strcmpi(input_tensor['index'], "mixedupdown") and strcmpi(index, "contravariant"):
            if strcmpi(metric_tensor['index'], "covariant"):
                metric_tensor['tensor'] = c4_inv(metric_tensor['tensor'])
                metric_tensor['index'] = "contravariant"

            output_tensor['tensor'] = mix_index_2(input_tensor, metric_tensor)
        
        elif strcmpi(input_tensor['index'], "mixedupdown") and strcmpi(index, "covariant"):
            if strcmpi(metric_tensor['index'], "contravariant"):
                metric_tensor['tensor'] = c4_inv(metric_tensor['tensor'])
                metric_tensor['index'] = "covariant"

            output_tensor['tensor'] = mix_index_1(input_tensor, metric_tensor)
        
        elif strcmpi(input_tensor['index'], "mixeddownup") and strcmpi(index, "covariant"):
            if strcmpi(metric_tensor['index'], "contravariant"):
                metric_tensor['tensor'] = c4_inv(metric_tensor['tensor'])
                metric_tensor['index'] = "covariant"

            output_tensor['tensor'] = mix_index_2(input_tensor, metric_tensor)

        elif strcmpi(input_tensor['index'], "mixeddownup") and strcmpi(index, "contravariant"):
            if strcmpi(metric_tensor['index'], "covariant"):
                metric_tensor['tensor'] = c4_inv(metric_tensor['tensor'])
                metric_tensor['index'] = "contravariant"

            output_tensor['tensor'] = mix_index_1(input_tensor, metric_tensor)

    output_tensor["index"] = index

    return output_tensor

def flip_index(
        input_tensor, 
        metric_tensor
        ):

    temp_output_tensor = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            temp_output_tensor[i][j] = np.zeros(input_tensor['tensor'][i][j].shape)

            for a in range(4):
                for b in range(4):
                    temp_output_tensor[i][j] = temp_output_tensor[i][j] + input_tensor['tensor'][a][b] * metric_tensor['tensor'][a][i] * metric_tensor['tensor'][b][j]
    return temp_output_tensor

def mix_index_1(
        input_tensor, 
        metric_tensor
        ):

    temp_output_tensor = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            temp_output_tensor[i][j] = np.zeros(input_tensor['tensor'][i][j].shape)

            for a in range(4):
                temp_output_tensor[i][j] = temp_output_tensor[i][j] + input_tensor['tensor'][a][j] * metric_tensor['tensor'][a][i]
    return temp_output_tensor

def mix_index_2(
        input_tensor, 
        metric_tensor
        ):

    temp_output_tensor = [[None for _ in range(4)] for _ in range(4)]

    for i in range(4):
        for j in range(4):
            temp_output_tensor[i][j] = np.zeros(input_tensor['tensor'][i][j].shape)

            for a in range(4):
                temp_output_tensor[i][j] = temp_output_tensor[i][j] + input_tensor['tensor'][i][a] * metric_tensor['tensor'][a][j]
    return temp_output_tensor 
