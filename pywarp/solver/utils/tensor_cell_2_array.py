import numpy as np

def tensor_cell_2_array(tensor):
    shape = tensor["tensor"][0][0].shape
    array_tensor = np.empty((*shape, 4, 4), dtype=np.result_type(*[
        tensor["tensor"][i][j]
        for i in range(4)
        for j in range(4)
    ]))

    for i in range(4):
        for j in range(4):
            array_tensor[..., i, j] = tensor["tensor"][i][j]

    return array_tensor
