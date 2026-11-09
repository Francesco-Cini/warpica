import numpy as np

from warpica.solver.utils.tensor_cell_2_array import tensor_cell_2_array


def test_tensor_cell_2_array_places_tensor_indices_last():
    grid_shape = (2, 3, 4, 5)
    components = [
        [np.full(grid_shape, 10 * i + j, dtype=float) for j in range(4)]
        for i in range(4)
    ]

    result = tensor_cell_2_array({"tensor": components})

    assert result.shape == (*grid_shape, 4, 4)
    for i in range(4):
        for j in range(4):
            np.testing.assert_array_equal(result[..., i, j], components[i][j])
