def c_det_2(cell_array):
    """Compute the pointwise determinant by cofactor expansion.

    Parameters
    ----------
    cell_array : array or nested sequence of arrays
        Square matrix of scalar fields indexed [i][j].

    Returns
    -------
    determinant : array or scalar
        Determinant at each grid cell.
    """
    h = len(cell_array)
    if h == 0 or any(len(row) != h for row in cell_array):
        raise ValueError("cell_array must be nonempty and square")
    if h == 1:
        return cell_array[0][0]
    if h == 2:
        return cell_array[0][0] * cell_array[1][1] - cell_array[0][1] * cell_array[1][0]

    cell_det = 0
    for i in range(h):
        sub_array = [
            [cell_array[j][k] for k in range(h) if k != i] for j in range(1, h)
        ]
        cell_det = cell_det + (-1) ** i * cell_array[0][i] * c_det_2(sub_array)
    return cell_det
