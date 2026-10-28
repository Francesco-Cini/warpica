def c3_inv(cell_array):
    """Invert a 3 by 3 matrix of fields pointwise.

    Parameters
    ----------
    cell_array : array or nested sequence of arrays
        Matrix components indexed [i][j], with 3 rows and columns.

    Returns
    -------
    inverse : list of lists of arrays
        A 3 by 3 component table, with the same grid shape as the input.

    Notes
    -----
    The matrix must be nonsingular at each grid cell.
    """
    if len(cell_array) != 3 or any(len(row) != 3 for row in cell_array):
        raise ValueError("Cell array is not 3x3")

    r = cell_array

    # Alias the components to match MATLAB notation
    r11, r12, r13 = r[0][0], r[0][1], r[0][2]
    r21, r22, r23 = r[1][0], r[1][1], r[1][2]
    r31, r32, r33 = r[2][0], r[2][1], r[2][2]

    # Determinant (element-wise over the grid)
    det = (
        r11 * r22 * r33
        - r11 * r23 * r32
        - r12 * r21 * r33
        + r12 * r23 * r31
        + r13 * r21 * r32
        - r13 * r22 * r31
    )

    # Inverse components (adjugate / det), element-wise
    inv11 = (r22 * r33 - r23 * r32) / det
    inv12 = (r13 * r32 - r12 * r33) / det
    inv13 = (r12 * r23 - r13 * r22) / det

    inv21 = (r23 * r31 - r21 * r33) / det
    inv22 = (r11 * r33 - r13 * r31) / det
    inv23 = (r13 * r21 - r11 * r23) / det

    inv31 = (r21 * r32 - r22 * r31) / det
    inv32 = (r12 * r31 - r11 * r32) / det
    inv33 = (r11 * r22 - r12 * r21) / det

    invcell_array = [
        [inv11, inv12, inv13],
        [inv21, inv22, inv23],
        [inv31, inv32, inv33],
    ]

    return invcell_array
