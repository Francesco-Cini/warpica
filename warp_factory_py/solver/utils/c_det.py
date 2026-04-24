import numpy as np

def c_det(cell_array):

    h = len(cell_array)
    w = len(cell_array[0])

    if h==2 and w==2:
       return cell_array[0][0] * cell_array[1][1] - cell_array[0][1] * cell_array[1][0]

    cell_det = 0

    for i in range(h):
        sub_array = [row[:] for row in cell_array]  
        sub_array.pop(0)                              
        for r in range(len(sub_array)):               
            sub_array[r].pop(i)

        sub_det = c_det(sub_array)
        cell_det = cell_det + (2 * ((i + 1) % 2) - 1) * cell_array[0][i] * sub_det

    return cell_det