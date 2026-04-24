import numpy as np

def get_slice_data(plane, slice_centre, tensor):

    s = np.array(tensor['tensor'][0][0]).shape
    index_data = [np.arange(1, s[0] + 1),
                 np.arange(1, s[1] + 1),
                 np.arange(1, s[2] + 1),
                 np.arange(1, s[3] + 1)]

    index_data[int(plane[0] - 1)] = int(slice_centre[0])
    index_data[int(plane[1] - 1)] = int(slice_centre[1])

    return index_data