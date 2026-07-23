import numpy as np
from solver.verify_tensor import verify_tensor
from solver.utils.strcmpi import strcmpi
from warp_factory_py.visualiser.utils.label_cartesian_axis import label_cartesian_axis
from warp_factory_py.visualiser.utils.get_slice_data import get_slice_data
from warp_factory_py.visualiser.utils.plot_component import plot_component

def plot_tensor(tensor, alpha = None, sliced_planes = None, slice_locations = None):

    # Default Input Arguments

    if alpha is None:
        alpha = 0.2

    if sliced_planes is None:
        sliced_planes = np.arrray([1, 4])

    if slice_locations is None:
        s = np.array(tensor.tensor[0][0]).shape
        slice_centres = np.round((s+1)/2)
        slice_locations[0] = slice_centres(sliced_planes[0])
        slice_locations[1] = slice_centres(sliced_planes[1])

    # Verify Tensor

    if not verify_tensor(tensor, 1):
        raise Exception("Tensor is not verified. Please verify tensor using verify_tensor(tensor)")

    # Check that the sliced planes are different

    if sliced_planes[0] == sliced_planes[1]:
        raise Exception("Selected planes must not be the same, select two different planes to slice along.")

    # Round slice_locations

    slice_locations = np.round(slice_locations)

    # Check that the slice_locations are inside the world

    arr = np.array(tensor.tensor[0][0])

    if (slice_locations[0] < 1 or slice_locations[1] < 1 or
        slice_locations[0] > arr.shape[int(sliced_planes[0]) - 1] or
        slice_locations[1] > arr.shape[int(sliced_planes[1]) - 1]):
        slice_locations[0]
        slice_locations[1]
        np.array(tensor.tensor[0][0], sliced_planes[0]).shape
        np.array(tensor.tensor[0][0], sliced_planes[0]).shape
        raise Exception("slice_locations are outside the world.")

    # Check tensor type

    if strcmpi(tensor['type'], 'Metric'):
        titleCharacter = 'g'
    elif strcmpi(tensor['type'], 'Stress-Energy'):
        titleCharacter = 'T'

    # Check tensor index

    if strcmpi(tensor['index'], 'covariant'):
        titleAugment1 = '_{'
        titleAugment2 = ''
    elif strcmpi(tensor['index'], 'contravariant'):
        titleAugment1 = '^{'
        titleAugment2 = ''
    elif strcmpi(tensor['index'], 'mixedupdown'):
        titleAugment1 = '^{'
        titleAugment2 = '}_{'
    elif strcmpi(tensor['index'], 'mixeddownup'):
        titleAugment1 = '_{'
        titleAugment2 = '}_{'
    
    # Check that the coords are Cartesian

    if strcmpi(tensor['coords'], 'cartesian'):
        xLabelText, yLabelText = label_cartesian_axis(sliced_planes)

        if strcmpi(tensor['index'], 'mixedupdown') or strcmpi(tensor['index'], 'mixeddownup'):
            c1 = np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4])
            c2 = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
        
        else:
            c1 = np.array([1, 1, 1, 1, 2, 3, 4, 2, 2, 3])
            c2 = np.array([1, 2, 3, 4, 2, 3, 4, 3, 4, 4])
        
        idx = get_slice_data(sliced_planes, slice_locations, tensor)

        for i in range(len(c1)):
            plot_component()
    
    else:
        raise Exception("Unknown coordinate system, must be: 'cartesian'")

    return plot_tensor
