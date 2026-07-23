import warnings
import numpy as np
from warp_factory_py.solver.utils.is_field import is_field
from warp_factory_py.solver.utils.strcmpi import strcmpi

def verify_tensor(input_tensor, suppress_msgs = None):

    # Handle input arguments

    if suppress_msgs is None:
        suppress_msgs = 0

    # Preset functions
    
    def disp_message(msg, sM):
        if not sM:
            print(msg)

    verified = True

    # Check if type field exists

    if is_field(input_tensor, 'type'): 
        if strcmpi(input_tensor['type'], 'Metric'):

            disp_message("Type: Metric", suppress_msgs)
        elif strcmpi(input_tensor['type'], "Stress-Energy"):

            disp_message("Type: Stress-Energy", suppress_msgs)
        elif not is_field(input_tensor, 'type'):
            warnings.warn("Tensor type field does not exist. Must be either 'Metric' or 'Stress-Energy'")
            verified = False
        else:
            warnings.warn("Unknown Type")
            verified = False
        
        # Check other properties

        # Tensor

        if is_field(input_tensor, 'tensor'):
            tensor = input_tensor['tensor']
            if (
                (
                    isinstance(tensor, (list, tuple))
                    and len(tensor) == 4
                    and all(isinstance(row, (list, tuple, np.ndarray)) and len(row) == 4 for row in tensor)
                )
                or (
                    isinstance(tensor, np.ndarray)
                    and tensor.shape[:2] == (4, 4)
                )
            ) and (
                hasattr(tensor[0][0], "shape")
                and len(tensor[0][0].shape) == 4
            ):
                disp_message("tensor: Verified", suppress_msgs)
            else:
                warnings.warn("Tensor is not formatted correctly. Tensor must be a 4x4 cell array of 4D values.")
                verified = False
        else:
            warnings.warn("tensor: Empty")
            verified = False

        # Coords

        if is_field(input_tensor, 'coords'):
            if strcmpi(str(input_tensor['coords']), 'cartesian'):
                disp_message("coords: " + str(input_tensor["coords"]),suppress_msgs)
            else:
                warnings.warn("Non-cartesian coordinates are not supported at this time. Set .coords to 'cartesian'.")
        else:
            warnings.warn("coords: Empty")
            verified = False

        # Index

        if is_field(input_tensor, 'index'):
            if (strcmpi(str(input_tensor['index']), 'contravariant') or
            strcmpi(str(input_tensor['index']), 'covariant') or
            strcmpi(str(input_tensor['index']), 'mixedupdown') or
            strcmpi(str(input_tensor['index']), 'mixeddownup')):

                disp_message("index" + str(input_tensor["index"]), suppress_msgs)
            else:
                warnings.warn("Unknown index")
                verified = False
        else:
            warnings.warn("index: Empty")
            verified = False
    else:
        warnings("Tensor type does not exist. Must be either 'Metric' or 'Stress-Energy'")
        verified = False

    # Reset user's backtrace setting
    
    return verified
