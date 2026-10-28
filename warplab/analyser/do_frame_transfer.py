import copy

import numpy as np

from warplab.analyser.change_tensor_index import change_tensor_index
from warplab.analyser.utils.get_eulerian_transformation_matrix import (
    get_eulerian_transformation_matrix,
)
from warplab.solver.utils.is_field import is_field
from warplab.solver.utils.strcmpi import strcmpi
from warplab.solver.utils.tensor_cell_2_array import tensor_cell_2_array
from warplab.solver.verify_tensor import verify_tensor


def do_frame_transfer(metric, energy_tensor, frame, gpu=None):
    """Transform stress-energy to a local Eulerian orthonormal frame.

    Parameters
    ----------
    metric : dict
        Metric metadata and components indexed as tensor[mu][nu], each
        with shape (nt, nx, ny, nz).
    energy_tensor : dict
        Stress-energy tensor in any supported index convention.
    frame : {'Eulerian'}
        Target frame, compared case-insensitively.
    gpu : str or None, optional
        Retained for compatibility; this transformation runs on the CPU.

    Returns
    -------
    transformed_energy_tensor : dict
        Independent tensor copy with contravariant Eulerian components.
    """
    if energy_tensor.get("frame", "").lower() == frame.lower() == "eulerian":
        return copy.deepcopy(energy_tensor)
    metric = change_tensor_index(metric, "covariant")

    transformed_energy_tensor = copy.deepcopy(energy_tensor)
    transformed_energy_tensor["tensor"] = [[None for _ in range(4)] for _ in range(4)]

    if not verify_tensor(metric, 1):
        raise Exception(
            "Metric is not verified. Please verify metric using verify_tensor(metric)."
        )

    if not verify_tensor(energy_tensor, 1):
        raise Exception(
            "Stress-energy is not verified. Please veify Stress-energy tensor using verify_tensor(energy_tensor)."
        )

    if strcmpi(frame, "Eulerian") and not (
        is_field(energy_tensor, "frame") and strcmpi(energy_tensor["frame"], "Eulerian")
    ):
        energy_tensor = change_tensor_index(energy_tensor, "covariant", metric)

        array_energy_tensor = tensor_cell_2_array(energy_tensor)
        array_metric_tensor = tensor_cell_2_array(metric)

        M = get_eulerian_transformation_matrix(array_metric_tensor, metric["coords"])

        transformed_temp_tensor = {}

        transformed_temp_tensor["tensor"] = np.matmul(
            np.matmul(np.swapaxes(M, -1, -2), array_energy_tensor), M
        )

        z = energy_tensor["tensor"][0][0].shape

        for i in range(4):
            for j in range(4):
                transformed_energy_tensor["tensor"][i][j] = transformed_temp_tensor[
                    "tensor"
                ][..., i, j].reshape(z)

        # Raising both indices in the orthonormal frame flips only mixed
        # time-space components because the signature is (-, +, +, +).
        for i in range(1, 4):
            transformed_energy_tensor["tensor"][0][i] = -transformed_energy_tensor[
                "tensor"
            ][0][i]
            transformed_energy_tensor["tensor"][i][0] = -transformed_energy_tensor[
                "tensor"
            ][i][0]

        transformed_energy_tensor["frame"] = "Eulerian"
        transformed_energy_tensor["index"] = "contravariant"

    else:
        raise Warning("Frame not found")

    return transformed_energy_tensor
