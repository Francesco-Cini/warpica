from copy import deepcopy
from datetime import datetime

from warplab.analyser.change_tensor_index import change_tensor_index
from warplab.gpu import asarray as gpu_asarray
from warplab.gpu import asnumpy as gpu_asnumpy
from warplab.solver.utils.met_2_den import met_2_den
from warplab.solver.utils.second_order.met_2_den_2 import met_2_den_2
from warplab.solver.utils.strcmpi import strcmpi
from warplab.solver.verify_tensor import verify_tensor


def get_energy_tensor(metric, diff_order="fourth", *, gpu=None):
    """Compute the contravariant stress-energy tensor of a metric.

    Parameters
    ----------
    metric : dict
        Metric metadata and components indexed as tensor[mu][nu], each
        with shape (nt, nx, ny, nz).
    diff_order : {'second', 'fourth'}, optional
        Finite-difference accuracy order. Default is 'fourth'.
    gpu : {None, 'numpy', 'cupy', 'torch-directml'}, optional
        Array backend for curvature calculations. None selects NumPy.

    Returns
    -------
    energy : dict
        Stress-energy components as host arrays, with coordinate, index,
        finite-difference order, and source metric metadata.

    Notes
    -----
    DirectML uses float32 curvature. Final SI scaling and index contraction
    use float64 on the CPU for every device backend. The input metric is
    not modified.
    """
    if diff_order is None:
        diff_order = "fourth"

    if not verify_tensor(metric, 1):
        raise Exception(
            "Metric is not verified. Please verify metric using verify_tensor(metric)."
        )

    if not strcmpi(metric["index"], "covariant"):
        metric = change_tensor_index(metric, "covariant")

    if gpu is not None:
        metric_tensor_gpu = [[None for _ in range(4)] for _ in range(4)]

        for i in range(4):
            for j in range(4):
                metric_tensor_gpu[i][j] = gpu_asarray(
                    metric["tensor"][i][j], library=gpu
                )

        # Curvature stays on the device. SI scaling and index contraction use
        # CPU float64: c**4/(8*pi*G) exceeds the float32 range on DirectML.
        import numpy as np

        from warplab.solver.utils.c4_inv import c4_inv
        from warplab.solver.utils.ein_e import ein_e
        from warplab.solver.utils.ein_t import ein_t
        from warplab.solver.utils.ricci_s import ricci_s
        from warplab.solver.utils.ricci_t import ricci_t
        from warplab.solver.utils.second_order.ein_e_2 import ein_e_2
        from warplab.solver.utils.second_order.ricci_t_2 import ricci_t_2

        inverse = c4_inv(metric_tensor_gpu)
        if strcmpi(diff_order, "fourth"):
            ricci = ricci_t(inverse, metric_tensor_gpu, metric["scaling"])
        elif strcmpi(diff_order, "second"):
            ricci = ricci_t_2(inverse, metric_tensor_gpu, metric["scaling"])
        else:
            raise ValueError("diff_order must be 'fourth' or 'second'")
        einstein = ein_t(ricci, ricci_s(ricci, inverse), metric_tensor_gpu)
        host_einstein = [
            [np.asarray(gpu_asnumpy(v), dtype=np.float64) for v in row]
            for row in einstein
        ]
        host_inverse = [
            [np.asarray(gpu_asnumpy(v), dtype=np.float64) for v in row]
            for row in inverse
        ]
        energy_tensor = (
            ein_e_2(host_einstein, host_inverse, np.ones(3))
            if strcmpi(diff_order, "second")
            else ein_e(host_einstein, host_inverse)
        )

    else:
        if strcmpi(diff_order, "fourth"):
            energy_tensor = met_2_den(metric["tensor"], metric["scaling"])

        elif strcmpi(diff_order, "second"):
            energy_tensor = met_2_den_2(metric["tensor"], metric["scaling"])

        else:
            raise Exception(
                "Order Flag Not Specified Correctly. Options: 'fourth' or 'second'"
            )

    energy = {}

    energy["type"] = "Stress-Energy"
    energy["tensor"] = energy_tensor
    energy["coords"] = metric["coords"]
    energy["index"] = "contravariant"
    energy["order"] = diff_order
    energy["name"] = metric["name"]
    energy["date"] = datetime.now()
    energy["scaling"] = deepcopy(metric["scaling"])
    if "params" in metric:
        energy["params"] = dict(metric["params"])

    return energy
