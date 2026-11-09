from copy import deepcopy
from datetime import datetime
from typing import Any

from warpica._typing import Backend, DiffOrder, TensorDict
from warpica.gpu import asarray as gpu_asarray
from warpica.gpu import asnumpy as gpu_asnumpy
from warpica.solver.utils.met_2_den import met_2_den
from warpica.solver.utils.second_order.met_2_den_2 import met_2_den_2
from warpica.solver.utils.strcmpi import strcmpi
from warpica.solver.utils.verify_grid_size import verify_grid_size
from warpica.solver.verify_tensor import verify_tensor


def get_energy_tensor(
    metric: TensorDict,
    diff_order: DiffOrder | None = "fourth",
    *,
    gpu: Backend | None = None,
) -> TensorDict:
    """Compute the contravariant stress-energy tensor of a metric.

    Parameters
    ----------
    metric : dict
        Metric metadata and components indexed as tensor[mu][nu], each
        with shape (nt, nx, ny, nz).
    diff_order : {'second', 'fourth'}, optional
        Finite-difference accuracy order. Default is 'fourth'.
    gpu : {None, 'numpy', 'cupy', 'torch-directml'}, optional
        Array backend for curvature calculations. None selects NumPy. Only
        the NumPy path is covered by the stability promise; 'cupy' and
        'torch-directml' are experimental.

    Returns
    -------
    energy : dict
        Stress-energy components as host arrays, with coordinate, index,
        finite-difference order, and source metric metadata.

    Notes
    -----
    Every axis must have either one sample or at least 3 (second order) or 5
    (fourth order); a shorter axis raises ValueError instead of silently giving
    zero derivatives.

    DirectML uses float32 curvature. Final SI scaling and index contraction
    use float64 on the CPU for every device backend. The input metric is
    not modified.
    """
    # Imported here because analyser imports solver; a module-level import is circular.
    from warpica.analyser.change_tensor_index import change_tensor_index

    if diff_order is None:
        diff_order = "fourth"

    if not verify_tensor(metric, 1):
        raise ValueError(
            "Metric is not verified. Please verify metric using verify_tensor(metric)."
        )

    verify_grid_size(metric["tensor"][0][0].shape, diff_order)

    # The curvature formulas below take the covariant metric g_mu_nu
    # (indices down), so convert first if the caller gave another form.
    if not strcmpi(metric["index"], "covariant"):
        metric = change_tensor_index(metric, "covariant")

    if gpu is not None:
        metric_tensor_gpu: list[list[Any]] = [
            [None for _ in range(4)] for _ in range(4)
        ]

        for i in range(4):
            for j in range(4):
                metric_tensor_gpu[i][j] = gpu_asarray(
                    metric["tensor"][i][j], library=gpu
                )

        # Curvature stays on the device. SI scaling and index contraction use
        # CPU float64: c**4/(8*pi*G) exceeds the float32 range on DirectML.
        import numpy as np

        from warpica.solver.utils.c4_inv import c4_inv
        from warpica.solver.utils.ein_e import ein_e
        from warpica.solver.utils.ein_t import ein_t
        from warpica.solver.utils.ricci_s import ricci_s
        from warpica.solver.utils.ricci_t import ricci_t
        from warpica.solver.utils.second_order.ein_e_2 import ein_e_2
        from warpica.solver.utils.second_order.ricci_t_2 import ricci_t_2

        # Same chain as met_2_den (inverse metric -> Ricci tensor -> Ricci
        # scalar -> Einstein tensor -> stress-energy), spelled out step by step
        # so that only the curvature runs on the device.
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
            raise ValueError(
                "Order Flag Not Specified Correctly. Options: 'fourth' or 'second'"
            )

    # Wrap the result with the metadata later stages need. The tensor is
    # contravariant T^mu_nu in SI units, on the same grid as the metric.
    energy: TensorDict = {}

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
