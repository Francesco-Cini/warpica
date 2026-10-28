from warplab.analyser.do_frame_transfer import do_frame_transfer
from warplab.analyser.get_energy_conditions import get_energy_conditions
from warplab.analyser.get_scalars import get_scalars
from warplab.solver.get_energy_tensor import get_energy_tensor


def eval_metric(
    metric,
    keep_positive=True,
    num_angular_vec=100,
    num_time_vec=10,
    gpu=None,
    *,
    diff_order="fourth",
):
    """Evaluate stress-energy, sampled energy conditions, and Eulerian scalars.

    Parameters
    ----------
    metric : dict
        Metric metadata and components indexed as tensor[mu][nu], each
        with shape (nt, nx, ny, nz).
    keep_positive : bool, optional
        Keep positive energy-condition values. If false, clip them to zero.
    num_angular_vec : int, optional
        Number of sampled spatial directions; default is 100.
    num_time_vec : int, optional
        Number of timelike speed samples; default is 10.
    gpu : {None, 'numpy', 'cupy', 'torch-directml'}, optional
        Array backend for curvature calculations. None selects NumPy.
    diff_order : {'second', 'fourth'}, optional
        Finite-difference accuracy order. Default is 'fourth'.

    Returns
    -------
    output : dict
        Input metric, coordinate and Eulerian stress-energy tensors, null,
        weak, strong, and dominant condition maps, and expansion, shear,
        and vorticity scalars.
    """
    if keep_positive is None:
        keep_positive = 1

    if num_angular_vec is None:
        num_angular_vec = 100

    if num_time_vec is None:
        num_time_vec = 10

    output = {}

    output["metric"] = metric

    output["energy_tensor"] = get_energy_tensor(metric, diff_order, gpu=gpu)
    output["energy_tensor_eulerian"] = do_frame_transfer(
        metric, output["energy_tensor"], "Eulerian"
    )

    output["null"] = get_energy_conditions(
        output["energy_tensor_eulerian"],
        metric,
        "Null",
        num_angular_vec,
        num_time_vec,
        0,
        gpu,
    )
    output["weak"] = get_energy_conditions(
        output["energy_tensor_eulerian"],
        metric,
        "Weak",
        num_angular_vec,
        num_time_vec,
        0,
        gpu,
    )
    output["strong"] = get_energy_conditions(
        output["energy_tensor_eulerian"],
        metric,
        "Strong",
        num_angular_vec,
        num_time_vec,
        0,
        gpu,
    )
    output["dominant"] = get_energy_conditions(
        output["energy_tensor_eulerian"],
        metric,
        "Dominant",
        num_angular_vec,
        num_time_vec,
        0,
        gpu,
    )

    if not keep_positive:
        output["null"][output["null"] > 0] = 0
        output["weak"][output["weak"] > 0] = 0
        output["strong"][output["strong"] > 0] = 0
        output["dominant"][output["dominant"] > 0] = 0

    output["expansion"], output["shear"], output["vorticity"] = get_scalars(
        metric, diff_order=diff_order
    )

    return output
