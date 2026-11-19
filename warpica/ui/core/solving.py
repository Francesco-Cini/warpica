"""The choices offered on the Solver page, and the function that applies them."""

import importlib.util

from warpica._typing import Backend, DiffOrder, TensorDict
from warpica.solver import get_energy_tensor

# Text shown to the user, and the value the solver expects. The first entry is
# the default, which is also the solver's own default.
DIFF_ORDERS: dict[str, DiffOrder] = {
    "Fourth order (more accurate)": "fourth",
    "Second order (faster)": "second",
}


def available_backends() -> dict[str, Backend | None]:
    """List the ways the solver can run on this computer.

    The CPU (NumPy) backend is always there and is the only one covered by the
    package's stability promise. The GPU backends are experimental and are
    offered only when their library is installed.

    Returns
    -------
    dict of str to {None, 'cupy', 'torch-directml'}
        Text to show, and the value to pass as ``gpu`` to the solver. The CPU
        comes first and is the default.
    """
    backends: dict[str, Backend | None] = {"CPU (NumPy)": None}
    if importlib.util.find_spec("cupy") is not None:
        backends["CUDA (CuPy), experimental"] = "cupy"
    if importlib.util.find_spec("torch_directml") is not None:
        backends["DirectML, experimental"] = "torch-directml"
    return backends


def solve_energy_tensor(
    metric: TensorDict, diff_order: DiffOrder, backend: Backend | None
) -> TensorDict:
    """Compute the stress-energy tensor of a metric.

    This is the function the background process runs. It sits at the top level
    of a module so the process can find it by name.

    Parameters
    ----------
    metric : dict
        The metric to solve.
    diff_order : {'second', 'fourth'}
        Finite-difference order.
    backend : {None, 'cupy', 'torch-directml'}
        Array backend, with None meaning the CPU.

    Returns
    -------
    dict
        The stress-energy tensor.
    """
    return get_energy_tensor(metric, diff_order, gpu=backend)
