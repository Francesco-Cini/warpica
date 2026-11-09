from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from warpica._typing import Array, Backend


def asarray(
    value: ArrayLike,
    *,
    library: Backend | None = None,
) -> Any:
    """Convert a value to the selected CPU or GPU array backend.

    Parameters
    ----------
    value : array_like
        Data to convert.
    library : {None, 'numpy', 'cupy', 'torch-directml'}, optional
        Target backend. None selects NumPy; GPU imports are deferred.

    Returns
    -------
    array : ndarray or torch.Tensor
        Converted data. DirectML uses float32; other backends infer dtype.
    """
    if library is None or library == "numpy":
        return np.asarray(value)

    # Backends are imported only when selected, so CuPy and torch-directml are not
    # required unless used.
    if library == "cupy":
        from .utils.cupy import asarray

        return asarray(value)

    if library == "torch-directml":
        from .utils.directml import asarray

        return asarray(value)

    raise ValueError(
        f"Unsupported array library: {library!r}. "
        "Use None, 'numpy', 'cupy', or 'torch-directml'."
    )


def asnumpy(
    value: Any,
) -> Array:
    """Return a NumPy view or host copy of an array.

    Parameters
    ----------
    value : array_like
        NumPy, CuPy, or PyTorch data.

    Returns
    -------
    array : numpy.ndarray
        Host array; PyTorch tensors are detached before conversion.
    """
    from array_api_compat import (
        is_cupy_array,
        is_torch_array,
    )

    # Device arrays are copied back to host NumPy arrays; anything else converts
    # directly.
    if is_cupy_array(value):
        from .utils.cupy import asnumpy

        return asnumpy(value)

    if is_torch_array(value):
        from .utils.directml import asnumpy

        return asnumpy(value)

    return np.asarray(value)
