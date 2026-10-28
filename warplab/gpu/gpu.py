import numpy as np


def asarray(value, *, library=None):
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


def asnumpy(value):
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

    if is_cupy_array(value):
        from .utils.cupy import asnumpy

        return asnumpy(value)

    if is_torch_array(value):
        from .utils.directml import asnumpy

        return asnumpy(value)

    return np.asarray(value)
