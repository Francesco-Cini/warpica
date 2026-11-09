def asarray(value):
    """Convert data to the CuPy backend.

    Parameters
    ----------
    value : array_like
        Input data to transfer or convert.

    Returns
    -------
    array : cupy.ndarray
        Converted device array.
    """
    import cupy as cp

    return cp.asarray(value)


def asnumpy(value):
    """Copy CuPy data to a NumPy host array.

    Parameters
    ----------
    value : cupy.ndarray
        Device array to transfer.

    Returns
    -------
    array : numpy.ndarray
        Host data.
    """
    import cupy as cp

    return cp.asnumpy(value)
