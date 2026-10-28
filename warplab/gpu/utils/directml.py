"""DirectML uses float32: some required float64 operators are unsupported."""


def asarray(value):
    """Convert data to the DirectML backend.

    Parameters
    ----------
    value : array_like
        Input data to transfer or convert.

    Returns
    -------
    array : torch.Tensor
        Converted device array with float32 dtype.
    """
    import torch
    import torch_directml

    return torch.as_tensor(
        value,
        dtype=torch.float32,
        device=torch_directml.device(),
    )


def asnumpy(value):
    """Copy DirectML data to a NumPy host array.

    Parameters
    ----------
    value : torch.Tensor
        Device array to transfer.

    Returns
    -------
    array : numpy.ndarray
        Host data, detached from the autograd graph.
    """
    return value.detach().cpu().numpy()
