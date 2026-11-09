from warpica.solver.utils.c4_inv import c4_inv


def get_trace(tensor, metric):
    """Contract a rank-two tensor without changing the reference metric.

    Parameters
    ----------
    tensor : dict
        Tensor in a covariant, contravariant, or mixed index convention.
    metric : dict
        Metric metadata and components indexed as `tensor[mu][nu]`, each
        with shape (nt, nx, ny, nz).

    Returns
    -------
    trace : ndarray
        Scalar trace at each grid cell.
    """
    # A mixed tensor is already contracted by position: the trace is the sum of the
    # diagonal. Otherwise contract with the metric (or its inverse, as needed).
    if tensor["index"].lower().startswith("mixed"):
        return sum(tensor["tensor"][i][i] for i in range(4))
    g = metric["tensor"]
    if tensor["index"].lower() == metric["index"].lower():
        g = c4_inv(g)
    return sum(g[i][j] * tensor["tensor"][i][j] for i in range(4) for j in range(4))
