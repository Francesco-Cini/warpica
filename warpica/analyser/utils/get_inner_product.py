from warpica.solver.utils.c4_inv import c4_inv


def get_inner_product(vec_a, vec_b, metric):
    """Contract two four-vector fields without changing either input.

    Parameters
    ----------
    vec_a, vec_b : dict
        Vector dictionaries with 'field' components and an 'index' convention.
    metric : dict
        Metric metadata and components indexed as tensor[mu][nu], each
        with shape (nt, nx, ny, nz).

    Returns
    -------
    inner_product : ndarray
        Scalar contraction at each grid cell.
    """
    # One vector up and one down contract directly, with no metric.
    if vec_a["index"].lower() != vec_b["index"].lower():
        return sum(vec_a["field"][i] * vec_b["field"][i] for i in range(4))
    # Both vectors are in the same index position, so contract with a metric in
    # the opposite position: g_ij for contravariant vectors, g^ij for covariant
    # ones. The stored metric is inverted when it sits in the vectors' position.
    g = metric["tensor"]
    if vec_a["index"].lower() == metric["index"].lower():
        g = c4_inv(g)
    return sum(
        vec_a["field"][i] * vec_b["field"][j] * g[i][j]
        for i in range(4)
        for j in range(4)
    )
