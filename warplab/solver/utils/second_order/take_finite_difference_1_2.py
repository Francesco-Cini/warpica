from array_api_compat import array_namespace


def take_finite_difference_1_2(A, k, delta):
    """Approximate a first derivative with second-order differences.

    Parameters
    ----------
    A : array_like, shape (nt, nx, ny, nz)
        Scalar field to differentiate; allocation follows its array backend.
    k : int
        Zero-based differentiation axis: 0=t, 1=x, 2=y, 3=z.
    delta : array_like, shape (4,)
        Grid spacings in (t, x, y, z) order.

    Returns
    -------
    B : array
        Derivative field with the same shape and backend as A.

    Notes
    -----
    Axes shorter than three samples return zeros.
    Boundary cells copy the nearest interior derivative.
    """
    if getattr(getattr(A, "device", None), "type", None) == "privateuseone":
        from warplab.gpu.utils.directml_derivatives import derivative

        return derivative(A, k, delta, 2)

    s = A.shape
    xp = array_namespace(A)
    B = xp.zeros_like(
        A, dtype=A.dtype if xp.isdtype(A.dtype, "real floating") else xp.float64
    )

    if s[k] >= 3:
        match k:
            case 0:
                B[1:-1, :, :, :] = (A[2:, :, :, :] - A[:-2, :, :, :]) / (2 * delta[k])
                B[0, :, :, :] = B[1, :, :, :]
                B[-1, :, :, :] = B[-2, :, :, :]
            case 1:
                B[:, 1:-1, :, :] = (A[:, 2:, :, :] - A[:, :-2, :, :]) / (2 * delta[k])
                B[:, 0, :, :] = B[:, 1, :, :]
                B[:, -1, :, :] = B[:, -2, :, :]
            case 2:
                B[:, :, 1:-1, :] = (A[:, :, 2:, :] - A[:, :, :-2, :]) / (2 * delta[k])
                B[:, :, 0, :] = B[:, :, 1, :]
                B[:, :, -1, :] = B[:, :, -2, :]
            case 3:
                B[:, :, :, 1:-1] = (A[:, :, :, 2:] - A[:, :, :, :-2]) / (2 * delta[k])
                B[:, :, :, 0] = B[:, :, :, 1]
                B[:, :, :, -1] = B[:, :, :, -2]

    return B
