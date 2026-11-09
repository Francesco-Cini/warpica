from array_api_compat import array_namespace


def take_finite_difference_1(A, k, delta):
    """Approximate a first derivative with fourth-order differences.

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
    Axes shorter than five samples return zeros.
    Boundary cells copy the nearest interior derivative.
    """
    # DirectML tensors take the dedicated GPU derivative kernel instead.
    if getattr(getattr(A, "device", None), "type", None) == "privateuseone":
        from warpica.gpu.utils.directml_derivatives import derivative

        return derivative(A, k, delta, 4)

    s = A.shape
    xp = array_namespace(A)
    B = xp.zeros_like(
        A, dtype=A.dtype if xp.isdtype(A.dtype, "real floating") else xp.float64
    )

    # Fourth-order central difference along axis k:
    #   dA/dx ~ (-A[i+2] + 8 A[i+1] - 8 A[i-1] + A[i-2]) / (12 h).
    # It needs two neighbours on each side, hence at least 5 samples.
    if s[k] >= 5:
        match k:
            case 0:
                B[2:-2, :, :, :] = (
                    -(A[4:, :, :, :] - A[:-4, :, :, :])
                    + 8 * (A[3:-1, :, :, :] - A[1:-3, :, :, :])
                ) / (12 * delta[k])
                # The stencil does not fit in the two outermost cells on each side, so
                # they copy the nearest computed value. Derivatives there are only
                # first-order accurate; keep the region of interest away from the edges.
                B[0, :, :, :] = B[2, :, :, :]
                B[1, :, :, :] = B[2, :, :, :]
                B[-2, :, :, :] = B[-3, :, :, :]
                B[-1, :, :, :] = B[-3, :, :, :]
            case 1:
                B[:, 2:-2, :, :] = (
                    -(A[:, 4:, :, :] - A[:, :-4, :, :])
                    + 8 * (A[:, 3:-1, :, :] - A[:, 1:-3, :, :])
                ) / (12 * delta[k])
                B[:, 0, :, :] = B[:, 2, :, :]
                B[:, 1, :, :] = B[:, 2, :, :]
                B[:, -2, :, :] = B[:, -3, :, :]
                B[:, -1, :, :] = B[:, -3, :, :]
            case 2:
                B[:, :, 2:-2, :] = (
                    -(A[:, :, 4:, :] - A[:, :, :-4, :])
                    + 8 * (A[:, :, 3:-1, :] - A[:, :, 1:-3, :])
                ) / (12 * delta[k])
                B[:, :, -2, :] = B[:, :, -3, :]
                B[:, :, -1, :] = B[:, :, -3, :]
                B[:, :, 0, :] = B[:, :, 2, :]
                B[:, :, 1, :] = B[:, :, 2, :]
            case 3:
                B[:, :, :, 2:-2] = (
                    -(A[:, :, :, 4:] - A[:, :, :, :-4])
                    + 8 * (A[:, :, :, 3:-1] - A[:, :, :, 1:-3])
                ) / (12 * delta[k])
                B[:, :, :, 0] = B[:, :, :, 2]
                B[:, :, :, 1] = B[:, :, :, 2]
                B[:, :, :, -2] = B[:, :, :, -3]
                B[:, :, :, -1] = B[:, :, :, -3]

    return B
