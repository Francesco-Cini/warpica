from array_api_compat import array_namespace


def take_finite_difference_2_2(A, k_1, k_2, delta):
    """Approximate a second derivative with second-order differences.

    Parameters
    ----------
    A : array_like, shape (nt, nx, ny, nz)
        Scalar field to differentiate; allocation follows its array backend.
    k_1, k_2 : int
        Zero-based differentiation axes; equal axes give an unmixed derivative.
    delta : array_like, shape (4,)
        Grid spacings in (t, x, y, z) order.

    Returns
    -------
    B : array
        Derivative field with the same shape and backend as A.

    Notes
    -----
    Axes shorter than three samples return zeros.
    Boundary cells copy the nearest interior derivative; mixed
    derivatives leave boundary faces zero.
    """
    # DirectML tensors take the dedicated GPU derivative kernel instead.
    if getattr(getattr(A, "device", None), "type", None) == "privateuseone":
        from warplab.gpu.utils.directml_derivatives import derivative

        return derivative(A, k_1, delta, 2, second_axis=k_2)

    s = A.shape
    xp = array_namespace(A)
    B = xp.zeros_like(
        A, dtype=A.dtype if xp.isdtype(A.dtype, "real floating") else xp.float64
    )

    # Second-order second derivative, needing 3 samples on both axes. Same axis:
    # (A[i+1] - 2 A[i] + A[i-1]) / h^2, with the first and last cell copying their
    # neighbour. Different axes: the four-corner formula
    # (A[+,+] - A[+,-] - A[-,+] + A[-,-]) / (4 h_S h_L), whose edge cells stay 0.
    if s[k_1] >= 3 and s[k_2] >= 3:
        if k_1 == k_2:
            match k_1:
                case 0:
                    B[1:-1, :, :, :] = (
                        A[2:, :, :, :] - 2 * A[1:-1, :, :, :] + A[:-2, :, :, :]
                    ) / (delta[k_1] ** 2)
                    B[0, :, :, :] = B[1, :, :, :]
                    B[-1, :, :, :] = B[-2, :, :, :]
                case 1:
                    B[:, 1:-1, :, :] = (
                        A[:, 2:, :, :] - 2 * A[:, 1:-1, :, :] + A[:, :-2, :, :]
                    ) / (delta[k_1] ** 2)
                    B[:, 0, :, :] = B[:, 1, :, :]
                    B[:, -1, :, :] = B[:, -2, :, :]
                case 2:
                    B[:, :, 1:-1, :] = (
                        A[:, :, 2:, :] - 2 * A[:, :, 1:-1, :] + A[:, :, :-2, :]
                    ) / (delta[k_1] ** 2)
                    B[:, :, 0, :] = B[:, :, 1, :]
                    B[:, :, -1, :] = B[:, :, -2, :]
                case 3:
                    B[:, :, :, 1:-1] = (
                        A[:, :, :, 2:] - 2 * A[:, :, :, 1:-1] + A[:, :, :, :-2]
                    ) / (delta[k_1] ** 2)
                    B[:, :, :, 0] = B[:, :, :, 1]
                    B[:, :, :, -1] = B[:, :, :, -2]
        else:
            k_L = max(k_1, k_2)
            k_S = min(k_1, k_2)

            x1 = slice(2, s[k_S])
            x0 = slice(1, s[k_S] - 1)
            x_1 = slice(0, s[k_S] - 2)

            y1 = slice(2, s[k_L])
            y0 = slice(1, s[k_L] - 1)
            y_1 = slice(0, s[k_L] - 2)

            match k_S:
                case 0:
                    match k_L:
                        case 1:
                            B[x0, y0, :, :] = (
                                A[x_1, y_1, :, :]
                                - A[x_1, y1, :, :]
                                - A[x1, y_1, :, :]
                                + A[x1, y1, :, :]
                            ) / (4 * delta[k_L] * delta[k_S])
                        case 2:
                            B[x0, :, y0, :] = (
                                A[x_1, :, y_1, :]
                                - A[x_1, :, y1, :]
                                - A[x1, :, y_1, :]
                                + A[x1, :, y1, :]
                            ) / (4 * delta[k_L] * delta[k_S])
                        case 3:
                            B[x0, :, :, y0] = (
                                A[x_1, :, :, y_1]
                                - A[x_1, :, :, y1]
                                - A[x1, :, :, y_1]
                                + A[x1, :, :, y1]
                            ) / (4 * delta[k_L] * delta[k_S])
                case 1:
                    match k_L:
                        case 2:
                            B[:, x0, y0, :] = (
                                A[:, x_1, y_1, :]
                                - A[:, x_1, y1, :]
                                - A[:, x1, y_1, :]
                                + A[:, x1, y1, :]
                            ) / (4 * delta[k_L] * delta[k_S])
                        case 3:
                            B[:, x0, :, y0] = (
                                A[:, x_1, :, y_1]
                                - A[:, x_1, :, y1]
                                - A[:, x1, :, y_1]
                                + A[:, x1, :, y1]
                            ) / (4 * delta[k_L] * delta[k_S])
                case 2:
                    match k_L:
                        case 3:
                            B[:, :, x0, y0] = (
                                A[:, :, x_1, y_1]
                                - A[:, :, x_1, y1]
                                - A[:, :, x1, y_1]
                                + A[:, :, x1, y1]
                            ) / (4 * delta[k_L] * delta[k_S])

    return B
