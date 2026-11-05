from array_api_compat import array_namespace


def take_finite_difference_2(A, k_1, k_2, delta):
    """Approximate a second derivative with fourth-order differences.

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
    Axes shorter than five samples return zeros.
    Boundary cells copy the nearest interior derivative; mixed
    derivatives leave boundary faces zero.
    """
    if getattr(getattr(A, "device", None), "type", None) == "privateuseone":
        from warplab.gpu.utils.directml_derivatives import derivative

        return derivative(A, k_1, delta, 4, second_axis=k_2)

    s = A.shape
    xp = array_namespace(A)
    B = xp.zeros_like(
        A, dtype=A.dtype if xp.isdtype(A.dtype, "real floating") else xp.float64
    )

    if s[k_1] >= 5 and s[k_2] >= 5:
        if k_1 == k_2:
            match k_1:
                case 0:
                    B[2:-2, :, :, :] = (
                        -(A[4:, :, :, :] + A[:-4, :, :, :])
                        + 16 * (A[3:-1, :, :, :] + A[1:-3, :, :, :])
                        - 30 * A[2:-2, :, :, :]
                    ) / (12 * (delta[k_1] ** 2))
                    B[0, :, :, :] = B[2, :, :, :]
                    B[1, :, :, :] = B[2, :, :, :]
                    B[-2, :, :, :] = B[-3, :, :, :]
                    B[-1, :, :, :] = B[-3, :, :, :]
                case 1:
                    B[:, 2:-2, :, :] = (
                        -(A[:, 4:, :, :] + A[:, :-4, :, :])
                        + 16 * (A[:, 3:-1, :, :] + A[:, 1:-3, :, :])
                        - 30 * A[:, 2:-2, :, :]
                    ) / (12 * (delta[k_1] ** 2))
                    B[:, 0, :, :] = B[:, 2, :, :]
                    B[:, 1, :, :] = B[:, 2, :, :]
                    B[:, -2, :, :] = B[:, -3, :, :]
                    B[:, -1, :, :] = B[:, -3, :, :]
                case 2:
                    B[:, :, 2:-2, :] = (
                        -(A[:, :, 4:, :] + A[:, :, :-4, :])
                        + 16 * (A[:, :, 3:-1, :] + A[:, :, 1:-3, :])
                        - 30 * A[:, :, 2:-2, :]
                    ) / (12 * (delta[k_1] ** 2))
                    B[:, :, -2, :] = B[:, :, -3, :]
                    B[:, :, -1, :] = B[:, :, -3, :]
                    B[:, :, 0, :] = B[:, :, 2, :]
                    B[:, :, 1, :] = B[:, :, 2, :]
                case 3:
                    B[:, :, :, 2:-2] = (
                        -(A[:, :, :, 4:] + A[:, :, :, :-4])
                        + 16 * (A[:, :, :, 3:-1] + A[:, :, :, 1:-3])
                        - 30 * A[:, :, :, 2:-2]
                    ) / (12 * (delta[k_1] ** 2))
                    B[:, :, :, 0] = B[:, :, :, 2]
                    B[:, :, :, 1] = B[:, :, :, 2]
                    B[:, :, :, -2] = B[:, :, :, -3]
                    B[:, :, :, -1] = B[:, :, :, -3]
        else:
            k_L = max(k_1, k_2)
            k_S = min(k_1, k_2)

            x2 = slice(4, s[k_S])
            x1 = slice(3, s[k_S] - 1)
            x0 = slice(2, s[k_S] - 2)
            x_n1 = slice(1, s[k_S] - 3)
            x_n2 = slice(0, s[k_S] - 4)

            y2 = slice(4, s[k_L])
            y1 = slice(3, s[k_L] - 1)
            y0 = slice(2, s[k_L] - 2)
            y_n1 = slice(1, s[k_L] - 3)
            y_n2 = slice(0, s[k_L] - 4)

            match k_S:
                case 0:
                    match k_L:
                        case 1:
                            B[x0, y0, :, :] = (
                                1 / (12**2 * delta[k_L] * delta[k_S])
                            ) * (
                                -(
                                    -(A[x2, y2, :, :] - A[x_n2, y2, :, :])
                                    + 8 * (A[x1, y2, :, :] - A[x_n1, y2, :, :])
                                )
                                + (
                                    -(A[x2, y_n2, :, :] - A[x_n2, y_n2, :, :])
                                    + 8 * (A[x1, y_n2, :, :] - A[x_n1, y_n2, :, :])
                                )
                                + 8
                                * (
                                    -(A[x2, y1, :, :] - A[x_n2, y1, :, :])
                                    + 8 * (A[x1, y1, :, :] - A[x_n1, y1, :, :])
                                )
                                - 8
                                * (
                                    -(A[x2, y_n1, :, :] - A[x_n2, y_n1, :, :])
                                    + 8 * (A[x1, y_n1, :, :] - A[x_n1, y_n1, :, :])
                                )
                            )

                        case 2:
                            B[x0, :, y0, :] = (
                                1 / (12**2 * delta[k_L] * delta[k_S])
                            ) * (
                                -(
                                    -(A[x2, :, y2, :] - A[x_n2, :, y2, :])
                                    + 8 * (A[x1, :, y2, :] - A[x_n1, :, y2, :])
                                )
                                + (
                                    -(A[x2, :, y_n2, :] - A[x_n2, :, y_n2, :])
                                    + 8 * (A[x1, :, y_n2, :] - A[x_n1, :, y_n2, :])
                                )
                                + 8
                                * (
                                    -(A[x2, :, y1, :] - A[x_n2, :, y1, :])
                                    + 8 * (A[x1, :, y1, :] - A[x_n1, :, y1, :])
                                )
                                - 8
                                * (
                                    -(A[x2, :, y_n1, :] - A[x_n2, :, y_n1, :])
                                    + 8 * (A[x1, :, y_n1, :] - A[x_n1, :, y_n1, :])
                                )
                            )

                        case 3:
                            B[x0, :, :, y0] = (
                                1 / (12**2 * delta[k_L] * delta[k_S])
                            ) * (
                                -(
                                    -(A[x2, :, :, y2] - A[x_n2, :, :, y2])
                                    + 8 * (A[x1, :, :, y2] - A[x_n1, :, :, y2])
                                )
                                + (
                                    -(A[x2, :, :, y_n2] - A[x_n2, :, :, y_n2])
                                    + 8 * (A[x1, :, :, y_n2] - A[x_n1, :, :, y_n2])
                                )
                                + 8
                                * (
                                    -(A[x2, :, :, y1] - A[x_n2, :, :, y1])
                                    + 8 * (A[x1, :, :, y1] - A[x_n1, :, :, y1])
                                )
                                - 8
                                * (
                                    -(A[x2, :, :, y_n1] - A[x_n2, :, :, y_n1])
                                    + 8 * (A[x1, :, :, y_n1] - A[x_n1, :, :, y_n1])
                                )
                            )

                case 1:
                    match k_L:
                        case 2:
                            B[:, x0, y0, :] = (
                                1 / (12**2 * delta[k_L] * delta[k_S])
                            ) * (
                                -(
                                    -(A[:, x2, y2, :] - A[:, x_n2, y2, :])
                                    + 8 * (A[:, x1, y2, :] - A[:, x_n1, y2, :])
                                )
                                + (
                                    -(A[:, x2, y_n2, :] - A[:, x_n2, y_n2, :])
                                    + 8 * (A[:, x1, y_n2, :] - A[:, x_n1, y_n2, :])
                                )
                                + 8
                                * (
                                    -(A[:, x2, y1, :] - A[:, x_n2, y1, :])
                                    + 8 * (A[:, x1, y1, :] - A[:, x_n1, y1, :])
                                )
                                - 8
                                * (
                                    -(A[:, x2, y_n1, :] - A[:, x_n2, y_n1, :])
                                    + 8 * (A[:, x1, y_n1, :] - A[:, x_n1, y_n1, :])
                                )
                            )

                        case 3:
                            B[:, x0, :, y0] = (
                                1 / (12**2 * delta[k_L] * delta[k_S])
                            ) * (
                                -(
                                    -(A[:, x2, :, y2] - A[:, x_n2, :, y2])
                                    + 8 * (A[:, x1, :, y2] - A[:, x_n1, :, y2])
                                )
                                + (
                                    -(A[:, x2, :, y_n2] - A[:, x_n2, :, y_n2])
                                    + 8 * (A[:, x1, :, y_n2] - A[:, x_n1, :, y_n2])
                                )
                                + 8
                                * (
                                    -(A[:, x2, :, y1] - A[:, x_n2, :, y1])
                                    + 8 * (A[:, x1, :, y1] - A[:, x_n1, :, y1])
                                )
                                - 8
                                * (
                                    -(A[:, x2, :, y_n1] - A[:, x_n2, :, y_n1])
                                    + 8 * (A[:, x1, :, y_n1] - A[:, x_n1, :, y_n1])
                                )
                            )

                case 2:
                    match k_L:
                        case 3:
                            B[:, :, x0, y0] = (
                                1 / (12**2 * delta[k_L] * delta[k_S])
                            ) * (
                                -(
                                    -(A[:, :, x2, y2] - A[:, :, x_n2, y2])
                                    + 8 * (A[:, :, x1, y2] - A[:, :, x_n1, y2])
                                )
                                + (
                                    -(A[:, :, x2, y_n2] - A[:, :, x_n2, y_n2])
                                    + 8 * (A[:, :, x1, y_n2] - A[:, :, x_n1, y_n2])
                                )
                                + 8
                                * (
                                    -(A[:, :, x2, y1] - A[:, :, x_n2, y1])
                                    + 8 * (A[:, :, x1, y1] - A[:, :, x_n1, y1])
                                )
                                - 8
                                * (
                                    -(A[:, :, x2, y_n1] - A[:, :, x_n2, y_n1])
                                    + 8 * (A[:, :, x1, y_n1] - A[:, :, x_n1, y_n1])
                                )
                            )

    return B
