"""Functional stencils avoid DirectML's incorrect strided slice assignments.

Keep each differentiated axis contiguous, then concatenate boundary cells.
No CPU round trip is used for the derivative calculation.
"""

import torch


def _interior(value, axis, coefficients, radius, denominator):
    permutation = [i for i in range(value.ndim) if i != axis] + [axis]
    inverse = [permutation.index(i) for i in range(value.ndim)]
    work = value.permute(permutation).contiguous()
    length = work.shape[-1] - 2 * radius
    result = torch.zeros_like(work[..., radius : radius + length])
    for offset, coefficient in coefficients:
        result = (
            result + coefficient * work[..., radius + offset : radius + offset + length]
        )
    return (result / float(denominator)).permute(inverse).contiguous()


def _boundary(value, axis, radius, copy):
    permutation = [i for i in range(value.ndim) if i != axis] + [axis]
    inverse = [permutation.index(i) for i in range(value.ndim)]
    work = value.permute(permutation).contiguous()
    low, high = work[..., :1].contiguous(), work[..., -1:].contiguous()
    if not copy:
        low, high = torch.zeros_like(low), torch.zeros_like(high)
    return (
        torch.cat([low] * radius + [work] + [high] * radius, dim=-1)
        .permute(inverse)
        .contiguous()
    )


def derivative(value, k, delta, order, second_axis=None):
    """Apply a finite-difference stencil without strided device assignment.

    Parameters
    ----------
    value : torch.Tensor
        Scalar field on a (t, x, y, z) grid.
    k : int
        Zero-based differentiation axis.
    delta : array_like, shape (4,)
        Grid spacings.
    order : {2, 4}
        Stencil accuracy order.
    second_axis : int or None, optional
        Second differentiation axis. None requests a first derivative.

    Returns
    -------
    result : torch.Tensor
        Derivative on the original device with the original shape.

    Notes
    -----
    Short axes yield zero derivatives. Unmixed derivatives copy interior
    values to the boundary; mixed derivatives keep boundary faces zero.
    """
    radius = 1 if order == 2 else 2
    axes = [k] if second_axis is None else [k, second_axis]
    if any(value.shape[axis] < 2 * radius + 1 for axis in axes):
        return torch.zeros_like(value)
    first = [(-1, -1), (1, 1)] if order == 2 else [(-2, 1), (-1, -8), (1, 8), (2, -1)]
    denominator = 2 if order == 2 else 12
    if second_axis is None:
        result = _interior(value, k, first, radius, denominator * delta[k])
        return _boundary(result, k, radius, True)
    if k == second_axis:
        coefficients = (
            [(-1, 1), (0, -2), (1, 1)]
            if order == 2
            else [(-2, -1), (-1, 16), (0, -30), (1, 16), (2, -1)]
        )
        result = _interior(
            value, k, coefficients, radius, (1 if order == 2 else 12) * delta[k] ** 2
        )
        return _boundary(result, k, radius, True)
    result = _interior(value, k, first, radius, denominator * delta[k])
    result = _interior(
        result, second_axis, first, radius, denominator * delta[second_axis]
    )
    result = _boundary(result, k, radius, False)
    return _boundary(result, second_axis, radius, False)
