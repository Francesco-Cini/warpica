from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from warpica._typing import Array, TensorDict
from warpica.solver.verify_tensor import verify_tensor
from warpica.visualiser.utils.get_slice_data import get_slice_data
from warpica.visualiser.utils.plot_component import plot_component


def _slice_setup(tensor, sliced_planes, slice_locations):
    """Resolve a plot slice into an index, axis coordinates and axis labels.

    Returns `(indices, coords, labels)` for the two axes that stay free.
    """
    # sliced_planes names the two axes to FIX (default: time and z), so the plot
    # shows the other two (x and y by default). Axis numbers and locations are
    # one-based; get_slice_data turns them into an index that fixes those axes.
    planes = (1, 4) if sliced_planes is None else sliced_planes
    indices = get_slice_data(planes, slice_locations, tensor)
    axes = [i for i, index in enumerate(indices) if isinstance(index, slice)]
    shape = np.shape(tensor["tensor"][0][0])
    spacing = tensor.get("scaling")
    # Generators record their world centre in params; without one, use 0.
    centre = tensor.get("params", {}).get("world_centre", np.zeros(4))
    coords, labels = [], []
    for axis in axes:
        label = ("t", "x", "y", "z")[axis]
        # Axis coordinates: physical position (index + 1) * spacing - centre, the same
        # convention as the generators, or one-based grid indices without a spacing.
        if spacing is None:
            coords.append(np.arange(1, shape[axis] + 1))
            labels.append(f"{label} (grid index)")
        else:
            coords.append((np.arange(shape[axis]) + 1) * spacing[axis] - centre[axis])
            labels.append(f"{label} ({'s' if axis == 0 else 'm'})")
    return indices, coords, labels


def _component_title(symbol, index, i, j):
    """Return the LaTeX title of component (i, j) for the given index type."""
    if index == "covariant":
        return rf"${symbol}_{{{i}{j}}}$"
    if index == "contravariant":
        return rf"${symbol}^{{{i}{j}}}$"
    if index == "mixedupdown":
        return rf"${symbol}^{{{i}}}_{{{j}}}$"
    return rf"${symbol}_{{{i}}}^{{{j}}}$"


def plot_tensor(
    tensor: TensorDict,
    alpha: float | None = None,
    sliced_planes: Sequence[int] | Array | None = None,
    slice_locations: Sequence[float] | Array | None = None,
) -> tuple[Figure, Array]:
    """Plot a two-dimensional slice of all sixteen tensor components.

    Parameters
    ----------
    tensor : dict
        Metric or stress-energy dictionary with 4D component fields.
    alpha : float or None, optional
        Plot opacity between zero and one. None selects full opacity.
    sliced_planes : array_like of int, shape (2,), optional
        Axes to fix, using 1=t, 2=x, 3=y, 4=z. Defaults to (1, 4).
    slice_locations : array_like, shape (2,), optional
        One-based grid locations, rounded to the nearest cell.
        Defaults to the middle cell on each fixed axis.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the component plots; not shown automatically.
    axes : ndarray, shape (4, 4)
        Matplotlib axes arranged by tensor component.

    Notes
    -----
    Coordinates use SI grid scaling if available, otherwise one-based grid
    indices. Call figure.savefig(...) or plt.show() to display the result.
    """
    if not verify_tensor(tensor, 1):
        raise ValueError("Invalid tensor")
    indices, coords, labels = _slice_setup(tensor, sliced_planes, slice_locations)
    fig, axes = plt.subplots(4, 4, figsize=(13, 11), layout="constrained")
    symbol = "g" if tensor["type"].lower() == "metric" else "T"
    # One panel per component, in a 4x4 grid; each panel has its own colour scale.
    for i in range(4):
        for j in range(4):
            plot_component(
                np.asarray(tensor["tensor"][i][j])[indices],
                _component_title(symbol, tensor["index"].lower(), i, j),
                labels[0],
                labels[1],
                alpha=1 if alpha is None else alpha,
                ax=axes[i, j],
                x=coords[0],
                y=coords[1],
            )
    # Equal aspect only when both plotted axes are spatial; a time axis is in
    # seconds, so its scale is not comparable with metres.
    if all(not label.startswith("t ") for label in labels):
        for ax in axes.flat:
            ax.set_aspect("equal")
    fig.suptitle(tensor.get("name", "Tensor"))
    return fig, axes
