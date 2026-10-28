import matplotlib.pyplot as plt
import numpy as np

from warplab.analyser.change_tensor_index import change_tensor_index
from warplab.metrics.three_plus_one_decomposer import three_plus_one_decomposer
from warplab.visualiser.plot_tensor import _slice_setup
from warplab.visualiser.utils.plot_component import plot_component


def plot_three_plus_one(metric, sliced_planes=None, slice_locations=None, alpha=None):
    """Plot slices of the lapse, covariant shift, and spatial metric.

    Parameters
    ----------
    metric : dict
        Metric dictionary with components on a (t, x, y, z) grid.
    sliced_planes : array_like of int, shape (2,), optional
        Axes to fix, using 1=t, 2=x, 3=y, 4=z. Defaults to (1, 4).
    slice_locations : array_like, shape (2,), optional
        One-based grid locations, rounded to the nearest cell.
        Defaults to the middle cell on each fixed axis.
    alpha : float or None, optional
        Plot opacity between zero and one. None selects full opacity.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the component plots; not shown automatically.
    axes : ndarray, shape (2, 5)
        Matplotlib axes for the ten independent 3+1 fields.
    """
    metric = change_tensor_index(metric, "covariant")
    indices, coords, labels = _slice_setup(metric, sliced_planes, slice_locations)
    lapse, beta, gamma, _, _ = three_plus_one_decomposer(metric)
    components = [(r"$\alpha$ (lapse)", lapse)]
    components += [(rf"$\beta_{{{i + 1}}}$", beta[i]) for i in range(3)]
    components += [
        (rf"$\gamma_{{{i + 1}{j + 1}}}$", gamma[i][j])
        for i in range(3)
        for j in range(i, 3)
    ]
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), layout="constrained")
    for ax, (title, field) in zip(axes.flat, components):
        plot_component(
            np.asarray(field)[indices],
            title,
            *labels,
            alpha=1 if alpha is None else alpha,
            ax=ax,
            x=coords[0],
            y=coords[1],
        )
    if all(not label.startswith("t ") for label in labels):
        for ax in axes.flat:
            ax.set_aspect("equal")
    fig.suptitle(metric.get("name", "Metric") + " - 3+1 decomposition")
    return fig, axes


def plotThreePlusOne(metric, slicedPlanes=None, sliceLocations=None, alpha=None):
    """Compatibility alias for the original MATLAB-style name."""
    return plot_three_plus_one(metric, slicedPlanes, sliceLocations, alpha)
