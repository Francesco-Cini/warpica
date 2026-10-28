import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def plot_component(
    array, title_text, x_label_text, y_label_text, alpha=1.0, *, ax=None, x=None, y=None
):
    """Plot a two-dimensional component as a signed heatmap.

    Parameters
    ----------
    array : array_like, ndim=2
        Finite scalar values; the first axis corresponds to x.
    title_text : str
        Plot title.
    x_label_text, y_label_text : str
        Coordinate axis labels.
    alpha : float, optional
        Opacity between zero and one, defaulting to one.
    ax : matplotlib.axes.Axes or None, optional
        Existing axes; None creates a figure and axes.
    x, y : array_like or None, optional
        Coordinates along the array axes. None uses one-based grid indices.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the component plots; not shown automatically.
    axes : matplotlib.axes.Axes
        Axes containing the heatmap.

    Notes
    -----
    Colour limits are symmetric about zero so that sign remains visible.
    """
    values = np.asarray(array)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Component must be a finite 2D array")
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    x = np.arange(1, values.shape[0] + 1) if x is None else np.asarray(x)
    y = np.arange(1, values.shape[1] + 1) if y is None else np.asarray(y)
    limit = float(np.max(np.abs(values))) or 1.0
    mesh = ax.pcolormesh(
        x,
        y,
        values.T,
        shading="nearest",
        cmap="RdBu_r",
        norm=Normalize(-limit, limit),
        alpha=alpha,
        rasterized=True,
    )
    ax.set(title=title_text, xlabel=x_label_text, ylabel=y_label_text)
    ax.figure.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    return ax.figure, ax
