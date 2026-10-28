import matplotlib.pyplot as plt
import numpy as np


def surf_q(*args, ax=None, **kwargs):
    """Plot a MATLAB-style surface from Z or X, Y, Z arrays.

    Parameters
    ----------
    *args : array_like
        A height array Z, or coordinate arrays X, Y, Z. Additional positional
        arguments are passed to plot_surface. Z-only input is transposed.
    ax : matplotlib.axes.Axes or None, optional
        Existing 3D axes; None creates new axes.
    **kwargs : dict
        Keyword arguments forwarded to Matplotlib plot_surface.

    Returns
    -------
    surface : mpl_toolkits.mplot3d.art3d.Poly3DCollection
        Surface artist with an associated colourbar.
    """
    if (
        len(args) >= 3
        and np.size(args[0]) > 3
        and np.size(args[1]) > 3
        and np.size(args[2]) > 3
        and isinstance(args[1], (float, np.ndarray, np.floating, int))
        and isinstance(args[2], (float, np.ndarray, np.floating, int))
    ):
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        surf = ax.plot_surface(
            args[0], args[1], np.squeeze(args[2]), *args[3:], **kwargs
        )
        fig.colorbar(surf, ax=ax)
    else:
        z = np.squeeze(args[0]).T
        n_y, n_x = z.shape
        x, y = np.meshgrid(np.arange(n_x), np.arange(n_y))

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        else:
            fig = ax.figure

        surf = ax.plot_surface(x, y, z, *args[1:], **kwargs)
        fig.colorbar(surf, ax=ax)

    if ax is None:
        plt.show()

    return surf
