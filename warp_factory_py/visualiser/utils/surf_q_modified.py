import numpy as np
import matplotlib.pyplot as plt


def sqm(
    *args,
    ax=None,
    dark=True,
    cmap=None,
    figsize=(9, 7),
    elev=28,
    azim=-135,
    title=None,
    xlabel="x",
    ylabel="y",
    zlabel="z",
    colorbar=True,
    edgecolor="none",
    linewidth=0,
    antialiased=True,
    pane_alpha=0.08,
    grid_alpha=0.15,
    background_color=None,
    title_pad=20,
    xlabel_pad=4,
    ylabel_pad=4,
    zlabel_pad=6,
    tick_pad=1,
    title_fontsize=10,
    label_fontsize=8,
    tick_fontsize=7,
    box_aspect=(1, 1, 0.5),
    **kwargs
):
    """
    A more aesthetic MATLAB-like surf wrapper for matplotlib.

    Usage:
        surf_q_modified(Z)
        surf_q_modified(X, Y, Z)
        surf_q_modified(..., ax=ax)
    """

    # ---------- Theme defaults ----------
    if dark:
        if cmap is None:
            cmap = "magma"
        if background_color is None:
            background_color = "#0e1117"
        text_color = "#e6edf3"
        grid_color = "white"
        pane_color = (1, 1, 1, pane_alpha)
        cbar_tick_color = "#e6edf3"
    else:
        if cmap is None:
            cmap = "viridis"
        if background_color is None:
            background_color = "white"
        text_color = "black"
        grid_color = "#808080"
        pane_color = (0, 0, 0, pane_alpha)
        cbar_tick_color = "black"

    # ---------- Create axes if needed ----------
    created_ax = ax is None
    if created_ax:
        fig = plt.figure(figsize=figsize, facecolor=background_color)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
        fig.patch.set_facecolor(background_color)

    ax.set_facecolor(background_color)

    # ---------- Detect call pattern ----------
    if (
        len(args) >= 3
        and np.size(args[0]) > 3
        and np.size(args[1]) > 3
        and np.size(args[2]) > 3
        and isinstance(args[1], (float, int, np.floating, np.ndarray))
        and isinstance(args[2], (float, int, np.floating, np.ndarray))
    ):
        x = np.asarray(args[0])
        y = np.asarray(args[1])
        z = np.squeeze(np.asarray(args[2]))
    else:
        z = np.squeeze(np.asarray(args[0])).T
        n_y, n_x = z.shape
        x, y = np.meshgrid(np.arange(n_x), np.arange(n_y))

    # ---------- Surface ----------
    surf = ax.plot_surface(
        x,
        y,
        z,
        cmap=cmap,
        edgecolor=edgecolor,
        linewidth=linewidth,
        antialiased=antialiased,
        **kwargs
    )

    # ---------- Axes styling ----------
    ax.view_init(elev=elev, azim=azim)

    ax.set_title(
        title if title is not None else "",
        color=text_color,
        pad=title_pad,
        fontsize=title_fontsize
    )
    ax.set_xlabel(
        xlabel,
        color=text_color,
        labelpad=xlabel_pad,
        fontsize=label_fontsize
    )
    ax.set_ylabel(
        ylabel,
        color=text_color,
        labelpad=ylabel_pad,
        fontsize=label_fontsize
    )
    ax.set_zlabel(
        zlabel,
        color=text_color,
        labelpad=zlabel_pad,
        fontsize=label_fontsize
    )

    ax.tick_params(
        colors=text_color,
        which="both",
        pad=tick_pad,
        labelsize=tick_fontsize
    )

    try:
        ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    except Exception:
        pass

    try:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-2, 2))
        ax.zaxis.set_major_formatter(formatter)
    except Exception:
        pass

    try:
        offset = ax.zaxis.get_offset_text()
        offset.set_color(text_color)
        offset.set_fontsize(7)
    except Exception:
        pass

    # ---------- 3D pane colors ----------
    try:
        ax.xaxis.set_pane_color(pane_color)
        ax.yaxis.set_pane_color(pane_color)
        ax.zaxis.set_pane_color(pane_color)
    except Exception:
        pass

    # ---------- Grid styling ----------
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        try:
            axis._axinfo["grid"]["color"] = (
                *plt.matplotlib.colors.to_rgb(grid_color),
                grid_alpha
            )
            axis._axinfo["grid"]["linewidth"] = 0.8
        except Exception:
            pass

    # ---------- Improve aspect feel ----------
    try:
        ax.set_box_aspect(box_aspect)
    except Exception:
        pass

    # ---------- Colorbar ----------
    if colorbar:
        cbar = fig.colorbar(surf, ax=ax, shrink=0.72, pad=0.08)
        cbar.ax.yaxis.set_tick_params(color=cbar_tick_color, labelsize=tick_fontsize)
        plt.setp(cbar.ax.get_yticklabels(), color=cbar_tick_color)
        cbar.outline.set_edgecolor(cbar_tick_color)
        cbar.ax.set_facecolor(background_color)

    if created_ax:
        plt.show()

    return surf