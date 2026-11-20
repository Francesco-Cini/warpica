"""Draw momentum flow lines over a slice of the energy density."""

from collections.abc import Sequence

import numpy as np
from matplotlib.figure import Figure

from warpica._typing import Array
from warpica.ui.core.analysis import FlowLines
from warpica.ui.core.slicing import AXIS_NAMES, PLANES, axis_extent, take_slice


def flow_line_figure(
    flow: FlowLines, axes: Sequence[Array], background: Array | None = None
) -> Figure:
    """Draw flow lines on the plane they were seeded on.

    Every line is drawn from its seed point, marked with a dot, with one arrow
    in the middle to show which way the momentum flows. The lines are traced in
    three dimensions, so they are drawn as seen from straight above the plane:
    a line that leaves the plane is shown by its shadow on it.

    Parameters
    ----------
    flow : FlowLines
        The lines to draw.
    axes : sequence of ndarray
        Coordinates of the grid points along t, x, y and z, which set the extent
        of the picture.
    background : ndarray, shape (nt, nx, ny, nz), optional
        A field to show faintly behind the lines, cut on the same plane, such
        as the energy density. Without it the plot area is left blank.

    Returns
    -------
    matplotlib.figure.Figure
        The finished figure. It is not attached to ``pyplot``.
    """
    first, second, fixed = PLANES[flow.plane]
    x_axis, y_axis = axes[first], axes[second]
    x_edges, y_edges = axis_extent(x_axis), axis_extent(y_axis)

    figure = Figure()
    plot = figure.add_subplot()

    if background is not None:
        values = take_slice(background, flow.plane, flow.position, flow.time_index)
        # Centre the colour scale on zero. The limit is the largest finite
        # value, or 1 if there is none, which keeps the scale valid.
        finite = np.abs(values[np.isfinite(values)])
        limit = float(finite.max()) if finite.size and finite.max() > 0 else 1.0
        plot.imshow(
            values.T,
            origin="lower",
            extent=(*x_edges, *y_edges),
            cmap="plasma",
            vmin=-limit,
            vmax=limit,
            alpha=0.45,
        )

    # Path columns are the x, y and z coordinates, so grid axis a (1, 2 or 3)
    # is column a - 1.
    for path in flow.paths:
        horizontal, vertical = path[:, first - 1], path[:, second - 1]
        plot.plot(horizontal, vertical, color="white", linewidth=1)
        plot.plot(horizontal[0], vertical[0], "o", color="white", markersize=3)
        if len(path) >= 3:
            # One arrow on the middle segment shows the direction of flow.
            middle = len(path) // 2
            plot.annotate(
                "",
                xy=(horizontal[middle + 1], vertical[middle + 1]),
                xytext=(horizontal[middle], vertical[middle]),
                arrowprops={"arrowstyle": "->", "color": "white", "linewidth": 1},
            )

    plot.set_xlim(x_edges)
    plot.set_ylim(y_edges)
    plot.set_aspect("equal")
    plot.set_xlabel(f"{AXIS_NAMES[first]} (m)")
    plot.set_ylabel(f"{AXIS_NAMES[second]} (m)")
    plot.set_title(
        f"Momentum flow lines, seeded on the {flow.plane} plane at "
        f"{AXIS_NAMES[fixed]} = {axes[fixed][flow.position]:.3g} m",
        fontsize=10,
    )
    return figure
