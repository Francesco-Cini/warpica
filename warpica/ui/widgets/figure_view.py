"""A widget that shows a matplotlib figure together with its toolbar."""

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtWidgets import QScrollArea, QVBoxLayout, QWidget

# A figure is drawn no smaller than this fraction of the size it was created
# with. Below that its plots get cramped and the labels overlap.
_MIN_SCALE = 0.7


def _use_plasma(figure: Figure) -> None:
    """Give every colour-mapped plot in ``figure`` the plasma colour map.

    The plots in ``warpica.visualiser`` choose their own colour map, so the
    matplotlib default does not reach them. Changing the map of the image or
    mesh also updates the colour bar that belongs to it.
    """
    for axes in figure.axes:
        for artist in (*axes.images, *axes.collections):
            # Only artists that map data to colour have a colour map.
            if artist.get_array() is not None:
                artist.set_cmap("plasma")


class FigureView(QWidget):
    """Show one matplotlib figure at a time, with zoom, pan and save buttons.

    Attributes
    ----------
    figure : matplotlib.figure.Figure or None
        The figure being shown, or None if the view is empty.
    """

    def __init__(self) -> None:
        super().__init__()
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        # The toolbar and scroll area currently on screen.
        self._widgets: list[QWidget] = []
        self.figure: Figure | None = None

    def show_figure(self, figure: Figure) -> None:
        """Replace whatever is shown with ``figure``.

        Parameters
        ----------
        figure : matplotlib.figure.Figure
            The figure to display. It can come from a ``pyplot`` call, such as
            the ones in ``warpica.visualiser``.
        """
        self.clear()

        # pyplot keeps every figure it creates alive until it is closed.
        # Closing it there releases it from pyplot's list. The figure itself
        # stays usable and this view now owns it.
        plt.close(figure)

        _use_plasma(figure)
        canvas = FigureCanvasQTAgg(figure)

        # Keep the figure at a readable size. If the window is too small to
        # fit it, the scroll area lets the user scroll instead of squeezing it.
        width, height = figure.get_size_inches() * plt.rcParams["figure.dpi"]
        canvas.setMinimumSize(int(width * _MIN_SCALE), int(height * _MIN_SCALE))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(canvas)

        toolbar = NavigationToolbar2QT(canvas, self)
        for widget in (toolbar, scroll):
            self._layout.addWidget(widget)
            self._widgets.append(widget)
        self.figure = figure

    def clear(self) -> None:
        """Remove the current figure and leave the view empty."""
        for widget in self._widgets:
            self._layout.removeWidget(widget)
            widget.deleteLater()
        self._widgets = []
        self.figure = None
