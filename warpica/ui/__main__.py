"""Start the desktop interface with ``python -m warpica.ui``."""

import sys

import matplotlib
import matplotlib.style

# Draw figures into our own Qt widgets, never into separate pyplot windows.
matplotlib.use("Agg")

# Dark figures with the plasma colour map, for every plot the interface draws,
# including the ones made by ``warpica.visualiser``. Set before any figure exists.
matplotlib.style.use("dark_background")
matplotlib.rcParams["image.cmap"] = "plasma"

from PySide6.QtWidgets import QApplication  # noqa: E402

from warpica.ui.theme import apply_dark_theme  # noqa: E402
from warpica.ui.windows.main_window import MainWindow  # noqa: E402


def main() -> int:
    """Open the main window and run until it is closed.

    Returns
    -------
    int
        Exit status for the operating system.
    """
    app = QApplication(sys.argv)
    # Before any window exists, so the matplotlib toolbar picks the light icons.
    apply_dark_theme(app)
    window = MainWindow()
    window.resize(1500, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
