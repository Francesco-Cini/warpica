"""Start the desktop interface with ``python -m warpica.ui``."""

import sys

import matplotlib

# Draw figures into our own Qt widgets, never into separate pyplot windows.
matplotlib.use("Agg")

from PySide6.QtWidgets import QApplication  # noqa: E402

from warpica.ui.windows.main_window import MainWindow  # noqa: E402


def main() -> int:
    """Open the main window and run until it is closed.

    Returns
    -------
    int
        Exit status for the operating system.
    """
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1500, 900)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
