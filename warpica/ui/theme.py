"""The dark colour scheme of the interface."""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Greys for the window, for the boxes the user types in, and for the text.
_WINDOW = QColor(30, 30, 36)
_BASE = QColor(20, 20, 26)
_BUTTON = QColor(45, 45, 54)
_TEXT = QColor(230, 230, 235)
_DISABLED_TEXT = QColor(120, 120, 130)

# A purple from the plasma colour map, used for the selected item and focus.
_HIGHLIGHT = QColor(156, 23, 158)


def apply_dark_theme(app: QApplication) -> None:
    """Give the whole application a dark palette.

    The Fusion style is used because it follows the palette on every platform,
    unlike the native Windows style, which ignores most of it.

    Parameters
    ----------
    app : QApplication
        The running application.
    """
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, _WINDOW)
    palette.setColor(QPalette.ColorRole.WindowText, _TEXT)
    palette.setColor(QPalette.ColorRole.Base, _BASE)
    palette.setColor(QPalette.ColorRole.AlternateBase, _WINDOW)
    palette.setColor(QPalette.ColorRole.Text, _TEXT)
    palette.setColor(QPalette.ColorRole.Button, _BUTTON)
    palette.setColor(QPalette.ColorRole.ButtonText, _TEXT)
    palette.setColor(QPalette.ColorRole.ToolTipBase, _BUTTON)
    palette.setColor(QPalette.ColorRole.ToolTipText, _TEXT)
    palette.setColor(QPalette.ColorRole.Highlight, _HIGHLIGHT)
    palette.setColor(QPalette.ColorRole.HighlightedText, _TEXT)

    # Greyed-out text must stay readable but clearly inactive, for example the
    # locked stages in the sidebar.
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(QPalette.ColorGroup.Disabled, role, _DISABLED_TEXT)

    app.setPalette(palette)
