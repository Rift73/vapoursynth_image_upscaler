"""
Dark theme configuration for the GUI.
"""

from __future__ import annotations

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor


def configure_dark_palette(app: QApplication) -> None:
    """
    Apply a dark color palette to the application.

    Args:
        app: The QApplication instance to configure.
    """
    palette = QPalette()

    # Window backgrounds
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(240, 240, 240))

    # Input fields and lists
    palette.setColor(QPalette.Base, QColor(45, 45, 45))
    palette.setColor(QPalette.AlternateBase, QColor(37, 37, 37))

    # Tooltips
    palette.setColor(QPalette.ToolTipBase, QColor(240, 240, 240))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))

    # Text
    palette.setColor(QPalette.Text, QColor(240, 240, 240))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))

    # Buttons
    palette.setColor(QPalette.Button, QColor(37, 37, 37))
    palette.setColor(QPalette.ButtonText, QColor(240, 240, 240))

    # Links
    palette.setColor(QPalette.Link, QColor(80, 160, 255))

    # Selection
    palette.setColor(QPalette.Highlight, QColor(70, 130, 180))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))

    app.setPalette(palette)
