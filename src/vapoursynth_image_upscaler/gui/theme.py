"""
Theme configuration for the GUI.

Provides multiple themes with different visual styles:
- Dark: Dark theme with subtle paper texture (default)
- Holo: Modern, sleek holographic theme with gradients and glow effects
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor

if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


# Available theme names
AVAILABLE_THEMES = ["Dark", "Holo"]
DEFAULT_THEME = "Dark"


class ThemeManager:
    """
    Manages application themes.

    Provides methods to apply different visual themes to the application.
    """

    _current_theme: str = DEFAULT_THEME
    _app: QApplication | None = None

    @classmethod
    def initialize(cls, app: QApplication, theme: str = DEFAULT_THEME) -> None:
        """
        Initialize the theme manager with the application instance.

        Args:
            app: The QApplication instance.
            theme: Initial theme name to apply.
        """
        cls._app = app
        cls.apply_theme(theme)

    @classmethod
    def get_current_theme(cls) -> str:
        """Get the name of the currently active theme."""
        return cls._current_theme

    @classmethod
    def apply_theme(cls, theme_name: str) -> None:
        """
        Apply a theme by name.

        Args:
            theme_name: Name of the theme to apply ("Dark" or "Holo").
        """
        if cls._app is None:
            return

        if theme_name not in AVAILABLE_THEMES:
            theme_name = DEFAULT_THEME

        cls._current_theme = theme_name

        if theme_name == "Dark":
            cls._apply_dark_theme()
        elif theme_name == "Holo":
            cls._apply_holo_theme()

    @classmethod
    def _apply_dark_theme(cls) -> None:
        """Apply the Dark theme with paper texture."""
        if cls._app is None:
            return

        palette = QPalette()

        # Window backgrounds - slightly warm dark with paper feel
        palette.setColor(QPalette.Window, QColor(32, 31, 30))
        palette.setColor(QPalette.WindowText, QColor(235, 230, 225))

        # Input fields and lists - subtle texture variation
        palette.setColor(QPalette.Base, QColor(42, 40, 38))
        palette.setColor(QPalette.AlternateBase, QColor(38, 36, 34))

        # Tooltips
        palette.setColor(QPalette.ToolTipBase, QColor(235, 230, 225))
        palette.setColor(QPalette.ToolTipText, QColor(20, 20, 20))

        # Text - warm white like aged paper
        palette.setColor(QPalette.Text, QColor(235, 230, 225))
        palette.setColor(QPalette.BrightText, QColor(255, 100, 80))

        # Buttons - paper-like with slight depth
        palette.setColor(QPalette.Button, QColor(48, 46, 44))
        palette.setColor(QPalette.ButtonText, QColor(235, 230, 225))

        # Links - muted warm blue
        palette.setColor(QPalette.Link, QColor(140, 170, 200))

        # Selection - warm sepia tone
        palette.setColor(QPalette.Highlight, QColor(100, 85, 70))
        palette.setColor(QPalette.HighlightedText, QColor(255, 250, 245))

        # Disabled text
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 115, 110))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 115, 110))

        cls._app.setPalette(palette)

        # Dark theme stylesheet with paper texture effect
        stylesheet = """
            QWidget {
                font-family: "Segoe UI", Arial, sans-serif;
            }

            QMainWindow, QDialog {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #201f1e,
                    stop: 0.3 #252423,
                    stop: 0.5 #222120,
                    stop: 0.7 #252423,
                    stop: 1 #1e1d1c
                );
            }

            QGroupBox {
                border: 1px solid #3a3836;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                background: rgba(45, 43, 41, 0.6);
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 8px;
                color: #b8b0a8;
            }

            QPushButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #403e3c,
                    stop: 0.4 #353331,
                    stop: 0.6 #302e2c,
                    stop: 1 #282624
                );
                border: 1px solid #4a4846;
                border-radius: 4px;
                padding: 6px 16px;
                color: #ebe6e1;
                min-height: 20px;
            }

            QPushButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4a4846,
                    stop: 0.4 #403e3c,
                    stop: 0.6 #3a3836,
                    stop: 1 #323030
                );
                border: 1px solid #5a5856;
            }

            QPushButton:pressed {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #282624,
                    stop: 1 #353331
                );
            }

            QPushButton:disabled {
                background: #2a2826;
                border: 1px solid #3a3836;
                color: #787470;
            }

            QLineEdit, QComboBox, QSpinBox {
                background: #2a2826;
                border: 1px solid #3a3836;
                border-radius: 3px;
                padding: 4px 8px;
                color: #ebe6e1;
                selection-background-color: #645546;
            }

            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border: 1px solid #6a6560;
            }

            QComboBox::drop-down {
                border: none;
                width: 20px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #ebe6e1;
                margin-right: 6px;
            }

            QComboBox QAbstractItemView {
                background: #2a2826;
                border: 1px solid #4a4846;
                selection-background-color: #645546;
            }

            QCheckBox {
                spacing: 6px;
                color: #ebe6e1;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #4a4846;
                border-radius: 3px;
                background: #2a2826;
            }

            QCheckBox::indicator:checked {
                background: #645546;
                border: 1px solid #7a6a5a;
            }

            QCheckBox::indicator:checked::after {
                content: "✓";
            }

            QProgressBar {
                border: 1px solid #3a3836;
                border-radius: 4px;
                background: #2a2826;
                text-align: center;
                color: #ebe6e1;
            }

            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #5a4a3a,
                    stop: 0.5 #6a5a4a,
                    stop: 1 #5a4a3a
                );
                border-radius: 3px;
            }

            QLabel {
                color: #ebe6e1;
            }

            QSlider::groove:horizontal {
                border: 1px solid #3a3836;
                height: 6px;
                background: #2a2826;
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                background: #5a5856;
                border: 1px solid #6a6866;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }

            QSlider::handle:horizontal:hover {
                background: #6a6866;
            }

            QToolTip {
                background: #ebe6e1;
                color: #201f1e;
                border: 1px solid #a09890;
                border-radius: 3px;
                padding: 4px;
            }

            QScrollBar:vertical {
                border: none;
                background: #2a2826;
                width: 10px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: #4a4846;
                border-radius: 5px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background: #5a5856;
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """

        cls._app.setStyleSheet(stylesheet)

    @classmethod
    def _apply_holo_theme(cls) -> None:
        """Apply the Holo theme - modern, sleek, holographic."""
        if cls._app is None:
            return

        palette = QPalette()

        # Window backgrounds - deep blue-black
        palette.setColor(QPalette.Window, QColor(15, 20, 30))
        palette.setColor(QPalette.WindowText, QColor(200, 220, 255))

        # Input fields - translucent dark
        palette.setColor(QPalette.Base, QColor(20, 30, 45))
        palette.setColor(QPalette.AlternateBase, QColor(25, 35, 50))

        # Tooltips
        palette.setColor(QPalette.ToolTipBase, QColor(30, 40, 60))
        palette.setColor(QPalette.ToolTipText, QColor(200, 220, 255))

        # Text - cool white with blue tint
        palette.setColor(QPalette.Text, QColor(200, 220, 255))
        palette.setColor(QPalette.BrightText, QColor(100, 200, 255))

        # Buttons - dark with holographic hints
        palette.setColor(QPalette.Button, QColor(30, 40, 55))
        palette.setColor(QPalette.ButtonText, QColor(200, 220, 255))

        # Links - cyan glow
        palette.setColor(QPalette.Link, QColor(0, 200, 255))

        # Selection - holographic cyan
        palette.setColor(QPalette.Highlight, QColor(0, 150, 200))
        palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

        # Disabled text
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(80, 100, 130))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(80, 100, 130))

        cls._app.setPalette(palette)

        # Holo theme stylesheet with holographic effects
        stylesheet = """
            QWidget {
                font-family: "Segoe UI", Arial, sans-serif;
            }

            QMainWindow, QDialog {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0a0f1a,
                    stop: 0.25 #0f1420,
                    stop: 0.5 #101828,
                    stop: 0.75 #0d1520,
                    stop: 1 #080d18
                );
            }

            QGroupBox {
                border: 1px solid qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #00a0c0,
                    stop: 0.5 #4060a0,
                    stop: 1 #8040c0
                );
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
                background: rgba(20, 30, 50, 0.7);
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 2px 10px;
                color: #60c0ff;
                background: rgba(0, 80, 120, 0.5);
                border-radius: 3px;
            }

            QPushButton {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(0, 100, 150, 0.8),
                    stop: 0.5 rgba(40, 80, 140, 0.8),
                    stop: 1 rgba(80, 60, 160, 0.8)
                );
                border: 1px solid qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #00c8ff,
                    stop: 0.5 #6080ff,
                    stop: 1 #c060ff
                );
                border-radius: 5px;
                padding: 6px 16px;
                color: #e0f0ff;
                min-height: 20px;
            }

            QPushButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 rgba(0, 140, 190, 0.9),
                    stop: 0.5 rgba(60, 100, 180, 0.9),
                    stop: 1 rgba(120, 80, 200, 0.9)
                );
                border: 1px solid #00e0ff;
            }

            QPushButton:pressed {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 rgba(0, 60, 100, 0.9),
                    stop: 1 rgba(0, 100, 150, 0.9)
                );
            }

            QPushButton:disabled {
                background: rgba(30, 40, 60, 0.6);
                border: 1px solid #304060;
                color: #506080;
            }

            QLineEdit, QComboBox, QSpinBox {
                background: rgba(15, 25, 40, 0.9);
                border: 1px solid #306080;
                border-radius: 4px;
                padding: 4px 8px;
                color: #c8dcff;
                selection-background-color: #0090c0;
            }

            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border: 1px solid #00c8ff;
                background: rgba(20, 35, 55, 0.95);
            }

            QComboBox::drop-down {
                border: none;
                width: 20px;
            }

            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #00c8ff;
                margin-right: 6px;
            }

            QComboBox QAbstractItemView {
                background: rgba(15, 25, 40, 0.95);
                border: 1px solid #00c8ff;
                selection-background-color: #0090c0;
            }

            QCheckBox {
                spacing: 6px;
                color: #c8dcff;
            }

            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 1px solid #306080;
                border-radius: 3px;
                background: rgba(15, 25, 40, 0.8);
            }

            QCheckBox::indicator:hover {
                border: 1px solid #00c8ff;
            }

            QCheckBox::indicator:checked {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #00a0c0,
                    stop: 1 #6060c0
                );
                border: 1px solid #00e0ff;
            }

            QProgressBar {
                border: 1px solid #306080;
                border-radius: 5px;
                background: rgba(15, 25, 40, 0.8);
                text-align: center;
                color: #c8dcff;
            }

            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #00a0c0,
                    stop: 0.3 #0080ff,
                    stop: 0.6 #6060ff,
                    stop: 1 #a040ff
                );
                border-radius: 4px;
            }

            QLabel {
                color: #c8dcff;
            }

            QSlider::groove:horizontal {
                border: 1px solid #306080;
                height: 6px;
                background: rgba(15, 25, 40, 0.8);
                border-radius: 3px;
            }

            QSlider::handle:horizontal {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #00c8ff,
                    stop: 1 #8080ff
                );
                border: 1px solid #00e0ff;
                width: 14px;
                margin: -4px 0;
                border-radius: 7px;
            }

            QSlider::handle:horizontal:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #40e0ff,
                    stop: 1 #a0a0ff
                );
            }

            QToolTip {
                background: rgba(20, 35, 55, 0.95);
                color: #c8dcff;
                border: 1px solid #00c8ff;
                border-radius: 4px;
                padding: 4px;
            }

            QScrollBar:vertical {
                border: none;
                background: rgba(15, 25, 40, 0.6);
                width: 10px;
                border-radius: 5px;
            }

            QScrollBar::handle:vertical {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #00a0c0,
                    stop: 1 #6060c0
                );
                border-radius: 5px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #00c0e0,
                    stop: 1 #8080e0
                );
            }

            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
        """

        cls._app.setStyleSheet(stylesheet)


def configure_dark_palette(app: QApplication) -> None:
    """
    Legacy function for backwards compatibility.

    Initializes the theme manager with the Dark theme.

    Args:
        app: The QApplication instance to configure.
    """
    ThemeManager.initialize(app, DEFAULT_THEME)
