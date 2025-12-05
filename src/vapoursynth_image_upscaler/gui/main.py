"""
GUI entry point for VapourSynth Image Upscaler.
"""

from __future__ import annotations

import sys
import traceback


def install_excepthook() -> None:
    """
    Install a custom exception hook to print uncaught exceptions from Qt/threads.
    """

    def handle_exception(exc_type, exc_value, exc_traceback):
        # Let Ctrl+C still behave normally
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        print("\n=== Uncaught exception ===", file=sys.stderr)
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print("=== End of traceback ===\n", file=sys.stderr)

    sys.excepthook = handle_exception


def main_gui() -> None:
    """
    Main entry point for the GUI application.

    Launches the PySide6 application with the main window.
    """
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print("PySide6 is not installed. Please install it with:")
        print("    pip install PySide6")
        sys.exit(1)

    install_excepthook()

    from .theme import configure_dark_palette
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    configure_dark_palette(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
