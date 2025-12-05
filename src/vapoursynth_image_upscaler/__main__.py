"""
Main entry point for VapourSynth Image Upscaler.

This module handles the mode detection and dispatches to either:
- Worker mode (--worker or --alpha-worker): Heavy VapourSynth processing
- GUI mode (default): PySide6 graphical interface

Usage:
    # GUI mode (default)
    python -m vapoursynth_image_upscaler

    # Worker mode (spawned by GUI)
    python -m vapoursynth_image_upscaler --worker <input> <output_dir> <secondary_dir>
    python -m vapoursynth_image_upscaler --alpha-worker <input> <output_dir> <secondary_dir>
"""

from __future__ import annotations

import sys
import traceback


def main() -> None:
    """
    Main entry point.

    Detects the mode based on command-line arguments and dispatches accordingly.
    """
    # Check for worker mode flags
    is_worker = "--worker" in sys.argv[1:] or "--alpha-worker" in sys.argv[1:]

    if is_worker:
        # Worker mode: import and run worker
        # Heavy VapourSynth imports happen inside the worker module
        from .worker import worker_main
        worker_main()
    else:
        # GUI mode
        try:
            from .gui import main_gui
            main_gui()
        except Exception:
            traceback.print_exc()
            input("\nAn error occurred. Press Enter to exit...")


if __name__ == "__main__":
    main()
