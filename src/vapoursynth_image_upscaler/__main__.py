"""
Main entry point for VapourSynth Image Upscaler.

This module handles the mode detection and dispatches to either:
- Worker mode (--worker, --alpha-worker, --batch-worker): Heavy VapourSynth processing
- GUI mode (default): PySide6 graphical interface

Usage:
    # GUI mode (default)
    python -m vapoursynth_image_upscaler

    # Worker mode (spawned by GUI)
    python -m vapoursynth_image_upscaler --worker <input> <output_dir> <secondary_dir>
    python -m vapoursynth_image_upscaler --alpha-worker <input> <output_dir> <secondary_dir>
    python -m vapoursynth_image_upscaler --batch-worker <manifest_json>
"""

from __future__ import annotations

import sys


def main() -> None:
    """
    Main entry point.

    Detects the mode based on command-line arguments and dispatches accordingly.
    """
    # Check for worker mode flags
    worker_flags = ("--worker", "--alpha-worker", "--batch-worker")
    is_worker = any(flag in sys.argv[1:] for flag in worker_flags)

    if is_worker:
        # Worker mode: import and run worker
        # Heavy VapourSynth imports happen inside the worker module
        from .worker import worker_main
        worker_main()
    else:
        # GUI mode
        from .gui import main_gui
        main_gui()


if __name__ == "__main__":
    main()
