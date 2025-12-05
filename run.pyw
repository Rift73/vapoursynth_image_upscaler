#!/usr/bin/env pythonw
"""
Standalone launcher for VapourSynth Image Upscaler.

This script can be double-clicked on Windows to launch the application
without requiring installation. It adds the src directory to the path
and launches the main module.

Use .pyw extension to run without a console window on Windows.
"""

import os
import subprocess
import sys
from pathlib import Path


def _relaunch_with_pythonw() -> bool:
    """
    If running with python.exe on Windows, relaunch with pythonw.exe.
    Returns True if relaunching (caller should exit), False otherwise.
    """
    if sys.platform != "win32":
        return False

    executable = Path(sys.executable)

    # Already using pythonw.exe, no need to relaunch
    if executable.name.lower() == "pythonw.exe":
        return False

    # Find pythonw.exe
    pythonw = executable.parent / "pythonw.exe"
    if not pythonw.exists():
        return False

    # Relaunch with pythonw.exe
    script = Path(__file__).resolve()
    subprocess.Popen(
        [str(pythonw), str(script)] + sys.argv[1:],
        creationflags=0x08000000,  # CREATE_NO_WINDOW
    )
    return True


if __name__ == "__main__":
    # Relaunch with pythonw if we're running with python.exe
    if _relaunch_with_pythonw():
        sys.exit(0)

    # Add src directory to path for development use
    src_dir = Path(__file__).parent / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))

    # Import and run
    from vapoursynth_image_upscaler.__main__ import main
    main()
