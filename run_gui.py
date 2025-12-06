#!/usr/bin/env python
"""
Entry point script for PyInstaller builds.

This script is used by PyInstaller to create a standalone executable.
It simply imports and runs the main entry point from the package.
"""

import sys
from pathlib import Path

# Add src to path for development runs
src_path = Path(__file__).parent / "src"
if src_path.exists():
    sys.path.insert(0, str(src_path))

from vapoursynth_image_upscaler.__main__ import main

if __name__ == "__main__":
    main()
