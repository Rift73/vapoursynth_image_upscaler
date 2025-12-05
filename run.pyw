#!/usr/bin/env python
"""
Standalone launcher for VapourSynth Image Upscaler.

This script can be double-clicked on Windows to launch the application
without requiring installation. It adds the src directory to the path
and launches the main module.

Use .pyw extension to run without a console window on Windows.
"""

import sys
from pathlib import Path

# Add src directory to path for development use
src_dir = Path(__file__).parent / "src"
if src_dir.exists():
    sys.path.insert(0, str(src_dir))

# Import and run
from vapoursynth_image_upscaler.__main__ import main

if __name__ == "__main__":
    main()
