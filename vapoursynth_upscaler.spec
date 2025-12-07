# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for VapourSynth Image Upscaler.

Build commands:
    # Standard build (folder with exe + dependencies)
    pyinstaller vapoursynth_upscaler.spec

    # Or use one of these shortcuts:
    pyinstaller vapoursynth_upscaler.spec --clean

Notes:
    - VapourSynth must be installed separately (cannot be bundled)
    - ONNX models must be provided separately
    - External tools (gifski, ffmpeg, avifenc) must be in PATH
"""

import sys
from pathlib import Path

block_cipher = None

# Project root
PROJECT_ROOT = Path(SPECPATH)
SRC_PATH = PROJECT_ROOT / "src"

# Collect all package data
datas = [
    # Include the icon
    (str(PROJECT_ROOT / "icon.png"), "."),
    (str(PROJECT_ROOT / "icon.ico"), "."),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # PySide6 plugins
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    # PIL/Pillow
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFilter",
    # NumPy
    "numpy",
    # Package modules
    "vapoursynth_image_upscaler",
    "vapoursynth_image_upscaler.gui",
    "vapoursynth_image_upscaler.gui.main",
    "vapoursynth_image_upscaler.gui.main_window",
    "vapoursynth_image_upscaler.gui.dialogs",
    "vapoursynth_image_upscaler.gui.theme",
    "vapoursynth_image_upscaler.gui.worker_thread",
    "vapoursynth_image_upscaler.core",
    "vapoursynth_image_upscaler.core.config",
    "vapoursynth_image_upscaler.core.constants",
    "vapoursynth_image_upscaler.core.utils",
    "vapoursynth_image_upscaler.worker",
    "vapoursynth_image_upscaler.worker.processor",
    "vapoursynth_image_upscaler.worker.settings",
]

# Excludes - things we don't need
excludes = [
    "tkinter",
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "sphinx",
    # Exclude PyQt6 to avoid conflict with PySide6
    "PyQt6",
    "PyQt5",
    # Exclude yt-dlp and related packages (not needed)
    "yt_dlp",
    "yt_dlp_ejs",
    "websockets",
    "curl_cffi",
    "brotli",
    "mutagen",
    "secretstorage",
    "Cryptodome",
    # Exclude torch/torchvision (not needed, reduces size significantly)
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "timm",
    # Exclude other ML packages
    "onnxruntime",
    "cv2",
    "opencv-python",
]

a = Analysis(
    [str(PROJECT_ROOT / "run_gui.py")],
    pathex=[str(SRC_PATH)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Single executable (onefile) build
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="VapourSynth Upscaler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window for GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "icon.ico"),
)
