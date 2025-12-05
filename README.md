# VapourSynth Image Upscaler

A **Windows desktop upscaling tool** using **VapourSynth + vsmlrt (TensorRT backend)** with a **PySide6 GUI** for batch super-resolution of images.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

## Features

- **Drag-and-drop friendly GUI** for batch super-resolution of images
- **Multiple input formats**: PNG, JPEG, BMP, TIFF, WebP, GIF
- **vsmlrt + TensorRT backend** for fast GPU-accelerated inference
- **Flexible model scales**: 1x, 2x, 4x, 8x (not hardcoded to any specific scale)
- **Alpha channel support**: High-quality SR upscaling for transparent images
- **Custom resolution output**: Downscale SR results to specific dimensions
- **Secondary output**: Generate additional resized versions (width-based, height-based, or 2x)
- **VRAM-friendly**: One worker process per image to prevent memory leaks
- **Progress tracking**: Real-time ETA, elapsed time, and average time per image

## Architecture

The application runs in two modes:

1. **GUI Mode** (default): PySide6 interface for user interaction
2. **Worker Mode** (CLI): Spawned by GUI for VapourSynth/TensorRT processing

This separation keeps the GUI responsive and prevents VapourSynth's heavy dependencies from affecting startup time.

## Requirements

### System Requirements

- **Windows 10/11** (64-bit)
- **NVIDIA GPU** with TensorRT support
- **Python 3.10+**

### Dependencies

**GUI (pip-installable):**
- PySide6 >= 6.5.0
- rich (optional, for fancy progress bars)

**VapourSynth ecosystem (install via vsrepo or manually):**
- VapourSynth R62+
- vstools
- vskernels
- vsmlrt (with TensorRT backend)
- vssource (BestSource)
- fpng (for fast PNG output)

## Installation

### Option 1: Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/vapoursynth_image_upscaler.git
cd vapoursynth_image_upscaler

# Install the package
pip install -e .

# Or with optional dependencies
pip install -e ".[rich]"
```

### Option 2: Run directly

```bash
# Install GUI dependencies
pip install PySide6

# Run the application
python -m vapoursynth_image_upscaler
```

### VapourSynth Setup

Ensure VapourSynth and required plugins are installed:

```bash
# Using vsrepo (recommended)
vsrepo install vstools vskernels vsmlrt vssource fpng

# Configure TensorRT backend for vsmlrt
# See vsmlrt documentation for TensorRT setup
```

## Usage

### GUI Mode

Launch the application:

```bash
# If installed
vs-upscale-gui

# Or run directly
python -m vapoursynth_image_upscaler
```

#### Main Window Features

1. **Input Selection**
   - Browse for files or folders
   - Drag & drop images, folders, or URLs
   - Paste images from clipboard (Ctrl+V)

2. **Output Options**
   - Set custom output folder
   - Save next to input with suffix
   - Overwrite existing files or auto-increment

3. **Processing Settings**
   - Select ONNX super-resolution model
   - Configure tile sizes (must be multiples of 64)
   - Choose model scale (1x, 2x, 4x, 8x)
   - Toggle precision modes (fp16, bf16, tf32)

4. **Advanced Options**
   - Enable alpha channel processing for transparent images
   - Custom main resolution (downscale from SR output)
   - Secondary resized output (width-based, height-based, or 2x)

### Worker Mode (CLI)

Workers are spawned automatically by the GUI. For debugging:

```bash
# Main color SR worker
python -m vapoursynth_image_upscaler --worker <input> <output_dir> <secondary_dir>

# Alpha-only worker
python -m vapoursynth_image_upscaler --alpha-worker <input> <output_dir> <secondary_dir>
```

Environment variables control worker behavior:
- `ONNX_PATH`: Path to ONNX model
- `MODEL_SCALE`: SR scale factor (1, 2, 4, 8)
- `TILE_W_LIMIT`, `TILE_H_LIMIT`: Tile dimensions
- `USE_FP16`, `USE_BF16`, `USE_TF32`: Precision flags
- See `src/vapoursynth_image_upscaler/worker/settings.py` for full list

## Project Structure

```
vapoursynth_image_upscaler/
├── src/
│   └── vapoursynth_image_upscaler/
│       ├── __init__.py          # Package metadata
│       ├── __main__.py          # Entry point with mode detection
│       ├── core/                # Shared utilities and configuration
│       │   ├── constants.py     # Application constants
│       │   ├── config.py        # Configuration management
│       │   └── utils.py         # Utility functions
│       ├── gui/                 # PySide6 GUI
│       │   ├── main.py          # GUI entry point
│       │   ├── main_window.py   # Main application window
│       │   ├── dialogs.py       # Dialog windows
│       │   ├── worker_thread.py # Background processing thread
│       │   └── theme.py         # Dark theme configuration
│       └── worker/              # VapourSynth worker mode
│           ├── main.py          # Worker entry point
│           ├── settings.py      # Environment-based settings
│           ├── pipeline.py      # SR processing pipeline
│           ├── processor.py     # File processing logic
│           └── progress.py      # Progress bar utilities
├── pyproject.toml               # Project configuration
├── requirements.txt             # Runtime dependencies
├── requirements-dev.txt         # Development dependencies
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Configuration

Settings are persisted to `vs_upscale_gui_config.json` in the project directory:

```json
{
  "onnx_path": "C:\\path\\to\\model.onnx",
  "tile_w_limit": 1088,
  "tile_h_limit": 1920,
  "model_scale": 4,
  "use_bf16": true,
  "overwrite": true,
  "use_alpha": false,
  ...
}
```

## Development

### Setup

```bash
# Clone and install dev dependencies
git clone https://github.com/yourusername/vapoursynth_image_upscaler.git
cd vapoursynth_image_upscaler
pip install -e ".[dev]"
```

### Code Quality

```bash
# Lint with ruff
ruff check src/

# Format with ruff
ruff format src/

# Type check with mypy
mypy src/
```

### Running Tests

```bash
pytest tests/
```

## Troubleshooting

### TensorRT Engine Build Failures

- Ensure tile sizes are multiples of 64
- Check GPU memory availability
- Verify TensorRT installation

### VRAM Issues

The application spawns separate processes per image to prevent VRAM accumulation. If you still experience issues:
- Reduce tile sizes
- Process fewer images at once
- Close other GPU-intensive applications

### Alpha Channel Not Working

- Ensure input format supports alpha (PNG, GIF, WebP)
- Check "Use alpha" is enabled
- Alpha processing runs as a separate pass after color SR

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [VapourSynth](https://github.com/vapoursynth/vapoursynth) - Video processing framework
- [vsmlrt](https://github.com/AmusementClub/vs-mlrt) - ML runtime for VapourSynth
- [PySide6](https://doc.qt.io/qtforpython/) - Qt bindings for Python
