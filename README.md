# VapourSynth Image Upscaler

A Windows desktop tool for batch image super-resolution using VapourSynth and TensorRT.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Platform: Windows](https://img.shields.io/badge/platform-Windows-lightgrey.svg)

## Features

- Drag-and-drop GUI for batch image upscaling
- Supports PNG, JPEG, BMP, TIFF, WebP, GIF (including animated GIFs)
- Animated output formats: GIF, WebP, AVIF, APNG
- TensorRT-accelerated inference via vsmlrt
- Flexible model scales: 1x, 2x, 4x, 8x
- Alpha channel processing for transparent images (including palette-based PNGs)
- Custom resolution and secondary output options
- Pre-scale downscaling before SR processing
- Post-upscale sharpening with CAS
- PNG optimization: quantization (pngquant) and lossless compression (pingo)
- Manga folder mode for preserving directory structure
- Real-time progress with ETA tracking
- Batch processing mode for faster inference on multiple images
- Built-in dependencies installer for VapourSynth plugins and external tools

## Requirements

- Windows 10/11 64-bit
- NVIDIA GPU with TensorRT support
- Python 3.10+
- 7-Zip (for extracting .7z plugin archives)

## Installation

```bash
git clone https://github.com/yourusername/vapoursynth_image_upscaler.git
cd vapoursynth_image_upscaler
pip install -e .
```

### Automatic Dependency Installation

Use the **Dependencies** button in the GUI to automatically install:

**Python packages:**
- PySide6, vsjetpack, Pillow, numpy

**VapourSynth plugins** (installed to `%APPDATA%/Vapoursynth/plugins64`):
- akarin, fmtconv, resize2, vapoursynth-zip, zsmooth
- bestsource, imwri, deblock, dctfilter, vsmlrt

**External tools** (installed to `%APPDATA%/vapoursynth-image-upscaler-GUI`):
- ffmpeg, gifski, avifenc (animated output)
- pngquant, pingo (PNG optimization)

The installer automatically adds tool directories to your user PATH.

**Note:** VapourSynth itself must be installed manually. Download from [VapourSynth Releases](https://github.com/vapoursynth/vapoursynth/releases).

### Manual Installation

Alternatively, install manually:

```bash
pip install -r requirements.txt
```

And manually install:
- VapourSynth R62+
- vsmlrt (TRT) with TensorRT backend
- vsjetpack plugins

Find upscaling models on [OpenModelDB](https://openmodeldb.info/)

## Usage

Launch the GUI:

```bash
vs-upscale-gui
# or
python -m vapoursynth_image_upscaler
```

Use `run.bat` to launch without a console window.

### Output Modes

**Standard**: Outputs to a specified folder or creates an "Upscaled" subfolder next to input.

**Save next to input**: Saves output files alongside originals with a customizable suffix.

**Manga folder**: Preserves folder hierarchy by applying the suffix to the root folder name. Useful for batch processing manga or comic archives where you want to maintain the original structure.

### Resolution Options

Access via the **Resolution** button:

- **Custom resolution**: Downscale SR output to a target width or height
- **Secondary output**: Generate an additional resized version
- **Pre-scale**: Downscale input before SR processing

Each supports width-based, height-based, or 2x scaling with Lanczos or Hermite kernels.

### Processing Settings

- **Tile size**: Configure width and height for vsmlrt inference
- **Model scale**: Match your ONNX model's upscaling factor
- **Precision**: Toggle fp16, bf16, tf32 for vsmlrt Backend.TRT
- **Transparency**: Enable alpha channel processing (supports palette-based PNGs with indexed transparency)
- **Sharpen**: Apply contrast adaptive sharpening to output
- **Overwrite**: Replace existing files or auto-increment names
- **Append model suffix**: Add model name to output filenames
- **Batch mode**: Process multiple same-resolution images together for faster inference

### Animated Output Options

Access via the **Animated Output** button to configure output format for animated content:

- **GIF**: Classic animated GIF using gifski for high-quality dithering
- **WebP**: Animated WebP with configurable quality and compression
- **AVIF**: Modern AV1-based format with excellent compression (requires avifenc)
- **APNG**: Animated PNG with lossless quality

Each format has specific quality settings:
- GIF: Quality (1-100), motion quality, lossy mode
- WebP: Quality (0-100), compression level (0-6), lossless mode
- AVIF: Quality (0-63), speed (0-10), separate alpha quality, lossless mode
- APNG: Predictor selection for optimization

### Batch Processing Mode

When enabled, batch mode groups images by resolution and format, then processes them as a continuous sequence through vsmlrt. This provides significantly faster inference (up to 2x) compared to processing images individually, as it reduces GPU initialization overhead.

Batch mode is automatically applied to static images (PNG, JPG, BMP, TIFF, WebP) that don't require alpha processing. Images with different resolutions or formats are grouped separately and processed in batches of up to 100 files.

## Architecture

The GUI spawns separate worker processes for each image to prevent VRAM accumulation. This keeps the interface responsive and avoids TensorRT memory leaks.

## Configuration

Settings persist to Windows Registry under `HKEY_CURRENT_USER\Software\VapourSynthImageUpscaler`.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [VapourSynth](https://github.com/vapoursynth/vapoursynth)
- [vsmlrt](https://github.com/AmusementClub/vs-mlrt)
- [PySide6](https://doc.qt.io/qtforpython/)
