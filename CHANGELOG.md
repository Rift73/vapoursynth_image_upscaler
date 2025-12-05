# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-12-05

### Added

- Initial release of VapourSynth Image Upscaler
- PySide6 GUI with dark theme
- Drag-and-drop support for images, folders, and URLs
- Clipboard paste support (Ctrl+V)
- Batch processing of multiple images
- Support for PNG, JPEG, BMP, TIFF, WebP, and GIF formats
- vsmlrt TensorRT backend integration
- Configurable model scales (1x, 2x, 4x, 8x)
- Alpha channel super-resolution for transparent images
- Custom main resolution output (downscale from SR)
- Secondary resized output (width-based, height-based, or 2x)
- Real-time progress tracking with ETA
- Average processing time per image display
- Settings persistence across sessions
- VRAM-friendly worker process isolation
- Configurable tile sizes and precision modes (fp16, bf16, tf32)
- Overwrite or auto-increment filename handling
- Save next to input with optional suffix

### Technical

- Modular architecture separating GUI and worker modes
- Type hints throughout the codebase
- Comprehensive docstrings
- pyproject.toml-based packaging
- ruff and mypy configuration for code quality

[1.0.0]: https://github.com/yourusername/vapoursynth_image_upscaler/releases/tag/v1.0.0
