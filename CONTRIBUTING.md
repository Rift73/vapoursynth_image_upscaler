# Contributing to VapourSynth Image Upscaler

Thank you for your interest in contributing to VapourSynth Image Upscaler!

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/vapoursynth_image_upscaler.git
   cd vapoursynth_image_upscaler
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

4. Ensure VapourSynth and plugins are installed (see README.md)

## Code Style

This project uses:
- **ruff** for linting and formatting
- **mypy** for type checking

Before submitting a PR:

```bash
# Format code
ruff format src/

# Lint code
ruff check src/ --fix

# Type check
mypy src/
```

## Architecture Guidelines

### GUI Mode vs Worker Mode

The application separates GUI and worker modes to:
1. Keep GUI startup fast (no heavy VapourSynth imports)
2. Prevent VRAM accumulation (fresh process per image)
3. Avoid TensorRT engine conflicts

**Important**: Never import VapourSynth modules in GUI code.

### Module Organization

- `core/`: Shared utilities, constants, and configuration
- `gui/`: PySide6 GUI components (no VS imports)
- `worker/`: VapourSynth processing (heavy imports here)

### Adding Features

When adding new features:

1. **Settings**: Add to `core/config.py` for GUI-side persistence
2. **Worker Settings**: Add to `worker/settings.py` for environment-based passing
3. **Environment Variables**: Workers receive settings via environment variables
4. **Keep Scale Generic**: Use `MODEL_SCALE` instead of hardcoding "4x"

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run linting and type checks
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Reporting Issues

When reporting issues, please include:
- Python version
- VapourSynth version
- GPU model and driver version
- Steps to reproduce
- Error messages/tracebacks

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
