"""
Application-wide constants and path definitions.
"""

import tempfile
from pathlib import Path

# ============================================================================
# Paths
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent.parent
"""Root directory of the project (where the script/package is located)."""

TEMP_BASE = Path(tempfile.gettempdir())
"""System temporary directory."""

CONFIG_PATH = SCRIPT_DIR / "vs_upscale_gui_config.json"
"""Path to the configuration file."""

GUI_INPUT_TMP_ROOT = TEMP_BASE / "vs_upscale_gui_inputs"
"""Temporary folder for drag-and-drop and clipboard images."""

WORKER_TMP_ROOT = TEMP_BASE / "vs_upscale_tmp"
"""Temporary folder for worker outputs and timing files."""

# ============================================================================
# File Extensions
# ============================================================================

SUPPORTED_IMAGE_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"
})
"""Set of supported image file extensions (lowercase, with dot)."""

ALPHA_SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".png", ".gif", ".webp"
})
"""Set of file extensions that support alpha channel."""

# ============================================================================
# Model Configuration
# ============================================================================

VALID_MODEL_SCALES: tuple[int, ...] = (1, 2, 4, 8)
"""Valid super-resolution model scale factors."""

DEFAULT_MODEL_SCALE: int = 4
"""Default model scale if not specified."""

DEFAULT_TILE_WIDTH: int = 1088
"""Default tile width for vsmlrt inference."""

DEFAULT_TILE_HEIGHT: int = 1920
"""Default tile height for vsmlrt inference."""

PADDING_ALIGNMENT: int = 64
"""Input images are padded to multiples of this value."""

# ============================================================================
# Windows Process Creation Flags
# ============================================================================

CREATE_NO_WINDOW: int = 0x08000000
"""Windows creation flag to hide console windows for worker processes."""

# Combined flags for completely hidden subprocess
SUBPROCESS_FLAGS: int = (
    0x08000000 |  # CREATE_NO_WINDOW
    0x00000008    # DETACHED_PROCESS
)
"""Combined Windows flags for hidden subprocess execution."""

# ============================================================================
# Default Paths (can be overridden by user)
# ============================================================================

DEFAULT_ONNX_PATH: str = r"C:\Executables\models\HAT\HAT_L_28k_bf16.onnx"
"""Default path to the ONNX super-resolution model."""

DEFAULT_OUTPUT_PATH: str = r"C:\Pictures\Temp"
"""Default output path for clipboard/browser image inputs."""
