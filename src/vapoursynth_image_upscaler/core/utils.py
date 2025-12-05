"""
Utility functions used across the application.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from .constants import WORKER_TMP_ROOT, GUI_INPUT_TMP_ROOT

if TYPE_CHECKING:
    pass

# Check for Windows console input support
_HAS_MSVCRT = False
try:
    import msvcrt
    _HAS_MSVCRT = True
except ImportError:
    pass


def get_pythonw_executable() -> str:
    """
    Get the path to pythonw.exe (windowless Python) on Windows.

    On Windows, this returns pythonw.exe to avoid console windows.
    On other platforms, returns the current Python executable.

    Returns:
        Path to the Python executable to use for GUI subprocesses.
    """
    if sys.platform != "win32":
        return sys.executable

    executable = Path(sys.executable)

    # If already using pythonw, return as-is
    if executable.name.lower() == "pythonw.exe":
        return sys.executable

    # Try to find pythonw.exe in the same directory
    pythonw = executable.parent / "pythonw.exe"
    if pythonw.exists():
        return str(pythonw)

    # Fallback to regular python (will need CREATE_NO_WINDOW flag)
    return sys.executable


def user_requested_quit() -> bool:
    """
    Check if the user pressed 'q' or 'Q' in the Windows console.

    Non-blocking check, primarily kept for possible CLI use/debug.

    Returns:
        True if 'q' or 'Q' was pressed, False otherwise.
    """
    if not _HAS_MSVCRT:
        return False

    if not msvcrt.kbhit():
        return False

    ch = msvcrt.getch()
    try:
        s = ch.decode("utf-8", errors="ignore")
    except Exception:
        s = ""
    return s.lower() == "q"


def read_time_file(base_name: str) -> float | None:
    """
    Read the per-image processing time written by worker processes.

    The time file is created by workers in WORKER_TMP_ROOT with the format
    `{base_name}.time` containing a float representing seconds.

    Args:
        base_name: The stem of the input file (without extension).

    Returns:
        Processing time in seconds, or None if the file doesn't exist
        or couldn't be parsed.
    """
    try:
        time_file = WORKER_TMP_ROOT / f"{base_name}.time"
        if not time_file.exists():
            return None

        with open(time_file, "r", encoding="utf-8") as f:
            txt = f.read().strip()

        try:
            value = float(txt)
        except ValueError:
            value = None

        # Clean up the time file after reading
        try:
            time_file.unlink()
        except OSError:
            pass

        return value
    except Exception:
        return None


def write_time_file(base_name: str, processing_time: float) -> None:
    """
    Write processing time to a time file for the given input.

    Args:
        base_name: The stem of the input file (without extension).
        processing_time: Processing time in seconds.
    """
    try:
        WORKER_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        time_file = WORKER_TMP_ROOT / f"{base_name}.time"
        with open(time_file, "w", encoding="utf-8") as f:
            f.write(f"{processing_time:.6f}")
    except Exception as e:
        print(f"Warning: Could not write time file: {e}")


def cleanup_tmp_root() -> None:
    """
    Remove the worker temporary folder used for outputs and timing files.

    Should be called at the end of a batch to clean up.
    """
    try:
        if WORKER_TMP_ROOT.exists():
            shutil.rmtree(WORKER_TMP_ROOT, ignore_errors=True)
    except Exception:
        pass


def cleanup_gui_input_tmp() -> None:
    """
    Remove the GUI input temporary folder used for drag-and-drop/clipboard images.

    Should be called at the end of a batch and on GUI close.
    """
    try:
        if GUI_INPUT_TMP_ROOT.exists():
            shutil.rmtree(GUI_INPUT_TMP_ROOT, ignore_errors=True)
    except Exception:
        pass


def format_time_hms(seconds: float) -> str:
    """
    Format a duration in seconds as HH:MM:SS.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string in HH:MM:SS format.
    """
    seconds = max(0, int(seconds + 0.5))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def get_env_int(name: str, default: int) -> int:
    """
    Get an integer value from an environment variable.

    Args:
        name: Name of the environment variable.
        default: Default value if not set or invalid.

    Returns:
        The integer value or the default.
    """
    import os

    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_bool(name: str, default: bool = False) -> bool:
    """
    Get a boolean value from an environment variable.

    The value is considered True if the env var is set to "1".

    Args:
        name: Name of the environment variable.
        default: Default value if not set.

    Returns:
        The boolean value.
    """
    import os

    value = os.environ.get(name)
    if value is None:
        return default
    return value == "1"


def get_env_str(name: str, default: str) -> str:
    """
    Get a string value from an environment variable.

    Args:
        name: Name of the environment variable.
        default: Default value if not set.

    Returns:
        The string value or the default.
    """
    import os
    return os.environ.get(name, default)


def compute_padding(width: int, height: int, alignment: int = 64) -> tuple[int, int, int, int]:
    """
    Compute symmetric padding to align dimensions to a multiple of alignment.

    Args:
        width: Original width.
        height: Original height.
        alignment: Target alignment (default 64).

    Returns:
        Tuple of (pad_left, pad_right, pad_top, pad_bottom).
    """
    total_pad_w = (((width + alignment - 1) // alignment) * alignment) - width
    total_pad_h = (((height + alignment - 1) // alignment) * alignment) - height

    pad_left = total_pad_w // 2
    pad_right = total_pad_w - pad_left
    pad_top = total_pad_h // 2
    pad_bottom = total_pad_h - pad_top

    return pad_left, pad_right, pad_top, pad_bottom
