"""
Background worker thread for spawning upscale worker processes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import QThread, Signal

from ..core.constants import CREATE_NO_WINDOW, TEMP_BASE
from ..core.utils import cleanup_tmp_root, get_pythonw_executable

if TYPE_CHECKING:
    pass


class UpscaleWorkerThread(QThread):
    """
    QThread that manages spawning worker processes for each input file.

    Emits signals for progress updates, thumbnails, and completion.
    """

    # Signal: (current_index, total, avg_per_image, filename, filepath)
    progress_signal = Signal(int, int, float, str, str)

    # Signal: image path for thumbnail update
    thumbnail_signal = Signal(str)

    # Signal: final status text
    finished_signal = Signal(str)

    def __init__(
        self,
        files: list[Path],
        output_dir: Path,
        secondary_output_dir: Path,
        single_input_is_file: bool,
        custom_res_enabled: bool,
        custom_width: int,
        custom_height: int,
        secondary_enabled: bool,
        secondary_mode: str,
        secondary_width: int,
        secondary_height: int,
        same_dir_enabled: bool,
        same_dir_suffix: str,
        overwrite_enabled: bool,
        onnx_path: str,
        tile_w: str,
        tile_h: str,
        model_scale: str,
        use_fp16: bool,
        use_bf16: bool,
        use_tf32: bool,
        use_alpha: bool,
        append_model_suffix_enabled: bool = False,
        prescale_enabled: bool = False,
        prescale_mode: str = "width",
        prescale_width: int = 1920,
        prescale_height: int = 1080,
        kernel: str = "lanczos",
        sharpen_enabled: bool = False,
        sharpen_value: float = 0.5,
        parent=None,
    ):
        super().__init__(parent)
        self.files = files
        self.output_dir = output_dir
        self.secondary_output_dir = secondary_output_dir
        self.single_input_is_file = single_input_is_file
        self.custom_res_enabled = custom_res_enabled
        self.custom_width = custom_width
        self.custom_height = custom_height
        self.secondary_enabled = secondary_enabled
        self.secondary_mode = secondary_mode
        self.secondary_width = secondary_width
        self.secondary_height = secondary_height
        self.same_dir_enabled = same_dir_enabled
        self.same_dir_suffix = same_dir_suffix
        self.overwrite_enabled = overwrite_enabled
        self.onnx_path = onnx_path
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.model_scale = model_scale
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.use_tf32 = use_tf32
        self.use_alpha = use_alpha
        self.append_model_suffix_enabled = append_model_suffix_enabled
        self.prescale_enabled = prescale_enabled
        self.prescale_mode = prescale_mode
        self.prescale_width = prescale_width
        self.prescale_height = prescale_height
        self.kernel = kernel
        self.sharpen_enabled = sharpen_enabled
        self.sharpen_value = sharpen_value
        self._cancel_flag = False

    def cancel(self) -> None:
        """Request cancellation after the current file completes."""
        self._cancel_flag = True

    def run(self) -> None:
        """Main thread execution."""
        # Get the entry point script path
        script_path = self._get_script_path()

        total_files = len(self.files)
        total_processing_time = 0.0
        num_timed = 0
        current_avg = 0.0

        for idx, f in enumerate(self.files, start=1):
            if self._cancel_flag:
                break

            # Emit thumbnail and initial progress (preserve current avg for ETA)
            self.thumbnail_signal.emit(str(f))
            self.progress_signal.emit(idx - 1, total_files, current_avg, f.name, str(f))

            # Determine per-file output directories
            per_output_dir, per_secondary_dir = self._compute_output_dirs(f)

            # Ensure directories exist
            self._ensure_dirs(per_output_dir, per_secondary_dir)

            # Build environment for worker
            env = self._build_worker_env()

            file_start = time.perf_counter()

            # Run main worker
            self._run_worker(script_path, "--worker", f, per_output_dir, per_secondary_dir, env)

            # Run alpha worker if enabled
            if self.use_alpha and not self._cancel_flag:
                self._run_worker(script_path, "--alpha-worker", f, per_output_dir, per_secondary_dir, env)

            # Use wall-clock time (includes subprocess overhead) for accurate ETA
            file_elapsed = time.perf_counter() - file_start
            total_processing_time += file_elapsed
            num_timed += 1

            current_avg = total_processing_time / num_timed if num_timed > 0 else 0.0
            self.progress_signal.emit(idx, total_files, current_avg, f.name, str(f))

        # Build summary
        if num_timed > 0:
            final_avg = total_processing_time / num_timed
            summary = f"Average processing time per image: {final_avg:.2f} s over {num_timed} image(s)."
        else:
            summary = "No timing information recorded."

        if self._cancel_flag:
            final_text = "Cancelled. " + summary
        else:
            final_text = "Done. " + summary

        cleanup_tmp_root()
        self.finished_signal.emit(final_text)

    def _get_script_path(self) -> Path:
        """Get the path to the main entry point script."""
        # When running as a package, we need to find the entry point
        # First try the installed entry point, then fall back to __main__.py
        main_module = sys.modules.get("__main__")
        if main_module and hasattr(main_module, "__file__") and main_module.__file__:
            return Path(main_module.__file__).resolve()
        # Fallback to package's __main__.py
        return Path(__file__).parent.parent / "__main__.py"

    def _compute_output_dirs(self, f: Path) -> tuple[Path, Path]:
        """Compute output directories for a specific file."""
        if self.same_dir_enabled:
            per_output_dir = self.output_dir  # Placeholder; worker uses input.parent
            base_dir = f.parent if f.is_file() else f
            per_secondary_dir = base_dir / "secondary-resized"
        else:
            if self.single_input_is_file:
                per_output_dir = self.output_dir
                per_secondary_dir = self.secondary_output_dir
            else:
                base_dir = f.parent if f.is_file() else f
                per_output_dir = base_dir / "Upscaled"
                per_secondary_dir = base_dir / "secondary-resized"

        return per_output_dir, per_secondary_dir

    def _ensure_dirs(self, output_dir: Path, secondary_dir: Path) -> None:
        """Ensure output directories exist."""
        try:
            if not self.same_dir_enabled:
                output_dir.mkdir(parents=True, exist_ok=True)
            if self.secondary_enabled:
                secondary_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create per-file dirs: {e}")

    def _build_worker_env(self) -> dict[str, str]:
        """Build environment variables for worker process."""
        env = os.environ.copy()
        env["ONNX_PATH"] = self.onnx_path
        env["TILE_W_LIMIT"] = self.tile_w
        env["TILE_H_LIMIT"] = self.tile_h
        env["MODEL_SCALE"] = self.model_scale
        env["USE_FP16"] = "1" if self.use_fp16 else "0"
        env["USE_BF16"] = "1" if self.use_bf16 else "0"
        env["USE_TF32"] = "1" if self.use_tf32 else "0"
        env["USE_SAME_DIR_OUTPUT"] = "1" if self.same_dir_enabled else "0"
        env["SAME_DIR_SUFFIX"] = self.same_dir_suffix
        env["OVERWRITE_OUTPUT"] = "1" if self.overwrite_enabled else "0"
        env["CUSTOM_RES_ENABLED"] = "1" if self.custom_res_enabled else "0"
        env["CUSTOM_WIDTH"] = str(self.custom_width)
        env["CUSTOM_HEIGHT"] = str(self.custom_height)
        env["USE_SECONDARY_OUTPUT"] = "1" if self.secondary_enabled else "0"
        env["SECONDARY_MODE"] = self.secondary_mode
        env["SECONDARY_WIDTH"] = str(self.secondary_width)
        env["SECONDARY_HEIGHT"] = str(self.secondary_height)
        env["USE_ALPHA"] = "1" if self.use_alpha else "0"
        env["APPEND_MODEL_SUFFIX"] = "1" if self.append_model_suffix_enabled else "0"
        env["PRESCALE_ENABLED"] = "1" if self.prescale_enabled else "0"
        env["PRESCALE_MODE"] = self.prescale_mode
        env["PRESCALE_WIDTH"] = str(self.prescale_width)
        env["PRESCALE_HEIGHT"] = str(self.prescale_height)
        env["KERNEL"] = self.kernel
        env["SHARPEN_ENABLED"] = "1" if self.sharpen_enabled else "0"
        env["SHARPEN_VALUE"] = str(self.sharpen_value)
        return env

    def _run_worker(
        self,
        script_path: Path,
        mode: str,
        input_file: Path,
        output_dir: Path,
        secondary_dir: Path,
        env: dict[str, str],
    ) -> None:
        """Run a worker subprocess without creating a console window."""
        # Use pythonw.exe on Windows to avoid console windows
        python_exe = get_pythonw_executable()

        cmd = [
            python_exe,
            str(script_path),
            mode,
            str(input_file),
            str(output_dir),
            str(secondary_dir),
        ]

        # On Windows, use CREATE_NO_WINDOW flag as additional safeguard
        # Also redirect stdin/stdout/stderr to DEVNULL to prevent any console attachment
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            subprocess.run(
                cmd,
                check=False,
                env=env,
                creationflags=CREATE_NO_WINDOW,
                startupinfo=startupinfo,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(cmd, check=False, env=env)


class ClipboardWorkerThread(QThread):
    """
    QThread that upscales a single image and outputs the path for clipboard copy.

    Outputs to a temporary file which the GUI then copies to clipboard.
    """

    # Signal: path to the upscaled image (or empty string on failure)
    result_signal = Signal(str)

    # Signal: status message
    status_signal = Signal(str)

    def __init__(
        self,
        input_file: Path,
        onnx_path: str,
        tile_w: str,
        tile_h: str,
        model_scale: str,
        use_fp16: bool,
        use_bf16: bool,
        use_tf32: bool,
        use_alpha: bool,
        custom_res_enabled: bool = False,
        custom_width: int = 0,
        custom_height: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self.input_file = input_file
        self.onnx_path = onnx_path
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.model_scale = model_scale
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.use_tf32 = use_tf32
        self.use_alpha = use_alpha
        self.custom_res_enabled = custom_res_enabled
        self.custom_width = custom_width
        self.custom_height = custom_height

    def run(self) -> None:
        """Run the upscale worker and emit the result path."""
        script_path = self._get_script_path()

        # Create a temporary output directory for clipboard result
        clipboard_tmp_dir = TEMP_BASE / "vs_upscale_clipboard"
        clipboard_tmp_dir.mkdir(parents=True, exist_ok=True)

        # Output file path (keep original extension)
        output_file = clipboard_tmp_dir / f"clipboard_result{self.input_file.suffix}"

        # Remove old result if exists
        if output_file.exists():
            try:
                output_file.unlink()
            except Exception:
                pass

        self.status_signal.emit("Upscaling image...")

        env = self._build_worker_env()

        # Run main worker
        self._run_worker(script_path, "--worker", output_file.parent, env)

        # Run alpha worker if enabled
        if self.use_alpha:
            self._run_worker(script_path, "--alpha-worker", output_file.parent, env)

        # Check for output file
        # The worker names output based on input stem, so look for it
        expected_output = clipboard_tmp_dir / f"{self.input_file.stem}.png"
        if not expected_output.exists():
            # Try original extension
            expected_output = clipboard_tmp_dir / f"{self.input_file.stem}{self.input_file.suffix}"

        if expected_output.exists():
            self.status_signal.emit("Image copied to clipboard!")
            self.result_signal.emit(str(expected_output))
        else:
            self.status_signal.emit("Failed to upscale image.")
            self.result_signal.emit("")

    def _get_script_path(self) -> Path:
        """Get the path to the main entry point script."""
        main_module = sys.modules.get("__main__")
        if main_module and hasattr(main_module, "__file__") and main_module.__file__:
            return Path(main_module.__file__).resolve()
        return Path(__file__).parent.parent / "__main__.py"

    def _build_worker_env(self) -> dict[str, str]:
        """Build environment variables for worker process."""
        env = os.environ.copy()
        env["ONNX_PATH"] = self.onnx_path
        env["TILE_W_LIMIT"] = self.tile_w
        env["TILE_H_LIMIT"] = self.tile_h
        env["MODEL_SCALE"] = self.model_scale
        env["USE_FP16"] = "1" if self.use_fp16 else "0"
        env["USE_BF16"] = "1" if self.use_bf16 else "0"
        env["USE_TF32"] = "1" if self.use_tf32 else "0"
        env["USE_SAME_DIR_OUTPUT"] = "0"
        env["SAME_DIR_SUFFIX"] = ""
        env["OVERWRITE_OUTPUT"] = "1"  # Always overwrite for clipboard
        env["CUSTOM_RES_ENABLED"] = "1" if self.custom_res_enabled else "0"
        env["CUSTOM_WIDTH"] = str(self.custom_width)
        env["CUSTOM_HEIGHT"] = str(self.custom_height)
        env["USE_SECONDARY_OUTPUT"] = "0"  # No secondary for clipboard
        env["SECONDARY_MODE"] = "width"
        env["SECONDARY_WIDTH"] = "0"
        env["SECONDARY_HEIGHT"] = "0"
        env["USE_ALPHA"] = "1" if self.use_alpha else "0"
        env["APPEND_MODEL_SUFFIX"] = "0"  # No suffix for clipboard
        return env

    def _run_worker(
        self,
        script_path: Path,
        mode: str,
        output_dir: Path,
        env: dict[str, str],
    ) -> None:
        """Run a worker subprocess without creating a console window."""
        python_exe = get_pythonw_executable()

        cmd = [
            python_exe,
            str(script_path),
            mode,
            str(self.input_file),
            str(output_dir),
            str(output_dir),  # secondary_dir (unused)
        ]

        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            subprocess.run(
                cmd,
                check=False,
                env=env,
                creationflags=CREATE_NO_WINDOW,
                startupinfo=startupinfo,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(cmd, check=False, env=env)
