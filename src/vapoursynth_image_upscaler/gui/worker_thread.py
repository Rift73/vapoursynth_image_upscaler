"""
Background worker thread for spawning upscale worker processes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from ..core.constants import CREATE_NO_WINDOW, TEMP_BASE, SUPPORTED_VIDEO_EXTENSIONS, WORKER_TMP_ROOT, MAX_BATCH_SIZE
from ..core.utils import cleanup_tmp_root, get_pythonw_executable, get_video_duration, get_video_fps

# Extensions eligible for batch processing (static images only)
BATCH_ELIGIBLE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


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
        input_roots: list[Path],
        custom_res_enabled: bool,
        custom_res_mode: str,
        custom_width: int,
        custom_height: int,
        custom_res_kernel: str,
        secondary_enabled: bool,
        secondary_mode: str,
        secondary_width: int,
        secondary_height: int,
        secondary_kernel: str,
        same_dir_enabled: bool,
        same_dir_suffix: str,
        manga_folder_enabled: bool,
        overwrite_enabled: bool,
        onnx_path: str,
        tile_w: str,
        tile_h: str,
        model_scale: str,
        use_fp16: bool,
        use_bf16: bool,
        use_tf32: bool,
        num_streams: int = 1,
        use_alpha: bool = False,
        append_model_suffix_enabled: bool = False,
        prescale_enabled: bool = False,
        prescale_mode: str = "width",
        prescale_width: int = 1920,
        prescale_height: int = 1080,
        prescale_kernel: str = "lanczos",
        sharpen_enabled: bool = False,
        sharpen_value: float = 0.5,
        use_batch_mode: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.files = files
        self.use_batch_mode = use_batch_mode
        self.output_dir = output_dir
        self.secondary_output_dir = secondary_output_dir
        self.single_input_is_file = single_input_is_file
        self.input_roots = input_roots
        self.custom_res_enabled = custom_res_enabled
        self.custom_res_mode = custom_res_mode
        self.custom_width = custom_width
        self.custom_height = custom_height
        self.custom_res_kernel = custom_res_kernel
        self.secondary_enabled = secondary_enabled
        self.secondary_mode = secondary_mode
        self.secondary_width = secondary_width
        self.secondary_height = secondary_height
        self.secondary_kernel = secondary_kernel
        self.same_dir_enabled = same_dir_enabled
        self.same_dir_suffix = same_dir_suffix
        self.manga_folder_enabled = manga_folder_enabled
        self.overwrite_enabled = overwrite_enabled
        self.onnx_path = onnx_path
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.model_scale = model_scale
        self.use_fp16 = use_fp16
        self.use_bf16 = use_bf16
        self.use_tf32 = use_tf32
        self.num_streams = num_streams
        self.use_alpha = use_alpha
        self.append_model_suffix_enabled = append_model_suffix_enabled
        self.prescale_enabled = prescale_enabled
        self.prescale_mode = prescale_mode
        self.prescale_width = prescale_width
        self.prescale_height = prescale_height
        self.prescale_kernel = prescale_kernel
        self.sharpen_enabled = sharpen_enabled
        self.sharpen_value = sharpen_value
        self._cancel_flag = False

    def cancel(self) -> None:
        """Request cancellation after the current file completes."""
        self._cancel_flag = True

    def run(self) -> None:
        """Main thread execution."""
        # Separate batch-eligible files from non-batch files
        batch_files: list[Path] = []
        single_files: list[Path] = []

        for f in self.files:
            ext = f.suffix.lower()
            # Only static images without alpha requirement are batch-eligible
            if self.use_batch_mode and ext in BATCH_ELIGIBLE_EXTENSIONS and not self.use_alpha:
                batch_files.append(f)
            else:
                single_files.append(f)

        print(f"Batch mode: {self.use_batch_mode}, Alpha: {self.use_alpha}")
        print(f"Total files: {len(self.files)}, Batch eligible: {len(batch_files)}, Single: {len(single_files)}")

        total_files = len(self.files)
        total_processing_time = 0.0
        num_timed = 0
        current_avg = 0.0
        processed_count = 0

        # Process batch-eligible files first
        if batch_files and not self._cancel_flag:
            batch_time, batch_count = self._run_batch_processing(batch_files, total_files, processed_count)
            total_processing_time += batch_time
            num_timed += batch_count
            processed_count += batch_count
            if batch_count > 0:
                current_avg = total_processing_time / num_timed

        # Process remaining files individually
        if single_files and not self._cancel_flag:
            single_time, single_count, current_avg = self._run_single_processing(
                single_files, total_files, processed_count, current_avg
            )
            total_processing_time += single_time
            num_timed += single_count

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

    def _run_batch_processing(
        self,
        files: list[Path],
        total_files: int,
        start_idx: int,
    ) -> tuple[float, int]:
        """
        Run batch processing for eligible files.

        Spawns separate subprocess for each chunk of MAX_BATCH_SIZE files
        to ensure VRAM is fully released between chunks.

        Returns:
            Tuple of (total_time, num_processed).
        """
        if not files:
            return 0.0, 0

        # Split files into chunks to spawn separate processes for each
        # This ensures VRAM is released when each subprocess exits
        chunks = [files[i:i + MAX_BATCH_SIZE] for i in range(0, len(files), MAX_BATCH_SIZE)]

        total_batch_time = 0.0
        total_processed = 0

        for chunk_idx, chunk_files in enumerate(chunks):
            if self._cancel_flag:
                break

            chunk_time, chunk_count = self._run_single_batch_chunk(
                chunk_files,
                total_files,
                start_idx + total_processed,
                chunk_idx + 1,
                len(chunks),
            )
            total_batch_time += chunk_time
            total_processed += chunk_count

        return total_batch_time, total_processed

    def _run_single_batch_chunk(
        self,
        files: list[Path],
        total_files: int,
        start_idx: int,
        chunk_num: int,
        total_chunks: int,
    ) -> tuple[float, int]:
        """
        Run a single batch chunk in a subprocess.

        Each chunk gets its own subprocess to ensure VRAM is released on exit.

        Returns:
            Tuple of (time, num_processed).
        """
        if not files:
            return 0.0, 0

        script_path = self._get_script_path()
        batch_start = time.perf_counter()

        # Emit initial progress
        chunk_label = f" (chunk {chunk_num}/{total_chunks})" if total_chunks > 1 else ""
        self.progress_signal.emit(start_idx, total_files, 0.0, f"Batch: {len(files)} images{chunk_label}", "")

        # Compute output dirs for all files
        output_dirs: list[Path] = []
        secondary_dirs: list[Path] = []
        for f in files:
            out_dir, sec_dir = self._compute_output_dirs(f)
            self._ensure_dirs(out_dir, sec_dir)
            output_dirs.append(out_dir)
            secondary_dirs.append(sec_dir)

        # Create manifest file with progress tracking
        WORKER_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        manifest_path = WORKER_TMP_ROOT / f"manifest_{time.time_ns()}.json"
        progress_file = manifest_path.with_suffix(".progress")
        manifest = {
            "files": [str(f) for f in files],
            "output_dirs": [str(d) for d in output_dirs],
            "secondary_dirs": [str(d) for d in secondary_dirs],
            "progress_file": str(progress_file),
        }
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest, mf)

        # Build environment
        env = self._build_worker_env()

        # Run batch worker
        python_exe = get_pythonw_executable()
        cmd = [
            python_exe,
            str(script_path),
            "--batch-worker",
            str(manifest_path),
        ]

        try:
            process = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=CREATE_NO_WINDOW,
            )

            # Poll for progress while process runs
            last_progress_idx = 0
            poll_interval = 0.25  # Check every 250ms
            while process.poll() is None:
                # Check for cancellation
                if self._cancel_flag:
                    process.terminate()
                    break

                # Read progress file
                if progress_file.exists():
                    try:
                        with open(progress_file, "r", encoding="utf-8") as pf:
                            content = pf.read().strip()
                        if content:
                            parts = content.split(",", 2)
                            if len(parts) >= 3:
                                current_idx = int(parts[0])
                                total = int(parts[1])
                                filename = parts[2]
                                if current_idx > last_progress_idx:
                                    last_progress_idx = current_idx
                                    # Calculate average time so far
                                    elapsed = time.perf_counter() - batch_start
                                    avg_time = elapsed / current_idx if current_idx > 0 else 0.0
                                    self.progress_signal.emit(
                                        start_idx + current_idx,
                                        total_files,
                                        avg_time,
                                        filename,
                                        "",
                                    )
                    except Exception:
                        pass

                time.sleep(poll_interval)

            # Get output after process completes
            stdout, stderr = process.communicate()

            if stdout:
                print(f"Batch worker output:\n{stdout}")
            if process.returncode != 0:
                print(f"Batch worker failed with code {process.returncode}")
                if stderr:
                    print(f"Batch worker stderr:\n{stderr}")
        except Exception as e:
            print(f"Batch worker exception: {e}")

        # Cleanup progress file
        try:
            if progress_file.exists():
                progress_file.unlink()
        except Exception:
            pass

        # Read timing results if available
        result_path = manifest_path.with_suffix(".result")
        times: list[float] = []
        if result_path.exists():
            try:
                with open(result_path, "r", encoding="utf-8") as rf:
                    result_data = json.load(rf)
                    times = result_data.get("times", [])
                result_path.unlink()
            except Exception:
                pass

        # Cleanup manifest
        try:
            manifest_path.unlink()
        except Exception:
            pass

        batch_time = time.perf_counter() - batch_start
        num_processed = len(files)

        # Emit final progress for batch
        avg_time = batch_time / num_processed if num_processed > 0 else 0.0
        self.progress_signal.emit(
            start_idx + num_processed,
            total_files,
            avg_time,
            f"Batch complete",
            "",
        )

        return batch_time, num_processed

    def _run_single_processing(
        self,
        files: list[Path],
        total_files: int,
        start_idx: int,
        current_avg: float,
    ) -> tuple[float, int, float]:
        """
        Run single-file processing for non-batch files.

        Returns:
            Tuple of (total_time, num_processed, updated_avg).
        """
        script_path = self._get_script_path()
        total_time = 0.0
        num_processed = 0

        for idx, f in enumerate(files):
            if self._cancel_flag:
                break

            global_idx = start_idx + idx + 1

            # Emit thumbnail and initial progress
            self.thumbnail_signal.emit(str(f))
            self.progress_signal.emit(global_idx - 1, total_files, current_avg, f.name, str(f))

            # Determine per-file output directories
            per_output_dir, per_secondary_dir = self._compute_output_dirs(f)
            self._ensure_dirs(per_output_dir, per_secondary_dir)

            # Get input file info for format detection
            input_ext = f.suffix.lower()
            input_duration = 0.0
            input_fps = 0.0
            if input_ext in SUPPORTED_VIDEO_EXTENSIONS or input_ext == ".gif":
                input_duration = get_video_duration(f)
                input_fps = get_video_fps(f)

            # Build environment for worker
            env = self._build_worker_env(input_ext, input_duration, input_fps)

            file_start = time.perf_counter()

            # Run main worker
            self._run_worker(script_path, "--worker", f, per_output_dir, per_secondary_dir, env)

            # Run alpha worker if enabled
            if self.use_alpha and not self._cancel_flag:
                self._run_worker(script_path, "--alpha-worker", f, per_output_dir, per_secondary_dir, env)

            # Use wall-clock time for accurate ETA
            file_elapsed = time.perf_counter() - file_start
            total_time += file_elapsed
            num_processed += 1

            # Update running average (including batch files)
            total_processed = start_idx + num_processed
            current_avg = (current_avg * start_idx + total_time) / total_processed if total_processed > 0 else 0.0
            self.progress_signal.emit(global_idx, total_files, current_avg, f.name, str(f))

        return total_time, num_processed, current_avg

    def _get_script_path(self) -> Path:
        """Get the path to the main entry point script."""
        # When running as a package, we need to find the entry point
        # First try the installed entry point, then fall back to __main__.py
        main_module = sys.modules.get("__main__")
        if main_module and hasattr(main_module, "__file__") and main_module.__file__:
            return Path(main_module.__file__).resolve()
        # Fallback to package's __main__.py
        return Path(__file__).parent.parent / "__main__.py"

    def _get_secondary_folder_name(self) -> str:
        """
        Get the secondary output folder name based on settings.

        Returns:
            - "Downscaled" if 2x mode is selected
            - "{width}P" for width mode
            - "{height}P" for height mode
        """
        if self.secondary_mode == "2x":
            return "Downscaled"
        elif self.secondary_mode == "height":
            return f"{self.secondary_height}P"
        else:
            # Width mode (default)
            return f"{self.secondary_width}P"

    def _compute_output_dirs(self, f: Path) -> tuple[Path, Path]:
        """Compute output directories for a specific file."""
        secondary_folder = self._get_secondary_folder_name()

        if self.manga_folder_enabled:
            # Manga folder mode: Parent Folder_suffix/Subfolder/.../
            per_output_dir = self._compute_manga_output_dir(f)
            per_secondary_dir = per_output_dir.parent / secondary_folder / per_output_dir.name
        elif self.same_dir_enabled:
            per_output_dir = self.output_dir  # Placeholder; worker uses input.parent
            base_dir = f.parent if f.is_file() else f
            per_secondary_dir = base_dir / secondary_folder
        else:
            if self.single_input_is_file:
                per_output_dir = self.output_dir
                per_secondary_dir = self.secondary_output_dir
            else:
                base_dir = f.parent if f.is_file() else f
                per_output_dir = base_dir / "Upscaled"
                per_secondary_dir = base_dir / secondary_folder

        return per_output_dir, per_secondary_dir

    def _compute_manga_output_dir(self, f: Path) -> Path:
        """
        Compute the manga folder output directory for a file.

        Input:  Parent Folder/Subfolder/Subfolder/Image001.png
        Output: Parent Folder_suffix/Subfolder/Subfolder/

        The parent folder that gets the suffix is determined by finding which
        input_root the file belongs to.
        """
        suffix = self.same_dir_suffix or "_upscaled"

        # Find which input root this file belongs to
        for root in self.input_roots:
            if root.is_file():
                # Single file input - use its parent with suffix
                if f == root:
                    return root.parent.parent / f"{root.parent.name}{suffix}"
            else:
                # Folder input - check if file is under this folder
                try:
                    rel_path = f.relative_to(root)
                    # Output: root.parent / (root.name + suffix) / rel_path.parent
                    manga_root = root.parent / f"{root.name}{suffix}"
                    return manga_root / rel_path.parent
                except ValueError:
                    # File is not under this root
                    continue

        # Fallback: use file's grandparent with suffix
        return f.parent.parent / f"{f.parent.name}{suffix}"

    def _ensure_dirs(self, output_dir: Path, secondary_dir: Path) -> None:
        """Ensure output directories exist."""
        try:
            if self.manga_folder_enabled or not self.same_dir_enabled:
                output_dir.mkdir(parents=True, exist_ok=True)
            if self.secondary_enabled:
                secondary_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Warning: Failed to create per-file dirs: {e}")

    def _build_worker_env(self, input_ext: str = ".png", input_duration: float = 0.0, input_fps: float = 0.0) -> dict[str, str]:
        """Build environment variables for worker process."""
        env = os.environ.copy()
        env["ONNX_PATH"] = self.onnx_path
        env["TILE_W_LIMIT"] = self.tile_w
        env["TILE_H_LIMIT"] = self.tile_h
        env["MODEL_SCALE"] = self.model_scale
        env["USE_FP16"] = "1" if self.use_fp16 else "0"
        env["USE_BF16"] = "1" if self.use_bf16 else "0"
        env["USE_TF32"] = "1" if self.use_tf32 else "0"
        env["NUM_STREAMS"] = str(self.num_streams)
        env["USE_SAME_DIR_OUTPUT"] = "1" if self.same_dir_enabled else "0"
        env["SAME_DIR_SUFFIX"] = self.same_dir_suffix
        env["MANGA_FOLDER_ENABLED"] = "1" if self.manga_folder_enabled else "0"
        env["OVERWRITE_OUTPUT"] = "1" if self.overwrite_enabled else "0"
        env["CUSTOM_RES_ENABLED"] = "1" if self.custom_res_enabled else "0"
        env["CUSTOM_RES_MODE"] = self.custom_res_mode
        env["CUSTOM_WIDTH"] = str(self.custom_width)
        env["CUSTOM_HEIGHT"] = str(self.custom_height)
        env["CUSTOM_RES_KERNEL"] = self.custom_res_kernel
        env["USE_SECONDARY_OUTPUT"] = "1" if self.secondary_enabled else "0"
        env["SECONDARY_MODE"] = self.secondary_mode
        env["SECONDARY_WIDTH"] = str(self.secondary_width)
        env["SECONDARY_HEIGHT"] = str(self.secondary_height)
        env["SECONDARY_KERNEL"] = self.secondary_kernel
        env["USE_ALPHA"] = "1" if self.use_alpha else "0"
        env["APPEND_MODEL_SUFFIX"] = "1" if self.append_model_suffix_enabled else "0"
        env["PRESCALE_ENABLED"] = "1" if self.prescale_enabled else "0"
        env["PRESCALE_MODE"] = self.prescale_mode
        env["PRESCALE_WIDTH"] = str(self.prescale_width)
        env["PRESCALE_HEIGHT"] = str(self.prescale_height)
        env["PRESCALE_KERNEL"] = self.prescale_kernel
        env["SHARPEN_ENABLED"] = "1" if self.sharpen_enabled else "0"
        env["SHARPEN_VALUE"] = str(self.sharpen_value)
        env["INPUT_EXTENSION"] = input_ext
        env["INPUT_DURATION"] = str(input_duration)
        env["INPUT_FPS"] = str(input_fps)
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

        # Log file for worker output (useful for debugging)
        log_file = TEMP_BASE / "worker_debug.log"

        # On Windows, use CREATE_NO_WINDOW flag as additional safeguard
        # Redirect stdout/stderr to log file for debugging
        if sys.platform == "win32":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"\n{'='*60}\n")
                log.write(f"Worker: {mode} | File: {input_file.name}\n")
                log.write(f"{'='*60}\n")
                log.flush()

                subprocess.run(
                    cmd,
                    check=False,
                    env=env,
                    creationflags=CREATE_NO_WINDOW,
                    startupinfo=startupinfo,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
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
        custom_res_mode: str = "width",
        custom_width: int = 0,
        custom_height: int = 0,
        custom_res_kernel: str = "lanczos",
        prescale_enabled: bool = False,
        prescale_mode: str = "width",
        prescale_width: int = 1920,
        prescale_height: int = 1080,
        prescale_kernel: str = "lanczos",
        sharpen_enabled: bool = False,
        sharpen_value: float = 0.5,
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
        self.custom_res_mode = custom_res_mode
        self.custom_width = custom_width
        self.custom_height = custom_height
        self.custom_res_kernel = custom_res_kernel
        self.prescale_enabled = prescale_enabled
        self.prescale_mode = prescale_mode
        self.prescale_width = prescale_width
        self.prescale_height = prescale_height
        self.prescale_kernel = prescale_kernel
        self.sharpen_enabled = sharpen_enabled
        self.sharpen_value = sharpen_value

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
        env["CUSTOM_RES_MODE"] = self.custom_res_mode
        env["CUSTOM_WIDTH"] = str(self.custom_width)
        env["CUSTOM_HEIGHT"] = str(self.custom_height)
        env["CUSTOM_RES_KERNEL"] = self.custom_res_kernel
        env["USE_SECONDARY_OUTPUT"] = "0"  # No secondary for clipboard
        env["SECONDARY_MODE"] = "width"
        env["SECONDARY_WIDTH"] = "0"
        env["SECONDARY_HEIGHT"] = "0"
        env["SECONDARY_KERNEL"] = "lanczos"
        env["USE_ALPHA"] = "1" if self.use_alpha else "0"
        env["APPEND_MODEL_SUFFIX"] = "0"  # No suffix for clipboard
        env["PRESCALE_ENABLED"] = "1" if self.prescale_enabled else "0"
        env["PRESCALE_MODE"] = self.prescale_mode
        env["PRESCALE_WIDTH"] = str(self.prescale_width)
        env["PRESCALE_HEIGHT"] = str(self.prescale_height)
        env["PRESCALE_KERNEL"] = self.prescale_kernel
        env["SHARPEN_ENABLED"] = "1" if self.sharpen_enabled else "0"
        env["SHARPEN_VALUE"] = str(self.sharpen_value)
        env["INPUT_EXTENSION"] = self.input_file.suffix.lower()
        env["INPUT_DURATION"] = "0.0"
        env["INPUT_FPS"] = "0.0"
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
