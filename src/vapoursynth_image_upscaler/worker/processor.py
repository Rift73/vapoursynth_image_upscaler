"""
File processing logic for worker mode.

Handles the complete workflow of processing a single file, including:
- Building SR clips
- Writing outputs to temporary locations
- Moving to final destinations
- Handling filename conflicts
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from vstools import vs, core, depth
from vskernels import Hermite

from ..core.constants import WORKER_TMP_ROOT
from ..core.utils import write_time_file
from .settings import WorkerSettings
from .pipeline import (
    build_clip,
    build_alpha_hq,
    apply_custom_resolution,
    compute_secondary_dimensions,
    clear_cache,
    get_process_start_time,
)
from .progress import track

if TYPE_CHECKING:
    pass


def process_one(
    input_path: Path,
    output_dir: Path,
    secondary_output_dir: Path,
    settings: WorkerSettings,
) -> None:
    """
    Process a single file in main worker mode (color only).

    Alpha is handled in a separate --alpha-worker process.

    Args:
        input_path: Path to the input file.
        output_dir: Directory for main output.
        secondary_output_dir: Directory for secondary output.
        settings: Worker settings from environment.
    """
    file_start = time.perf_counter()
    base_name = input_path.stem

    # Compute destination paths
    model_suffix = settings.get_model_suffix()

    if settings.use_same_dir_output:
        suffix = settings.same_dir_suffix or ""
        dest_stem = f"{base_name}{suffix}{model_suffix}"
        dest_dir_main = input_path.parent
    else:
        dest_stem = f"{base_name}{model_suffix}"
        dest_dir_main = output_dir

    dest_dir_secondary = secondary_output_dir

    # Create temporary working directory with ASCII-only name
    WORKER_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = f"job_{uuid.uuid4().hex}"
    tmp_dir = WORKER_TMP_ROOT / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_main = tmp_dir / "output_main.png"
    tmp_secondary = tmp_dir / "output_secondary.png" if settings.use_secondary_output else None

    try:
        # Build SR clip
        clip_sr = build_clip(input_path, settings)
        clip_main = clip_sr

        # Apply custom main resolution if enabled
        if settings.custom_res_enabled and settings.custom_width > 0 and settings.custom_height > 0:
            clip_main = apply_custom_resolution(clip_main, settings.custom_width, settings.custom_height)

        # Convert to RGB24 for output
        clip_main = core.resize.Point(clip_main, format=vs.RGB24)

        # Write main output
        main_sink = clip_main.fpng.Write(
            filename=str(tmp_main),
            alpha=None,
            overwrite=1,
        )
        for _ in track(main_sink.frames(close=True), total=len(main_sink), prefix="Frames (main)"):
            pass

        # Process secondary output if enabled
        if settings.use_secondary_output and tmp_secondary is not None:
            _process_secondary(tmp_main, tmp_secondary, settings)

        # Ensure destination directories exist
        if not settings.use_same_dir_output:
            dest_dir_main.mkdir(parents=True, exist_ok=True)
        if settings.use_secondary_output:
            dest_dir_secondary.mkdir(parents=True, exist_ok=True)

        # Determine final filenames (with conflict resolution)
        dest_path, dest_path_secondary = _resolve_output_paths(
            dest_stem,
            dest_dir_main,
            dest_dir_secondary if settings.use_secondary_output else None,
            settings.overwrite_output,
        )

        # Move temp files to final locations
        _move_output(tmp_main, dest_path)
        if settings.use_secondary_output and tmp_secondary is not None and dest_path_secondary is not None:
            _move_output(tmp_secondary, dest_path_secondary)

    finally:
        # Clean up temp directory
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

        # Clear caches
        clear_cache()

    # Write timing file
    process_start = get_process_start_time()
    if process_start is not None:
        processing_time = time.perf_counter() - process_start
    else:
        processing_time = time.perf_counter() - file_start

    write_time_file(base_name, processing_time)


def process_one_alpha(
    input_path: Path,
    output_dir: Path,
    secondary_output_dir: Path,
    settings: WorkerSettings,
) -> None:
    """
    Process alpha channel for a single file.

    Assumes the main color SR has already been written by the normal
    --worker process (without alpha). This process:
    - Computes SR alpha using vsmlrt
    - Reads the existing color PNG
    - Rewrites that PNG in-place with the HQ alpha

    This runs in a separate process to avoid vsmlrt/TensorRT overlap.

    Args:
        input_path: Path to the original input file.
        output_dir: Directory where main output was written.
        secondary_output_dir: Directory for secondary output (unused here).
        settings: Worker settings from environment.
    """
    file_start = time.perf_counter()
    base_name = input_path.stem

    # Resolve main destination path (same logic as main worker)
    model_suffix = settings.get_model_suffix()

    if settings.use_same_dir_output:
        suffix = settings.same_dir_suffix or ""
        dest_stem = f"{base_name}{suffix}{model_suffix}"
        dest_dir_main = input_path.parent
    else:
        dest_stem = f"{base_name}{model_suffix}"
        dest_dir_main = output_dir

    dest_name = f"{dest_stem}.png"
    dest_path = dest_dir_main / dest_name

    if not dest_path.exists():
        print(f"[alpha-worker] Main PNG not found, skipping alpha: {dest_path}")
        return

    try:
        # Build SR alpha clip
        alpha_sr = build_alpha_hq(input_path, settings)

        # Read color main PNG
        color_sr = core.imwri.Read(str(dest_path))

        # Match alpha to color resolution and format
        alpha_sr = depth(alpha_sr, 16)
        alpha_sr = core.resize.Bicubic(
            alpha_sr,
            width=color_sr.width,
            height=color_sr.height,
            format=vs.GRAY8,
            matrix=1,
            range_in=1,
            range=1,
        )

        # Rewrite main PNG in-place with alpha
        sink = color_sr.fpng.Write(
            filename=str(dest_path),
            alpha=alpha_sr,
            overwrite=1,
        )
        for _ in track(sink.frames(close=True), total=len(sink), prefix="Merge color + HQ alpha"):
            pass

    except Exception as e:
        print(f"[alpha-worker] Alpha upscale failed: {e}")

    finally:
        clear_cache()

    # Update timing file (add alpha time to main time)
    process_start = get_process_start_time()
    if process_start is not None:
        alpha_time = time.perf_counter() - process_start
    else:
        alpha_time = time.perf_counter() - file_start

    _update_time_file_alpha(base_name, alpha_time)


def _process_secondary(
    main_output: Path,
    secondary_output: Path,
    settings: WorkerSettings,
) -> None:
    """Process secondary resized output from the main PNG."""
    try:
        clip_sec_src = core.imwri.Read(str(main_output))
        clip_sec = depth(clip_sec_src, 32)

        new_w, new_h = compute_secondary_dimensions(
            clip_sec.width,
            clip_sec.height,
            settings.secondary_mode,
            settings.secondary_width,
            settings.secondary_height,
        )

        clip_sec = Hermite().scale(clip_sec, new_w, new_h, linear=True)
        clip_sec = core.resize.Point(clip_sec, format=vs.RGB24)

        sec_sink = clip_sec.fpng.Write(
            filename=str(secondary_output),
            alpha=None,
            overwrite=1,
        )
        for _ in track(sec_sink.frames(close=True), total=len(sec_sink), prefix="Frames (secondary)"):
            pass
    except Exception as e:
        print(f"Warning: Secondary resize failed: {e}")


def _resolve_output_paths(
    dest_stem: str,
    dest_dir_main: Path,
    dest_dir_secondary: Path | None,
    overwrite: bool,
) -> tuple[Path, Path | None]:
    """
    Resolve final output paths, handling overwrite and conflict resolution.

    Returns paths that are free across all enabled outputs.
    """
    if overwrite:
        dest_name = f"{dest_stem}.png"
        dest_path = dest_dir_main / dest_name
        dest_path_secondary = dest_dir_secondary / dest_name if dest_dir_secondary else None

        # Remove existing files
        if dest_path.exists():
            try:
                dest_path.unlink()
            except Exception:
                pass
        if dest_path_secondary and dest_path_secondary.exists():
            try:
                dest_path_secondary.unlink()
            except Exception:
                pass

        return dest_path, dest_path_secondary

    # Find first free name: base.png, then _002, _003, etc.
    index = 1
    while True:
        if index == 1:
            candidate_name = f"{dest_stem}.png"
        else:
            candidate_name = f"{dest_stem}_{index:03d}.png"

        candidate_main = dest_dir_main / candidate_name
        candidate_secondary = dest_dir_secondary / candidate_name if dest_dir_secondary else None

        conflicts = candidate_main.exists()
        if candidate_secondary and candidate_secondary.exists():
            conflicts = True

        if not conflicts:
            return candidate_main, candidate_secondary

        index += 1


def _move_output(src: Path, dest: Path) -> None:
    """Move a file from temp location to final destination."""
    if not src.exists():
        print(f"Warning: No output written at {src}")
        return
    try:
        shutil.move(str(src), str(dest))
    except Exception as e:
        print(f"Warning: Failed to move {src} -> {dest}: {e}")


def _update_time_file_alpha(base_name: str, alpha_time: float) -> None:
    """Update timing file by adding alpha processing time to existing main time."""
    try:
        time_file = WORKER_TMP_ROOT / f"{base_name}.time"

        main_time = 0.0
        if time_file.exists():
            with open(time_file, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            try:
                main_time = float(txt)
            except ValueError:
                main_time = 0.0

        combined = main_time + alpha_time
        WORKER_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        with open(time_file, "w", encoding="utf-8") as f:
            f.write(f"{combined:.6f}")
    except Exception as e:
        print(f"[alpha-worker] Could not update time file: {e}")
