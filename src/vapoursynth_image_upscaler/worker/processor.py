"""
File processing logic for worker mode.

Handles the complete workflow of processing a single file, including:
- Building SR clips
- Writing outputs to temporary locations (using imwri.Write)
- Moving to final destinations
- Handling filename conflicts
- Supporting multiple output formats (PNG, JPEG, GIF, etc.)
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from vstools import vs, core, depth
from vskernels import Hermite, Lanczos

from ..core.constants import (
    WORKER_TMP_ROOT,
    SUPPORTED_VIDEO_EXTENSIONS,
    MAX_VIDEO_DURATION_FOR_GIF,
)
from ..core.utils import write_time_file
from .settings import WorkerSettings
from .pipeline import (
    build_clip,
    build_alpha_hq,
    apply_custom_resolution,
    apply_sharpening,
    compute_secondary_dimensions,
    clear_cache,
    get_process_start_time,
)
from .progress import track

if TYPE_CHECKING:
    pass


def _determine_output_format(settings: WorkerSettings) -> tuple[str, str]:
    """
    Determine the output format based on input file type.

    Returns:
        Tuple of (imgformat for imwri, file_extension).
    """
    input_ext = settings.input_extension.lower()
    input_duration = settings.input_duration

    # GIF input -> GIF output
    if input_ext == ".gif":
        return "GIF", ".gif"

    # Video input under 5 minutes -> GIF output
    if input_ext in SUPPORTED_VIDEO_EXTENSIONS:
        if 0 < input_duration <= MAX_VIDEO_DURATION_FOR_GIF:
            return "GIF", ".gif"

    # JPEG inputs -> JPEG output
    if input_ext in {".jpg", ".jpeg"}:
        return "JPEG", ".jpg"

    # TIFF inputs -> TIFF output
    if input_ext in {".tif", ".tiff"}:
        return "TIFF", ".tif"

    # WebP inputs -> WebP output (note: imwri might not support WebP, fallback to PNG)
    if input_ext == ".webp":
        return "WEBP", ".webp"

    # BMP inputs -> BMP output
    if input_ext == ".bmp":
        return "BMP", ".bmp"

    # Default to PNG for everything else
    return "PNG", ".png"


def _get_frame_delay_ms(settings: WorkerSettings) -> int:
    """
    Get the frame delay in milliseconds for animated GIF output.

    Args:
        settings: Worker settings with input_fps.

    Returns:
        Frame delay in milliseconds (minimum 20ms for GIF compatibility).
    """
    if settings.input_fps > 0:
        delay = int(1000.0 / settings.input_fps)
        # GIF has minimum delay of 20ms (some viewers may not handle < 20ms)
        return max(20, delay)
    return 100  # Default to 100ms (10fps)


def _write_clip_imwri(
    clip: vs.VideoNode,
    output_path: Path,
    imgformat: str,
    alpha_clip: vs.VideoNode | None = None,
    quality: int = 95,
    prefix: str = "Frames",
    frame_delay_ms: int = 100,
) -> None:
    """
    Write a clip to disk using imwri.Write.

    For animated GIF output, writes individual frames then combines them.

    Args:
        clip: The clip to write (should be RGB format).
        output_path: Path to the output file.
        imgformat: Image format name for imwri (e.g., "PNG", "JPEG", "GIF").
        alpha_clip: Optional alpha channel clip (GRAY format).
        quality: Output quality (0-100).
        prefix: Progress prefix for tracking.
        frame_delay_ms: Delay between frames for animated GIF (in milliseconds).
    """
    num_frames = len(clip)

    # For animated GIF (multi-frame), write frames then combine
    if imgformat == "GIF" and num_frames > 1:
        _write_animated_gif(clip, output_path, alpha_clip, prefix, frame_delay_ms)
        return

    # Convert to appropriate bit depth based on format
    if imgformat in {"JPEG", "GIF", "BMP"}:
        # These formats only support 8-bit
        clip = core.resize.Point(clip, format=vs.RGB24)
        if alpha_clip is not None:
            alpha_clip = core.resize.Point(alpha_clip, format=vs.GRAY8)
    elif imgformat == "PNG":
        # PNG supports 8-bit or 16-bit
        clip = core.resize.Point(clip, format=vs.RGB24)
        if alpha_clip is not None:
            alpha_clip = core.resize.Point(alpha_clip, format=vs.GRAY8)
    else:
        # Default to 8-bit for other formats
        clip = core.resize.Point(clip, format=vs.RGB24)
        if alpha_clip is not None:
            alpha_clip = core.resize.Point(alpha_clip, format=vs.GRAY8)

    # Use imwri.Write
    sink = core.imwri.Write(
        clip,
        imgformat=imgformat,
        filename=str(output_path),
        quality=quality,
        alpha=alpha_clip,
        overwrite=True,
    )

    for _ in track(sink.frames(close=True), total=len(sink), prefix=prefix):
        pass


def _write_animated_gif(
    clip: vs.VideoNode,
    output_path: Path,
    alpha_clip: vs.VideoNode | None = None,
    prefix: str = "Frames",
    frame_delay_ms: int = 100,
) -> None:
    """
    Write an animated GIF using Pillow.

    Args:
        clip: The clip to write (multi-frame).
        output_path: Path to the output GIF file.
        alpha_clip: Optional alpha channel clip (GRAY format).
        prefix: Progress prefix for tracking.
        frame_delay_ms: Delay between frames in milliseconds.
    """
    from PIL import Image
    import numpy as np

    # Convert clip to RGB24
    clip = core.resize.Point(clip, format=vs.RGB24)

    # Prepare alpha if provided
    if alpha_clip is not None:
        alpha_clip = core.resize.Point(alpha_clip, format=vs.GRAY8)

    frames: list[Image.Image] = []
    num_frames = len(clip)

    print(f"{prefix}: Extracting {num_frames} frames...")

    for i, frame in enumerate(clip.frames(close=True)):
        # Extract frame data
        frame_data = np.asarray(frame[0])  # R plane
        r = frame_data.copy()
        g = np.asarray(frame[1]).copy()
        b = np.asarray(frame[2]).copy()

        # Stack RGB
        rgb = np.stack([r, g, b], axis=-1)

        # Create PIL image
        if alpha_clip is not None:
            # Get corresponding alpha frame
            alpha_frame = alpha_clip.get_frame(i)
            alpha_data = np.asarray(alpha_frame[0]).copy()
            rgba = np.concatenate([rgb, alpha_data[:, :, np.newaxis]], axis=-1)
            img = Image.fromarray(rgba, mode="RGBA")
        else:
            img = Image.fromarray(rgb, mode="RGB")

        # Convert to palette mode for GIF (better quality)
        img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        frames.append(img)

        if (i + 1) % 10 == 0 or i == num_frames - 1:
            print(f"{prefix}: {i + 1}/{num_frames} frames extracted")

    if not frames:
        print(f"Warning: No frames to write for animated GIF")
        return

    print(f"{prefix}: Saving animated GIF...")

    # Save animated GIF
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:] if len(frames) > 1 else [],
        duration=frame_delay_ms,
        loop=0,  # 0 = infinite loop
        optimize=True,
    )

    print(f"{prefix}: Animated GIF saved with {len(frames)} frames")


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

    # Determine output format based on input
    imgformat, file_ext = _determine_output_format(settings)

    # Create temporary working directory with ASCII-only name
    WORKER_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    job_id = f"job_{uuid.uuid4().hex}"
    tmp_dir = WORKER_TMP_ROOT / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    tmp_main = tmp_dir / f"output_main{file_ext}"
    tmp_secondary = tmp_dir / f"output_secondary{file_ext}" if settings.use_secondary_output else None

    try:
        # Build SR clip
        clip_sr = build_clip(input_path, settings)
        clip_main = clip_sr

        # Apply custom main resolution if enabled
        if settings.custom_res_enabled and settings.custom_width > 0 and settings.custom_height > 0:
            clip_main = apply_custom_resolution(
                clip_main, settings.custom_width, settings.custom_height, settings.kernel
            )

        # Apply sharpening at the end of main output processing
        clip_main = apply_sharpening(clip_main, settings)

        # Get frame delay for animated GIF
        frame_delay = _get_frame_delay_ms(settings)

        # Write main output using imwri.Write
        _write_clip_imwri(clip_main, tmp_main, imgformat, prefix="Frames (main)", frame_delay_ms=frame_delay)

        # Process secondary output if enabled
        # Note: Secondary output is only supported for single-frame images
        if settings.use_secondary_output and tmp_secondary is not None:
            if imgformat == "GIF" and len(clip_main) > 1:
                print("Note: Secondary output is not supported for animated GIFs, skipping")
            else:
                _process_secondary(tmp_main, tmp_secondary, settings, imgformat, file_ext)

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
            file_ext,
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
    - Reads the existing output file
    - Rewrites that file in-place with the HQ alpha

    This runs in a separate process to avoid vsmlrt/TensorRT overlap.

    Args:
        input_path: Path to the original input file.
        output_dir: Directory where main output was written.
        secondary_output_dir: Directory for secondary output (unused here).
        settings: Worker settings from environment.
    """
    file_start = time.perf_counter()
    base_name = input_path.stem

    # Determine output format (same as main worker)
    imgformat, file_ext = _determine_output_format(settings)

    # Only process alpha for formats that support it
    if imgformat not in {"PNG", "GIF", "WEBP", "TIFF"}:
        print(f"[alpha-worker] Format {imgformat} doesn't support alpha, skipping")
        return

    # Resolve main destination path (same logic as main worker)
    model_suffix = settings.get_model_suffix()

    if settings.use_same_dir_output:
        suffix = settings.same_dir_suffix or ""
        dest_stem = f"{base_name}{suffix}{model_suffix}"
        dest_dir_main = input_path.parent
    else:
        dest_stem = f"{base_name}{model_suffix}"
        dest_dir_main = output_dir

    dest_name = f"{dest_stem}{file_ext}"
    dest_path = dest_dir_main / dest_name

    if not dest_path.exists():
        print(f"[alpha-worker] Main output not found, skipping alpha: {dest_path}")
        return

    try:
        # Build SR alpha clip
        alpha_sr = build_alpha_hq(input_path, settings)

        # Read color main output
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

        # Rewrite output in-place with alpha using imwri.Write
        _write_clip_imwri(
            color_sr,
            dest_path,
            imgformat,
            alpha_clip=alpha_sr,
            prefix="Merge color + HQ alpha",
        )

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
    imgformat: str = "PNG",
    file_ext: str = ".png",
) -> None:
    """Process secondary resized output from the main output."""
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

        # Use selected kernel for scaling
        if settings.kernel.lower() == "hermite":
            kernel = Hermite()
        else:
            kernel = Lanczos()

        clip_sec = kernel.scale(clip_sec, new_w, new_h, linear=True)

        # Apply sharpening at the end of secondary output processing
        clip_sec = apply_sharpening(clip_sec, settings)

        # Write secondary output using imwri.Write
        _write_clip_imwri(clip_sec, secondary_output, imgformat, prefix="Frames (secondary)")

    except Exception as e:
        print(f"Warning: Secondary resize failed: {e}")


def _resolve_output_paths(
    dest_stem: str,
    dest_dir_main: Path,
    dest_dir_secondary: Path | None,
    overwrite: bool,
    file_ext: str = ".png",
) -> tuple[Path, Path | None]:
    """
    Resolve final output paths, handling overwrite and conflict resolution.

    Returns paths that are free across all enabled outputs.
    """
    if overwrite:
        dest_name = f"{dest_stem}{file_ext}"
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

    # Find first free name: base{ext}, then _002{ext}, _003{ext}, etc.
    index = 1
    while True:
        if index == 1:
            candidate_name = f"{dest_stem}{file_ext}"
        else:
            candidate_name = f"{dest_stem}_{index:03d}{file_ext}"

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
