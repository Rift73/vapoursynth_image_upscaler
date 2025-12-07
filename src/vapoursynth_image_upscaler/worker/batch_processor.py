"""
Batch processing logic for worker mode.

Processes multiple images as a single sequence to avoid VRAM leaks.
Images are grouped by resolution and format, then processed through
vsmlrt as a continuous stream.
"""

from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from ..core.constants import WORKER_TMP_ROOT, MAX_BATCH_SIZE
from ..core.utils import optimize_png

if TYPE_CHECKING:
    from .settings import WorkerSettings


def get_image_info(path: Path) -> tuple[int, int, str]:
    """
    Get image dimensions and normalized format.

    Returns:
        Tuple of (width, height, format).
        Format is normalized: .jpeg -> .jpg
    """
    try:
        with Image.open(str(path)) as img:
            ext = path.suffix.lower()
            # Normalize .jpeg to .jpg
            if ext == ".jpeg":
                ext = ".jpg"
            return img.size[0], img.size[1], ext
    except Exception:
        return (0, 0, "")


def group_by_resolution_and_format(
    files: list[Path],
) -> dict[tuple[int, int, str], list[Path]]:
    """
    Group files by their resolution and format.

    JPGs and PNGs are kept separate to avoid conversion.
    """
    groups: dict[tuple[int, int, str], list[Path]] = {}
    for f in files:
        w, h, fmt = get_image_info(f)
        if w > 0 and h > 0 and fmt:
            key = (w, h, fmt)
            if key not in groups:
                groups[key] = []
            groups[key].append(f)
    return groups


def split_into_batches(files: list[Path], max_size: int = MAX_BATCH_SIZE) -> list[list[Path]]:
    """Split a list of files into batches of max_size."""
    batches = []
    for i in range(0, len(files), max_size):
        batches.append(files[i:i + max_size])
    return batches


def prepare_sequence(
    files: list[Path],
    temp_dir: Path,
    fmt: str,
) -> tuple[str, dict[int, Path]]:
    """
    Prepare a sequence of images for batch processing.

    Copies files with sequential naming, preserving format.

    Args:
        files: List of input file paths.
        temp_dir: Temporary directory for the sequence.
        fmt: File format extension (e.g., ".png", ".jpg").

    Returns:
        Tuple of (sequence_pattern, frame_to_file_mapping).
    """
    temp_dir.mkdir(parents=True, exist_ok=True)
    frame_map: dict[int, Path] = {}

    for idx, src_path in enumerate(files):
        frame_map[idx] = src_path
        dest_path = temp_dir / f"seq_{idx:06d}{fmt}"
        # Copy file directly (no conversion)
        shutil.copy2(str(src_path), str(dest_path))

    pattern = str(temp_dir / f"seq_%06d{fmt}")
    return pattern, frame_map


def process_batch(
    files: list[Path],
    output_dirs: list[Path],
    secondary_dirs: list[Path],
    settings: "WorkerSettings",
    progress_file: Path | None = None,
    batch_offset: int = 0,
    total_files: int = 0,
) -> list[float]:
    """
    Process a batch of same-resolution, same-format images as a single sequence.

    Args:
        files: List of input file paths (all same resolution and format).
        output_dirs: List of output directories for each file.
        secondary_dirs: List of secondary output directories for each file.
        settings: Worker settings from environment.
        progress_file: Optional path to write progress updates for GUI tracking.
        batch_offset: Offset for progress reporting (files already processed).
        total_files: Total number of files across all batches.

    Returns:
        List of processing times per image.
    """
    # Deferred imports to avoid loading in GUI mode
    import vsmlrt
    from vstools import vs, core, depth, padder

    from ..core.utils import compute_padding
    from ..core.constants import PADDING_ALIGNMENT
    from .pipeline import (
        _build_backend,
        apply_prescale,
        apply_custom_resolution,
        apply_sharpening,
        compute_custom_resolution_dimensions,
        compute_secondary_dimensions,
        clear_cache,
    )

    if not files:
        return []

    batch_start = time.perf_counter()
    num_files = len(files)

    # Determine format from first file
    fmt = files[0].suffix.lower()
    if fmt == ".jpeg":
        fmt = ".jpg"

    # Create temp directory for this batch
    job_id = f"batch_{uuid.uuid4().hex}"
    temp_dir = WORKER_TMP_ROOT / job_id
    input_seq_dir = temp_dir / "input"
    output_seq_dir = temp_dir / "output"

    try:
        # Prepare input sequence
        print(f"Preparing batch of {num_files} {fmt} images...")
        seq_pattern, frame_map = prepare_sequence(files, input_seq_dir, fmt)

        # Load as image sequence
        input_files = [str(input_seq_dir / f"seq_{i:06d}{fmt}") for i in range(num_files)]
        print(f"Loading sequence: {len(input_files)} files")
        src = core.imwri.Read(input_files, mismatch=1)

        # Convert to RGBS
        clip = core.resize.Bicubic(
            src, format=vs.RGBS,
            matrix_in=0, transfer_in=13, primaries_in=1,
            range_in=1, range=1
        )

        # Apply pre-scaling if enabled
        clip = apply_prescale(clip, settings)

        # Compute and apply padding
        pad_left, pad_right, pad_top, pad_bottom = compute_padding(
            clip.width, clip.height, PADDING_ALIGNMENT
        )
        clip = padder.MIRROR(clip, pad_left, pad_right, pad_top, pad_bottom)

        # Clamp to valid range
        clip = core.std.Expr(clip, expr=["x 0 max 1 min"])

        # Compute tile sizes
        tile_h = min(settings.tile_h_limit, clip.height)
        tile_w = min(settings.tile_w_limit, clip.width)

        # Run through vsmlrt - single inference call for entire batch
        print(f"Running inference on {num_files} frames...")
        clip = vsmlrt.inference(
            clip,
            backend=_build_backend(settings),
            overlap=[16, 16],
            tilesize=[tile_w, tile_h],
            network_path=settings.onnx_path,
        )

        # Remove padding
        scale = settings.model_scale
        clip = core.std.Crop(
            clip,
            left=pad_left * scale,
            right=pad_right * scale,
            top=pad_top * scale,
            bottom=pad_bottom * scale,
        )

        # Apply custom resolution if enabled
        if settings.custom_res_enabled:
            custom_w, custom_h = compute_custom_resolution_dimensions(
                clip.width, clip.height,
                settings.custom_res_mode,
                settings.custom_width,
                settings.custom_height,
            )
            if custom_w > 0 and custom_h > 0:
                clip = apply_custom_resolution(
                    clip, custom_w, custom_h, settings.custom_res_kernel
                )

        # Apply sharpening
        clip = apply_sharpening(clip, settings)

        # Write output sequence
        output_seq_dir.mkdir(parents=True, exist_ok=True)
        secondary_seq_dir = temp_dir / "secondary"

        # Prepare secondary clip if enabled
        clip_secondary = None
        if settings.use_secondary_output:
            secondary_seq_dir.mkdir(parents=True, exist_ok=True)
            sec_w, sec_h = compute_secondary_dimensions(
                clip.width, clip.height,
                settings.secondary_mode,
                settings.secondary_width,
                settings.secondary_height,
            )
            if sec_w > 0 and sec_h > 0:
                from vskernels import Lanczos, Hermite
                sec_kernel = Hermite if settings.secondary_kernel == "hermite" else Lanczos
                clip_secondary = depth(clip, 32)
                clip_secondary = sec_kernel().scale(clip_secondary, sec_w, sec_h, linear=True)
                clip_secondary = core.resize.Point(clip_secondary, format=vs.RGB24)
                print(f"Secondary output enabled: {sec_w}x{sec_h}")

        # Convert main to RGB24 for output
        clip = core.resize.Point(clip, format=vs.RGB24)

        # Output format: PNG for lossless
        out_fmt = ".png"

        # Write frames and move to final destination immediately
        report_total = total_files if total_files > 0 else num_files
        print(f"Processing {num_files} output frames...")
        for idx in range(num_files):
            src_file = frame_map[idx]
            output_dir = output_dirs[idx]
            secondary_dir = secondary_dirs[idx]

            # Compute destination path using shared helper
            dest_stem, output_dir = settings.compute_dest_stem_and_dir(src_file, output_dir)

            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)

            # Determine final destination path for main output
            dest_name = f"{dest_stem}.png"
            dest_path = output_dir / dest_name

            # Handle overwrite for main output
            if dest_path.exists():
                if settings.overwrite_output:
                    dest_path.unlink()
                else:
                    counter = 2
                    while dest_path.exists():
                        dest_name = f"{dest_stem}_{counter:03d}.png"
                        dest_path = output_dir / dest_name
                        counter += 1

            # Write main output to temp, then move immediately
            frame_clip = clip[idx]
            out_path = output_seq_dir / f"out_{idx:06d}{out_fmt}"
            sink = frame_clip.fpng.Write(
                filename=str(out_path),
                overwrite=1,
            )
            for _ in sink.frames(close=True):
                pass

            # Move main file to final destination immediately
            if out_path.exists():
                shutil.move(str(out_path), str(dest_path))
                # Apply PNG optimization if enabled (batch files don't have alpha)
                if settings.png_quantize_enabled or settings.png_optimize_enabled:
                    optimize_png(
                        dest_path,
                        quantize_enabled=settings.png_quantize_enabled,
                        quantize_colors=settings.png_quantize_colors,
                        optimize_enabled=settings.png_optimize_enabled,
                    )

            # Write and move secondary output if enabled
            if clip_secondary is not None:
                secondary_dir.mkdir(parents=True, exist_ok=True)
                sec_dest_name = f"{dest_stem}.png"
                sec_dest_path = secondary_dir / sec_dest_name

                # Handle overwrite for secondary output
                if sec_dest_path.exists():
                    if settings.overwrite_output:
                        sec_dest_path.unlink()
                    else:
                        counter = 2
                        while sec_dest_path.exists():
                            sec_dest_name = f"{dest_stem}_{counter:03d}.png"
                            sec_dest_path = secondary_dir / sec_dest_name
                            counter += 1

                sec_frame = clip_secondary[idx]
                sec_path = secondary_seq_dir / f"sec_{idx:06d}{out_fmt}"
                sec_sink = sec_frame.fpng.Write(
                    filename=str(sec_path),
                    overwrite=1,
                )
                for _ in sec_sink.frames(close=True):
                    pass

                # Move secondary file to final destination immediately
                if sec_path.exists():
                    shutil.move(str(sec_path), str(sec_dest_path))

            # Update progress file for GUI tracking
            if progress_file is not None:
                try:
                    global_idx = batch_offset + idx + 1
                    with open(progress_file, "w", encoding="utf-8") as pf:
                        pf.write(f"{global_idx},{report_total},{frame_map[idx].name}")
                except Exception:
                    pass

            if (idx + 1) % 10 == 0 or idx == num_files - 1:
                print(f"  Processed {idx + 1}/{num_files}")

        total_time = time.perf_counter() - batch_start
        avg_time = total_time / num_files if num_files > 0 else 0
        print(f"Batch complete: {num_files} images in {total_time:.2f}s (avg {avg_time:.2f}s/image)")

        # Distribute total time evenly for reporting
        distributed_time = total_time / num_files
        return [distributed_time] * num_files

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(str(temp_dir), ignore_errors=True)
        except Exception:
            pass

        # Clear caches
        clear_cache()
