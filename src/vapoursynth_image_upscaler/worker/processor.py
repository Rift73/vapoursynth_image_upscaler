"""
File processing logic for worker mode.

Handles the complete workflow of processing a single file, including:
- Building SR clips
- Writing outputs to temporary locations (using imwri.Write)
- Moving to final destinations
- Handling filename conflicts
- Supporting multiple output formats (PNG, JPEG, GIF, etc.)
- Frame deduplication for GIF inputs with variable frame rate output
"""

from __future__ import annotations

import hashlib
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
    build_clip_with_frame_selection,
    build_alpha_hq,
    build_alpha_hq_with_frame_selection,
    build_alpha_from_arrays,
    apply_custom_resolution,
    apply_sharpening,
    compute_custom_resolution_dimensions,
    compute_secondary_dimensions,
    clear_cache,
    get_process_start_time,
)
from .progress import track

if TYPE_CHECKING:
    pass


def _extract_gif_alpha_masks(gif_path: Path, frame_indices: list[int] | None = None) -> list | None:
    """
    Extract alpha masks from a GIF file using Pillow.

    GIFs use palette-based transparency (a single color index is marked as transparent).
    This function extracts proper alpha masks for each frame.

    Args:
        gif_path: Path to the GIF file.
        frame_indices: Optional list of frame indices to extract. If None, extract all.

    Returns:
        List of alpha numpy arrays (H, W) as uint8, or None if no transparency.
    """
    from PIL import Image
    import numpy as np

    alpha_masks: list = []
    has_any_transparency = False

    try:
        with Image.open(str(gif_path)) as img:
            num_frames = getattr(img, 'n_frames', 1)
            indices_to_extract = frame_indices if frame_indices else list(range(num_frames))

            for frame_idx in indices_to_extract:
                if frame_idx >= num_frames:
                    continue

                img.seek(frame_idx)

                # Convert to RGBA to get proper alpha
                frame_rgba = img.convert('RGBA')
                frame_array = np.array(frame_rgba)

                # Extract alpha channel
                alpha = frame_array[:, :, 3]
                alpha_masks.append(alpha)

                # Check if this frame has any transparency
                if np.any(alpha < 255):
                    has_any_transparency = True

    except Exception as e:
        print(f"Warning: Could not extract GIF alpha masks: {e}")
        return None

    if not has_any_transparency:
        print("GIF has no transparency, skipping alpha processing")
        return None

    return alpha_masks


def _extract_gif_frame_delays(gif_path: Path) -> list[int]:
    """
    Extract per-frame delays from a GIF file.

    Args:
        gif_path: Path to the GIF file.

    Returns:
        List of delays in milliseconds for each frame.
    """
    from PIL import Image

    delays: list[int] = []
    try:
        with Image.open(str(gif_path)) as img:
            for frame_idx in range(getattr(img, 'n_frames', 1)):
                img.seek(frame_idx)
                # GIF delay is in centiseconds (1/100th of a second)
                # Convert to milliseconds
                delay = img.info.get('duration', 100)
                # Ensure minimum delay of 20ms for GIF compatibility
                delays.append(max(20, delay))
    except Exception as e:
        print(f"Warning: Could not extract GIF frame delays: {e}")

    return delays


def _detect_duplicate_frames(
    input_path: Path,
    num_frames: int,
    frame_delays: list[int],
) -> tuple[list[int], list[int]]:
    """
    Detect duplicate frames in a GIF by comparing frame content.

    Uses perceptual hashing to detect frames that are identical or very similar.
    When duplicates are found, their delays are accumulated into the preceding
    unique frame to maintain the same total animation duration.

    Args:
        input_path: Path to the input GIF file.
        num_frames: Total number of frames in the source clip.
        frame_delays: Original per-frame delays from the GIF.

    Returns:
        Tuple of (unique_frame_indices, accumulated_delays).
        - unique_frame_indices: List of frame indices to keep.
        - accumulated_delays: List of delays for each unique frame.
    """
    from PIL import Image
    import numpy as np

    unique_indices: list[int] = []
    accumulated_delays: list[int] = []
    prev_hash: str | None = None

    try:
        with Image.open(str(input_path)) as img:
            actual_frames = getattr(img, 'n_frames', 1)

            # Use the minimum of provided frame count and actual frames
            check_frames = min(num_frames, actual_frames)

            for frame_idx in range(check_frames):
                img.seek(frame_idx)
                # Convert to RGB for consistent comparison
                frame_rgb = img.convert('RGB')
                # Resize to small size for faster comparison (perceptual hash approach)
                frame_small = frame_rgb.resize((16, 16), Image.Resampling.LANCZOS)
                # Compute hash of the small frame
                frame_array = np.array(frame_small)
                frame_hash = hashlib.md5(frame_array.tobytes()).hexdigest()

                # Get delay for this frame
                delay = frame_delays[frame_idx] if frame_idx < len(frame_delays) else 100

                if prev_hash is None or frame_hash != prev_hash:
                    # New unique frame
                    unique_indices.append(frame_idx)
                    accumulated_delays.append(delay)
                    prev_hash = frame_hash
                else:
                    # Duplicate frame - accumulate delay to previous unique frame
                    if accumulated_delays:
                        accumulated_delays[-1] += delay

    except Exception as e:
        print(f"Warning: Frame deduplication failed: {e}")
        # Fall back to all frames with original delays
        return list(range(num_frames)), frame_delays[:num_frames] if frame_delays else [100] * num_frames

    if not unique_indices:
        # No frames found, return original
        return list(range(num_frames)), frame_delays[:num_frames] if frame_delays else [100] * num_frames

    return unique_indices, accumulated_delays


def _determine_output_format(settings: WorkerSettings) -> tuple[str, str]:
    """
    Determine the output format based on input file type.

    GIF and short videos output as GIF; all other formats output as PNG.

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

    # All other formats (JPG, TIFF, BMP, WebP, PNG, etc.) -> PNG output
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
    frame_delays: list[int] | int = 100,
) -> None:
    """
    Write a clip to disk.

    Uses fpng.Write for single-frame PNG (fastest), imwri.Write for other formats,
    and Pillow for animated GIF output.

    Args:
        clip: The clip to write (should be RGB format).
        output_path: Path to the output file.
        imgformat: Image format name for imwri (e.g., "PNG", "JPEG", "GIF").
        alpha_clip: Optional alpha channel clip (GRAY format).
        quality: Output quality (0-100).
        prefix: Progress prefix for tracking.
        frame_delays: Delay(s) for animated GIF - either a single int (uniform delay)
                      or a list of ints (per-frame delays for variable frame rate).
    """
    num_frames = len(clip)

    # For animated GIF (multi-frame), write frames then combine
    if imgformat == "GIF" and num_frames > 1:
        _write_animated_gif(clip, output_path, alpha_clip, prefix, frame_delays)
        return

    # For single-frame PNG, use fpng.Write (much faster than imwri)
    if imgformat == "PNG" and num_frames == 1:
        clip = core.resize.Point(clip, format=vs.RGB24)
        if alpha_clip is not None:
            alpha_clip = core.resize.Point(alpha_clip, format=vs.GRAY8)

        sink = clip.fpng.Write(
            filename=str(output_path),
            alpha=alpha_clip,
            overwrite=1,
        )
        for _ in track(sink.frames(close=True), total=len(sink), prefix=prefix):
            pass
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

    # Use imwri.Write for other formats
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
    frame_delays: list[int] | int = 100,
) -> None:
    """
    Write an animated GIF using Pillow with variable frame rate support.

    Args:
        clip: The clip to write (multi-frame).
        output_path: Path to the output GIF file.
        alpha_clip: Optional alpha channel clip (GRAY format).
        prefix: Progress prefix for tracking.
        frame_delays: Either a single int for uniform delay, or a list of ints
                      for per-frame delays (variable frame rate).
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

    # Normalize frame_delays to a list
    if isinstance(frame_delays, int):
        delays_list = [frame_delays] * num_frames
    else:
        delays_list = list(frame_delays)
        # Extend with last delay if list is shorter than frames
        while len(delays_list) < num_frames:
            delays_list.append(delays_list[-1] if delays_list else 100)

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

    print(f"{prefix}: Saving animated GIF with variable frame rate...")

    # Save animated GIF with per-frame delays
    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:] if len(frames) > 1 else [],
        duration=delays_list[:len(frames)],  # Per-frame delays
        loop=0,  # 0 = infinite loop
        optimize=True,
    )

    print(f"{prefix}: Animated GIF saved with {len(frames)} frames (variable frame rate)")


def _extract_color_frames_to_list(
    clip: vs.VideoNode,
    prefix: str = "Color frames",
) -> list:
    """
    Extract color frames from a clip to a list of numpy arrays.

    This allows clearing VRAM before processing alpha.

    Args:
        clip: The clip to extract (multi-frame).
        prefix: Progress prefix for tracking.

    Returns:
        List of RGB numpy arrays (H, W, 3) as uint8.
    """
    import numpy as np

    # Convert clip to RGB24
    clip = core.resize.Point(clip, format=vs.RGB24)

    rgb_frames: list = []
    num_frames = len(clip)

    print(f"{prefix}: Extracting {num_frames} frames to memory...")

    for i, frame in enumerate(clip.frames(close=True)):
        # Extract frame data
        r = np.asarray(frame[0]).copy()
        g = np.asarray(frame[1]).copy()
        b = np.asarray(frame[2]).copy()

        # Stack RGB
        rgb = np.stack([r, g, b], axis=-1)
        rgb_frames.append(rgb)

        if (i + 1) % 10 == 0 or i == num_frames - 1:
            print(f"{prefix}: {i + 1}/{num_frames} frames extracted")

    return rgb_frames


def _merge_and_write_animated_gif(
    rgb_frames: list,
    alpha_clip: vs.VideoNode | None,
    output_path: Path,
    frame_delays: list[int] | int = 100,
    prefix: str = "Merge",
) -> None:
    """
    Merge pre-extracted RGB frames with alpha clip and write animated GIF.

    Args:
        rgb_frames: List of RGB numpy arrays (H, W, 3) as uint8.
        alpha_clip: Optional VapourSynth clip (GRAY8 format) with SR-upscaled alpha.
        output_path: Path to the output GIF file.
        frame_delays: Either a single int for uniform delay, or a list of ints.
        prefix: Progress prefix for tracking.
    """
    from PIL import Image
    import numpy as np

    if not rgb_frames:
        print(f"Warning: No frames to write for animated GIF")
        return

    num_frames = len(rgb_frames)
    has_alpha = alpha_clip is not None and len(alpha_clip) == num_frames

    # Prepare alpha clip if provided
    if has_alpha:
        alpha_clip = core.resize.Point(alpha_clip, format=vs.GRAY8)

    # Normalize frame_delays to a list
    if isinstance(frame_delays, int):
        delays_list = [frame_delays] * num_frames
    else:
        delays_list = list(frame_delays)
        while len(delays_list) < num_frames:
            delays_list.append(delays_list[-1] if delays_list else 100)

    frames: list[Image.Image] = []

    print(f"{prefix}: Processing {num_frames} frames (alpha: {has_alpha})...")

    for i, rgb in enumerate(rgb_frames):
        if has_alpha:
            # Get corresponding alpha frame from VS clip
            alpha_frame = alpha_clip.get_frame(i)
            alpha_data = np.asarray(alpha_frame[0]).copy()

            # Resize alpha if dimensions don't match (shouldn't happen with SR, but safety check)
            if alpha_data.shape[0] != rgb.shape[0] or alpha_data.shape[1] != rgb.shape[1]:
                print(f"  Warning: Alpha size mismatch at frame {i}: alpha={alpha_data.shape}, rgb={rgb.shape[:2]}")
                alpha_img = Image.fromarray(alpha_data, mode="L")
                alpha_img = alpha_img.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.LANCZOS)
                alpha_data = np.array(alpha_img)

            rgba = np.concatenate([rgb, alpha_data[:, :, np.newaxis]], axis=-1)
            img = Image.fromarray(rgba, mode="RGBA")

            # Check if this frame has any transparency
            alpha_mask = alpha_data < 128  # Binary threshold for transparency

            if np.any(alpha_mask):
                # Use FASTOCTREE for RGBA quantization (only reliable option in standard Pillow)
                # Reserve 255 colors to leave room for transparent color at index 0
                img_p = img.quantize(colors=255, method=Image.Quantize.FASTOCTREE)

                # Get the palette
                palette = img_p.getpalette()

                # Get the quantized pixel data
                px_data = np.array(img_p)

                # Shift all indices by 1 to make room for transparent at index 0
                px_data = px_data + 1
                px_data = np.clip(px_data, 0, 255).astype(np.uint8)

                # Set transparent pixels to index 0
                px_data[alpha_mask] = 0

                new_img = Image.fromarray(px_data, mode="P")

                # Shift palette to make room for transparent color at index 0
                new_palette = [0, 0, 0] + palette[:765]  # RGB for index 0 (doesn't matter, it's transparent)
                new_img.putpalette(new_palette)

                frames.append(new_img)
            else:
                # No transparency in this frame, use standard RGB conversion
                img_rgb = Image.fromarray(rgb, mode="RGB")
                img_p = img_rgb.convert("P", palette=Image.ADAPTIVE, colors=256)
                frames.append(img_p)
        else:
            img = Image.fromarray(rgb, mode="RGB")
            # Convert to palette mode for GIF (better quality)
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
            frames.append(img)

        if (i + 1) % 10 == 0 or i == num_frames - 1:
            print(f"{prefix}: {i + 1}/{num_frames} frames processed")

    print(f"{prefix}: Saving animated GIF with variable frame rate...")

    # Save animated GIF with per-frame delays
    save_kwargs = {
        "save_all": True,
        "append_images": frames[1:] if len(frames) > 1 else [],
        "duration": delays_list[:len(frames)],
        "loop": 0,
        "optimize": False,  # Don't optimize when we have custom transparency
    }

    if has_alpha:
        save_kwargs["transparency"] = 0  # Index 0 is transparent
        save_kwargs["disposal"] = 2  # Clear frame before drawing next

    frames[0].save(str(output_path), **save_kwargs)

    print(f"{prefix}: Animated GIF saved with {len(frames)} frames")


def process_one(
    input_path: Path,
    output_dir: Path,
    secondary_output_dir: Path,
    settings: WorkerSettings,
) -> None:
    """
    Process a single file in main worker mode.

    For single-frame images, alpha is handled in a separate --alpha-worker process.
    For animated GIFs with alpha enabled, alpha is processed inline because the
    alpha-worker cannot read multi-frame GIFs with imwri.Read.

    For GIF inputs, performs frame deduplication to remove duplicate frames
    and outputs a variable frame rate GIF with accumulated delays.

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

    if settings.manga_folder_enabled:
        # Manga folder mode: suffix is in the parent folder name, not the file
        dest_stem = f"{base_name}{model_suffix}"
        dest_dir_main = output_dir
    elif settings.use_same_dir_output:
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

    # For GIF inputs, perform frame deduplication
    frame_indices: list[int] | None = None
    frame_delays: list[int] | int = _get_frame_delay_ms(settings)
    is_animated_gif = False

    if settings.input_extension.lower() == ".gif":
        # Extract original frame delays from the GIF
        original_delays = _extract_gif_frame_delays(input_path)

        if original_delays:
            # Get total frame count from the GIF
            from PIL import Image
            try:
                with Image.open(str(input_path)) as img:
                    total_frames = getattr(img, 'n_frames', 1)
            except Exception:
                total_frames = len(original_delays)

            is_animated_gif = total_frames > 1

            # Detect and remove duplicate frames
            frame_indices, frame_delays = _detect_duplicate_frames(
                input_path, total_frames, original_delays
            )

            if len(frame_indices) < total_frames:
                removed = total_frames - len(frame_indices)
                print(f"GIF deduplication: {removed} duplicate frames removed "
                      f"({total_frames} -> {len(frame_indices)} unique frames)")

    # Determine if we need to process alpha inline (for animated GIFs)
    # Alpha-worker cannot read multi-frame GIFs with imwri.Read, so we handle it here
    # We use a two-pass approach to avoid doubling VRAM usage:
    # Pass 1: Process color frames, extract to memory, clear VRAM
    # Pass 2: Process alpha frames, merge with color, write GIF
    process_alpha_inline = settings.use_alpha and is_animated_gif and imgformat == "GIF"

    # Store dimensions for alpha processing (needed after clearing cache)
    main_width = 0
    main_height = 0

    try:
        # Build SR clip (with or without frame selection)
        if frame_indices is not None and len(frame_indices) > 0:
            clip_sr = build_clip_with_frame_selection(input_path, settings, frame_indices)
        else:
            clip_sr = build_clip(input_path, settings)

        clip_main = clip_sr

        # Apply custom main resolution if enabled
        if settings.custom_res_enabled:
            custom_w, custom_h = compute_custom_resolution_dimensions(
                clip_main.width, clip_main.height,
                settings.custom_res_mode,
                settings.custom_width,
                settings.custom_height,
            )
            if custom_w > 0 and custom_h > 0:
                clip_main = apply_custom_resolution(
                    clip_main, custom_w, custom_h, settings.custom_res_kernel
                )

        # Apply sharpening at the end of main output processing
        clip_main = apply_sharpening(clip_main, settings)

        # Store dimensions for alpha matching
        main_width = clip_main.width
        main_height = clip_main.height

        # For animated GIF with alpha: two-pass approach to avoid double VRAM
        if process_alpha_inline:
            print("Pass 1: Processing color frames...")
            # Extract color frames to memory
            rgb_frames = _extract_color_frames_to_list(clip_main, prefix="Color frames")

            # Clear VRAM before processing alpha
            del clip_main
            del clip_sr
            clear_cache()

            # Extract alpha masks from original GIF using Pillow
            # (BestSource/VapourSynth don't reliably expose GIF alpha)
            print("Pass 2: Extracting alpha masks from original GIF...")
            alpha_masks = _extract_gif_alpha_masks(input_path, frame_indices)

            if alpha_masks is not None:
                # Build SR alpha clip from the extracted alpha masks
                print("Pass 2: Upscaling alpha with SR model...")
                try:
                    alpha_clip = build_alpha_from_arrays(
                        alpha_masks, settings,
                        target_width=main_width,
                        target_height=main_height,
                    )

                    # Merge and write with SR-upscaled alpha
                    _merge_and_write_animated_gif(
                        rgb_frames, alpha_clip, tmp_main, frame_delays, prefix="Merge + write"
                    )

                    # Clean up alpha clip VRAM immediately after use
                    del alpha_clip
                    clear_cache()
                except Exception as e:
                    print(f"Warning: Alpha SR processing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    _merge_and_write_animated_gif(
                        rgb_frames, None, tmp_main, frame_delays, prefix="Write (no alpha)"
                    )
            else:
                # No alpha in source, write without
                _merge_and_write_animated_gif(
                    rgb_frames, None, tmp_main, frame_delays, prefix="Write (no alpha)"
                )

            # Clean up rgb_frames to free memory
            del rgb_frames
        else:
            # Standard path: write directly
            _write_clip_imwri(clip_main, tmp_main, imgformat, prefix="Frames (main)", frame_delays=frame_delays)

        # Process secondary output if enabled
        # Note: Secondary output is only supported for single-frame images
        if settings.use_secondary_output and tmp_secondary is not None:
            if is_animated_gif:
                print("Note: Secondary output is not supported for animated GIFs, skipping")
            else:
                _process_secondary(tmp_main, tmp_secondary, settings, imgformat, file_ext)

        # Ensure destination directories exist
        if settings.manga_folder_enabled or not settings.use_same_dir_output:
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

    Note: Animated GIFs are handled inline by the main worker because
    imwri.Read cannot properly read multi-frame animated GIFs. This function
    only processes single-frame images.

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

    # Skip animated GIFs - they are handled inline by the main worker
    # because imwri.Read cannot properly read multi-frame animated GIFs
    if imgformat == "GIF" and settings.input_extension.lower() == ".gif":
        from PIL import Image
        try:
            with Image.open(str(input_path)) as img:
                if getattr(img, 'n_frames', 1) > 1:
                    print(f"[alpha-worker] Animated GIF alpha was processed inline, skipping")
                    return
        except Exception:
            pass

    # Resolve main destination path (same logic as main worker)
    model_suffix = settings.get_model_suffix()

    if settings.manga_folder_enabled:
        # Manga folder mode: suffix is in the parent folder name, not the file
        dest_stem = f"{base_name}{model_suffix}"
        dest_dir_main = output_dir
    elif settings.use_same_dir_output:
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
        if settings.secondary_kernel.lower() == "hermite":
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
