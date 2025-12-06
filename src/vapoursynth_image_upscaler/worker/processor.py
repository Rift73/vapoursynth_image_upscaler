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
import subprocess
import sys
import time
import uuid
from pathlib import Path

from vstools import vs, core, depth
from vskernels import Hermite, Lanczos

from ..core.constants import (
    WORKER_TMP_ROOT,
    SUPPORTED_VIDEO_EXTENSIONS,
    MAX_VIDEO_DURATION_FOR_GIF,
    CREATE_NO_WINDOW,
)
from ..core.utils import write_time_file
from .settings import WorkerSettings
from .pipeline import (
    build_clip,
    build_clip_with_frame_selection,
    build_alpha_hq,
    build_alpha_from_arrays,
    load_clip_no_upscale,
    load_clip_no_upscale_with_frame_selection,
    load_alpha_no_upscale,
    load_alpha_from_arrays_no_upscale,
    apply_custom_resolution,
    apply_sharpening,
    compute_custom_resolution_dimensions,
    compute_secondary_dimensions,
    clear_cache,
    get_process_start_time,
)
from .progress import track


def _run_subprocess_hidden(cmd: list, capture_output: bool = True) -> subprocess.CompletedProcess:
    """
    Run a subprocess with hidden console window on Windows.

    Args:
        cmd: Command list to execute.
        capture_output: Whether to capture stdout/stderr.

    Returns:
        CompletedProcess instance with return code and output.
    """
    kwargs = {
        "check": False,
        "text": True,
    }

    if capture_output:
        kwargs["capture_output"] = True

    # On Windows, hide the console window
    if sys.platform == "win32":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
        kwargs["creationflags"] = CREATE_NO_WINDOW
        kwargs["startupinfo"] = startupinfo

    return subprocess.run(cmd, **kwargs)


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
    Determine the output format based on input file type and animated output settings.

    For animated inputs (GIF and short videos), uses the user-selected format.
    For static images, outputs as PNG.

    Returns:
        Tuple of (imgformat for imwri, file_extension).
    """
    input_ext = settings.input_extension.lower()
    input_duration = settings.input_duration

    # Check if this is animated content
    is_animated = False
    if input_ext == ".gif":
        is_animated = True
    elif input_ext in SUPPORTED_VIDEO_EXTENSIONS:
        if 0 < input_duration <= MAX_VIDEO_DURATION_FOR_GIF:
            is_animated = True

    if is_animated:
        # Use user-selected animated output format
        fmt = settings.animated_output_format.upper()
        if fmt == "WEBP":
            return "WEBP", ".webp"
        elif fmt == "AVIF":
            return "AVIF", ".avif"
        elif fmt == "APNG":
            return "APNG", ".png"
        else:
            # Default to GIF
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
    settings: WorkerSettings | None = None,
) -> None:
    """
    Write a clip to disk.

    Uses fpng.Write for single-frame PNG (fastest), imwri.Write for other formats,
    Pillow for animated GIF output, and FFmpeg for animated WebP/AVIF.

    Args:
        clip: The clip to write (should be RGB format).
        output_path: Path to the output file.
        imgformat: Image format name for imwri (e.g., "PNG", "JPEG", "GIF", "WEBP", "AVIF").
        alpha_clip: Optional alpha channel clip (GRAY format).
        quality: Output quality (0-100).
        prefix: Progress prefix for tracking.
        frame_delays: Delay(s) for animated output - either a single int (uniform delay)
                      or a list of ints (per-frame delays for variable frame rate).
        settings: WorkerSettings with encoder options for WebP/AVIF.
    """
    num_frames = len(clip)

    # For animated GIF (multi-frame), write frames then combine
    if imgformat == "GIF" and num_frames > 1:
        _write_animated_gif(clip, output_path, alpha_clip, prefix, frame_delays, settings)
        return

    # For animated WebP, AVIF, or APNG (multi-frame), use FFmpeg/avifenc
    # Use rawvideo pipe for WebP/APNG (faster, no temp files), temp PNG for AVIF
    if imgformat in {"WEBP", "AVIF", "APNG"} and num_frames > 1:
        _write_animated_ffmpeg_pipe(clip, output_path, imgformat, alpha_clip, prefix, frame_delays, settings)
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
    settings: WorkerSettings | None = None,
) -> None:
    """
    Write an animated GIF using gifski for superior quality.

    gifski uses pngquant's lossy algorithm with temporal dithering and
    motion-aware quantization for the best possible GIF quality.

    Args:
        clip: The clip to write (multi-frame).
        output_path: Path to the output GIF file.
        alpha_clip: Optional alpha channel clip (GRAY format).
        prefix: Progress prefix for tracking.
        frame_delays: Either a single int for uniform delay, or a list of ints
                      for per-frame delays (variable frame rate).
        settings: WorkerSettings with GIF quality options.
    """
    from PIL import Image
    import numpy as np
    import tempfile

    # Convert clip to RGB24
    clip = core.resize.Point(clip, format=vs.RGB24)

    # Prepare alpha if provided
    if alpha_clip is not None:
        alpha_clip = core.resize.Point(alpha_clip, format=vs.GRAY8)

    num_frames = len(clip)

    # Normalize frame_delays to a list
    if isinstance(frame_delays, int):
        delays_list = [frame_delays] * num_frames
    else:
        delays_list = list(frame_delays)
        while len(delays_list) < num_frames:
            delays_list.append(delays_list[-1] if delays_list else 100)

    # Calculate average FPS from delays for gifski
    avg_delay_ms = sum(delays_list) / len(delays_list) if delays_list else 100
    fps = 1000.0 / avg_delay_ms if avg_delay_ms > 0 else 10.0

    # Get settings or use defaults
    gif_quality = settings.gif_quality if settings else 90
    gif_fast = settings.gif_fast if settings else False

    # Create temporary directory for PNG frames
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print(f"{prefix}: Extracting {num_frames} frames as PNG...")

        for i, frame in enumerate(clip.frames(close=True)):
            # Extract frame data
            frame_data = np.asarray(frame[0])
            r = frame_data.copy()
            g = np.asarray(frame[1]).copy()
            b = np.asarray(frame[2]).copy()

            # Stack RGB
            rgb = np.stack([r, g, b], axis=-1)

            # Create PIL image
            frame_path = tmp_path / f"frame_{i:06d}.png"
            if alpha_clip is not None:
                # Get corresponding alpha frame
                alpha_frame = alpha_clip.get_frame(i)
                alpha_data = np.asarray(alpha_frame[0]).copy()
                rgba = np.concatenate([rgb, alpha_data[:, :, np.newaxis]], axis=-1)
                img = Image.fromarray(rgba, mode="RGBA")
            else:
                img = Image.fromarray(rgb, mode="RGB")

            img.save(str(frame_path), "PNG")

            if (i + 1) % 10 == 0 or i == num_frames - 1:
                print(f"{prefix}: {i + 1}/{num_frames} frames extracted")

        # Build gifski command
        print(f"{prefix}: Encoding with gifski (quality={gif_quality}, fps={fps:.2f}, fast={gif_fast})...")

        cmd = [
            "gifski",
            "--quality", str(gif_quality),
            "--fps", str(fps),
            "-o", str(output_path),
        ]

        if gif_fast:
            cmd.append("--fast")

        # Add all frame files in order
        for i in range(num_frames):
            frame_path = tmp_path / f"frame_{i:06d}.png"
            cmd.append(str(frame_path))

        # Run gifski
        try:
            result = _run_subprocess_hidden(cmd)
            if result.returncode != 0:
                print(f"gifski encoding failed: {result.stderr}")
                print("Falling back to Pillow GIF encoding...")
                _write_animated_gif_pillow(
                    tmp_path, output_path, num_frames, delays_list, alpha_clip is not None, prefix
                )
            else:
                print(f"{prefix}: GIF saved with {num_frames} frames (gifski)")
        except FileNotFoundError:
            print("Error: gifski not found. Falling back to Pillow GIF encoding...")
            _write_animated_gif_pillow(
                tmp_path, output_path, num_frames, delays_list, alpha_clip is not None, prefix
            )
        except Exception as e:
            print(f"gifski error: {e}")
            print("Falling back to Pillow GIF encoding...")
            _write_animated_gif_pillow(
                tmp_path, output_path, num_frames, delays_list, alpha_clip is not None, prefix
            )


def _write_animated_gif_pillow(
    tmp_path: Path,
    output_path: Path,
    num_frames: int,
    delays_list: list[int],
    has_alpha: bool,
    prefix: str = "Frames",
) -> None:
    """
    Fallback GIF writer using Pillow when gifski is not available.

    Args:
        tmp_path: Path to temporary directory containing PNG frames.
        output_path: Path to the output GIF file.
        num_frames: Number of frames.
        delays_list: Per-frame delays in milliseconds.
        has_alpha: Whether frames have alpha channel.
        prefix: Progress prefix for tracking.
    """
    from PIL import Image

    frames: list[Image.Image] = []

    for i in range(num_frames):
        frame_path = tmp_path / f"frame_{i:06d}.png"
        if not frame_path.exists():
            continue

        img = Image.open(str(frame_path))

        if has_alpha:
            # Convert RGBA to palette with transparency
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
        else:
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)

        frames.append(img)

    if not frames:
        print(f"Warning: No frames to write for animated GIF")
        return

    print(f"{prefix}: Saving animated GIF with Pillow fallback...")

    frames[0].save(
        str(output_path),
        save_all=True,
        append_images=frames[1:] if len(frames) > 1 else [],
        duration=delays_list[:len(frames)],
        loop=0,
        optimize=True,
    )

    print(f"{prefix}: Animated GIF saved with {len(frames)} frames (Pillow fallback)")


def _write_animated_ffmpeg(
    clip: vs.VideoNode,
    output_path: Path,
    output_format: str,
    alpha_clip: vs.VideoNode | None = None,
    prefix: str = "Frames",
    frame_delays: list[int] | int = 100,
    settings: WorkerSettings | None = None,
) -> None:
    """
    Write animated WebP, AVIF, or APNG.

    For WebP: Uses FFmpeg with libwebp encoder.
    For AVIF: Uses avifenc from libavif (proper alpha support).
    For APNG: Uses FFmpeg with apng encoder.

    Args:
        clip: The clip to write (multi-frame).
        output_path: Path to the output file.
        output_format: "WEBP", "AVIF", or "APNG".
        alpha_clip: Optional alpha channel clip (GRAY format).
        prefix: Progress prefix for tracking.
        frame_delays: Either a single int for uniform delay, or a list of ints
                      for per-frame delays (variable frame rate).
        settings: WorkerSettings with encoder options for WebP/AVIF/APNG.
    """
    import subprocess
    import tempfile
    import numpy as np
    from PIL import Image

    # Convert clip to RGB24
    clip = core.resize.Point(clip, format=vs.RGB24)
    num_frames = len(clip)

    # Prepare alpha if provided
    has_alpha = alpha_clip is not None and len(alpha_clip) == num_frames
    if has_alpha:
        alpha_clip = core.resize.Point(alpha_clip, format=vs.GRAY8)

    # Normalize frame_delays to a list
    if isinstance(frame_delays, int):
        delays_list = [frame_delays] * num_frames
    else:
        delays_list = list(frame_delays)
        while len(delays_list) < num_frames:
            delays_list.append(delays_list[-1] if delays_list else 100)

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # AVIF: Use avifenc with Pillow for PNG frames (proper RGBA support)
        if output_format.upper() == "AVIF":
            print(f"{prefix}: Extracting {num_frames} frames for avifenc (has_alpha={has_alpha})...")

            # Extract frames with Pillow (ensures proper RGBA PNG)
            for i, frame in enumerate(clip.frames(close=True)):
                r = np.asarray(frame[0]).copy()
                g = np.asarray(frame[1]).copy()
                b = np.asarray(frame[2]).copy()
                rgb = np.stack([r, g, b], axis=-1)

                frame_path = tmp_path / f"frame_{i:06d}.png"

                if has_alpha:
                    alpha_frame = alpha_clip.get_frame(i)
                    alpha_data = np.asarray(alpha_frame[0]).copy()
                    rgba = np.concatenate([rgb, alpha_data[:, :, np.newaxis]], axis=-1)
                    img = Image.fromarray(rgba, mode="RGBA")
                    # Debug: verify first frame has alpha
                    if i == 0:
                        print(f"  First frame: mode={img.mode}, has_transparency={np.any(alpha_data < 255)}")
                else:
                    img = Image.fromarray(rgb, mode="RGB")

                img.save(str(frame_path), "PNG")

                if (i + 1) % 10 == 0 or i == num_frames - 1:
                    print(f"{prefix}: {i + 1}/{num_frames} frames extracted")

            # Build avifenc command
            # Use 1000 as timescale (1ms precision)
            timescale = 1000
            print(f"{prefix}: Encoding with avifenc...")

            # Get settings or use defaults
            avif_speed = settings.avif_speed if settings else 6
            avif_quality = settings.avif_quality if settings else 80
            avif_quality_alpha = settings.avif_quality_alpha if settings else 90
            avif_lossless = settings.avif_lossless if settings else False

            cmd = [
                "avifenc",
                "--timescale", str(timescale),
                "-s", str(avif_speed),
            ]

            if avif_lossless:
                cmd.append("-l")  # Lossless mode
            else:
                cmd.extend(["-q", str(avif_quality)])  # Color quality
                if has_alpha:
                    cmd.extend(["--qalpha", str(avif_quality_alpha)])  # Alpha quality

            # Add frame durations (in timescale units = milliseconds)
            # Each --duration applies to the following input file
            for i in range(num_frames):
                frame_path = tmp_path / f"frame_{i:06d}.png"
                duration_ms = delays_list[i]
                cmd.extend(["--duration", str(duration_ms), str(frame_path)])

            # Add output file
            cmd.extend(["-o", str(output_path)])

            # Debug: print avifenc command (first few args)
            cmd_preview = cmd[:15]  # First 15 args include flags
            print(f"  avifenc cmd: {' '.join(cmd_preview)}...")

            # Run avifenc
            try:
                result = _run_subprocess_hidden(cmd)
                if result.returncode != 0:
                    print(f"avifenc encoding failed: {result.stderr}")
                    print(f"Command was: {' '.join(cmd[:20])}...")
                else:
                    print(f"{prefix}: AVIF saved with {num_frames} frames (avifenc)")
            except FileNotFoundError:
                print("Error: avifenc not found. Please ensure libavif tools are installed and in PATH.")
            except Exception as e:
                print(f"avifenc error: {e}")
            return  # Exit early for AVIF

        # WebP/APNG: Use FFmpeg with concat demuxer
        concat_file = tmp_path / "concat.txt"

        print(f"{prefix}: Extracting {num_frames} frames for FFmpeg...")

        # Write frames and build concat file
        concat_entries = []

        for i, frame in enumerate(clip.frames(close=True)):
            r = np.asarray(frame[0]).copy()
            g = np.asarray(frame[1]).copy()
            b = np.asarray(frame[2]).copy()
            rgb = np.stack([r, g, b], axis=-1)

            frame_path = tmp_path / f"frame_{i:06d}.png"

            if has_alpha:
                alpha_frame = alpha_clip.get_frame(i)
                alpha_data = np.asarray(alpha_frame[0]).copy()
                rgba = np.concatenate([rgb, alpha_data[:, :, np.newaxis]], axis=-1)
                img = Image.fromarray(rgba, mode="RGBA")
            else:
                img = Image.fromarray(rgb, mode="RGB")

            img.save(str(frame_path), "PNG")

            delay_sec = delays_list[i] / 1000.0
            concat_entries.append(f"file '{frame_path.name}'\nduration {delay_sec:.6f}")

            if (i + 1) % 10 == 0 or i == num_frames - 1:
                print(f"{prefix}: {i + 1}/{num_frames} frames extracted")

        # Add last frame again (required by FFmpeg concat demuxer)
        if concat_entries:
            last_frame = f"frame_{num_frames - 1:06d}.png"
            concat_entries.append(f"file '{last_frame}'")

        # Write concat file
        with open(concat_file, "w", encoding="utf-8") as f:
            f.write("\n".join(concat_entries))

        # Build FFmpeg command based on output format
        print(f"{prefix}: Encoding with FFmpeg ({output_format})...")

        if output_format.upper() == "APNG":
            # APNG encoding with FFmpeg apng encoder
            apng_pred = settings.apng_pred if settings else "mixed"

            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "apng",
                "-pred", apng_pred,
                "-plays", "0",  # Infinite loop
                "-f", "apng",
                str(output_path),
            ]
        else:
            # WebP encoding with FFmpeg libwebp
            webp_quality = settings.webp_quality if settings else 90
            webp_lossless = settings.webp_lossless if settings else True
            webp_preset = settings.webp_preset if settings else "none"

            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c:v", "libwebp",
                "-lossless", "1" if webp_lossless else "0",
                "-quality", str(webp_quality),
                "-loop", "0",
            ]

            if webp_preset and webp_preset != "none":
                cmd.extend(["-preset", webp_preset])

            cmd.extend(["-f", "webp", str(output_path)])

        # Run FFmpeg
        try:
            result = _run_subprocess_hidden(cmd)
            if result.returncode != 0:
                print(f"FFmpeg encoding failed: {result.stderr}")
                if output_format.upper() == "WEBP" and "libwebp" in result.stderr:
                    print("Note: FFmpeg may need to be compiled with libwebp support for WebP output")
            else:
                print(f"{prefix}: {output_format} saved with {num_frames} frames")
        except FileNotFoundError:
            print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in PATH.")
        except Exception as e:
            print(f"FFmpeg error: {e}")


def _write_animated_ffmpeg_pipe(
    clip: vs.VideoNode,
    output_path: Path,
    output_format: str,
    alpha_clip: vs.VideoNode | None = None,
    prefix: str = "Frames",
    frame_delays: list[int] | int = 100,
    settings: WorkerSettings | None = None,
) -> None:
    """
    Write animated WebP or APNG using rawvideo pipe (no temp PNG files).

    This is an optimized version that pipes RGBA/RGB data directly to FFmpeg,
    eliminating the disk I/O overhead of temporary PNG files.

    For WebP: Uses FFmpeg with libwebp encoder.
    For APNG: Uses FFmpeg with apng encoder.

    Note: AVIF still uses temp PNG files because avifenc's --stdin only accepts Y4M
    (which doesn't support alpha channel).

    Args:
        clip: The clip to write (multi-frame).
        output_path: Path to the output file.
        output_format: "WEBP" or "APNG" (AVIF not supported via pipe).
        alpha_clip: Optional alpha channel clip (GRAY format).
        prefix: Progress prefix for tracking.
        frame_delays: Either a single int for uniform delay, or a list of ints
                      for per-frame delays (variable frame rate).
        settings: WorkerSettings with encoder options for WebP/APNG.
    """
    import subprocess
    import numpy as np

    # AVIF doesn't support rawvideo pipe with alpha, fall back to temp PNG method
    if output_format.upper() == "AVIF":
        _write_animated_ffmpeg(clip, output_path, output_format, alpha_clip, prefix, frame_delays, settings)
        return

    # Convert clip to RGB24
    clip = core.resize.Point(clip, format=vs.RGB24)
    num_frames = len(clip)
    width = clip.width
    height = clip.height

    # Prepare alpha if provided
    has_alpha = alpha_clip is not None and len(alpha_clip) == num_frames
    if has_alpha:
        alpha_clip = core.resize.Point(alpha_clip, format=vs.GRAY8)

    # Normalize frame_delays to a list
    if isinstance(frame_delays, int):
        delays_list = [frame_delays] * num_frames
    else:
        delays_list = list(frame_delays)
        while len(delays_list) < num_frames:
            delays_list.append(delays_list[-1] if delays_list else 100)

    # Calculate average FPS from delays
    avg_delay_ms = sum(delays_list) / len(delays_list) if delays_list else 100
    fps = 1000.0 / avg_delay_ms if avg_delay_ms > 0 else 10.0

    # Determine pixel format
    pix_fmt = "rgba" if has_alpha else "rgb24"

    # Build FFmpeg command for rawvideo input
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",  # Read from stdin
    ]

    if output_format.upper() == "WEBP":
        # Get settings or use defaults
        webp_quality = settings.webp_quality if settings else 90
        webp_lossless = settings.webp_lossless if settings else True
        webp_preset = settings.webp_preset if settings else "none"

        cmd.extend([
            "-c:v", "libwebp",
            "-lossless", "1" if webp_lossless else "0",
            "-quality", str(webp_quality),
            "-loop", "0",
        ])
        if webp_preset and webp_preset != "none":
            cmd.extend(["-preset", webp_preset])
        cmd.extend(["-f", "webp"])

    elif output_format.upper() == "APNG":
        # APNG encoding with FFmpeg apng encoder
        apng_pred = settings.apng_pred if settings else "mixed"

        cmd.extend([
            "-c:v", "apng",
            "-pred", apng_pred,
            "-plays", "0",  # Infinite loop
            "-f", "apng",
        ])

    cmd.append(str(output_path))

    print(f"{prefix}: Encoding {num_frames} frames via pipe (pix_fmt={pix_fmt}, fps={fps:.2f})...")

    # Start FFmpeg process
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        # Write frames to pipe
        for i, frame in enumerate(clip.frames(close=True)):
            # Extract RGB planes
            r = np.asarray(frame[0]).copy()
            g = np.asarray(frame[1]).copy()
            b = np.asarray(frame[2]).copy()

            if has_alpha:
                # Get alpha and pack as RGBA
                alpha_frame = alpha_clip.get_frame(i)
                a = np.asarray(alpha_frame[0]).copy()
                # Pack as RGBA (interleaved)
                rgba = np.stack([r, g, b, a], axis=-1)
                process.stdin.write(rgba.tobytes())
            else:
                # Pack as RGB24 (interleaved)
                rgb = np.stack([r, g, b], axis=-1)
                process.stdin.write(rgb.tobytes())

            if (i + 1) % 10 == 0 or i == num_frames - 1:
                print(f"{prefix}: {i + 1}/{num_frames} frames piped")

        # Close stdin and wait for FFmpeg to finish
        process.stdin.close()
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"FFmpeg encoding failed: {stderr.decode('utf-8', errors='replace')}")
            # Fall back to temp PNG method
            print("Falling back to temp PNG method...")
            _write_animated_ffmpeg(clip, output_path, output_format, alpha_clip, prefix, frame_delays, settings)
        else:
            print(f"{prefix}: {output_format} saved with {num_frames} frames (rawvideo pipe)")

    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in PATH.")
    except Exception as e:
        print(f"FFmpeg pipe error: {e}")
        # Fall back to temp PNG method
        print("Falling back to temp PNG method...")
        _write_animated_ffmpeg(clip, output_path, output_format, alpha_clip, prefix, frame_delays, settings)


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
    settings: WorkerSettings | None = None,
) -> None:
    """
    Merge pre-extracted RGB frames with alpha clip and write animated GIF using gifski.

    Args:
        rgb_frames: List of RGB numpy arrays (H, W, 3) as uint8.
        alpha_clip: Optional VapourSynth clip (GRAY8 format) with SR-upscaled alpha.
        output_path: Path to the output GIF file.
        frame_delays: Either a single int for uniform delay, or a list of ints.
        prefix: Progress prefix for tracking.
        settings: WorkerSettings with GIF quality options.
    """
    from PIL import Image
    import numpy as np
    import tempfile

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

    # Calculate average FPS from delays for gifski
    avg_delay_ms = sum(delays_list) / len(delays_list) if delays_list else 100
    fps = 1000.0 / avg_delay_ms if avg_delay_ms > 0 else 10.0

    # Get settings or use defaults
    gif_quality = settings.gif_quality if settings else 90
    gif_fast = settings.gif_fast if settings else False

    # Create temporary directory for PNG frames
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        print(f"{prefix}: Writing {num_frames} frames as PNG (alpha: {has_alpha})...")

        for i, rgb in enumerate(rgb_frames):
            frame_path = tmp_path / f"frame_{i:06d}.png"

            if has_alpha:
                # Get corresponding alpha frame from VS clip
                alpha_frame = alpha_clip.get_frame(i)
                alpha_data = np.asarray(alpha_frame[0]).copy()

                # Resize alpha if dimensions don't match
                if alpha_data.shape[0] != rgb.shape[0] or alpha_data.shape[1] != rgb.shape[1]:
                    print(f"  Warning: Alpha size mismatch at frame {i}: alpha={alpha_data.shape}, rgb={rgb.shape[:2]}")
                    alpha_img = Image.fromarray(alpha_data, mode="L")
                    alpha_img = alpha_img.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.LANCZOS)
                    alpha_data = np.array(alpha_img)

                rgba = np.concatenate([rgb, alpha_data[:, :, np.newaxis]], axis=-1)
                img = Image.fromarray(rgba, mode="RGBA")
            else:
                img = Image.fromarray(rgb, mode="RGB")

            img.save(str(frame_path), "PNG")

            if (i + 1) % 10 == 0 or i == num_frames - 1:
                print(f"{prefix}: {i + 1}/{num_frames} frames written")

        # Build gifski command
        print(f"{prefix}: Encoding with gifski (quality={gif_quality}, fps={fps:.2f}, fast={gif_fast})...")

        cmd = [
            "gifski",
            "--quality", str(gif_quality),
            "--fps", str(fps),
            "-o", str(output_path),
        ]

        if gif_fast:
            cmd.append("--fast")

        # Add all frame files in order
        for i in range(num_frames):
            frame_path = tmp_path / f"frame_{i:06d}.png"
            cmd.append(str(frame_path))

        # Run gifski
        try:
            result = _run_subprocess_hidden(cmd)
            if result.returncode != 0:
                print(f"gifski encoding failed: {result.stderr}")
                print("Falling back to Pillow GIF encoding...")
                _merge_and_write_animated_gif_pillow(
                    tmp_path, output_path, num_frames, delays_list, has_alpha, prefix
                )
            else:
                print(f"{prefix}: GIF saved with {num_frames} frames (gifski)")
        except FileNotFoundError:
            print("Error: gifski not found. Falling back to Pillow GIF encoding...")
            _merge_and_write_animated_gif_pillow(
                tmp_path, output_path, num_frames, delays_list, has_alpha, prefix
            )
        except Exception as e:
            print(f"gifski error: {e}")
            print("Falling back to Pillow GIF encoding...")
            _merge_and_write_animated_gif_pillow(
                tmp_path, output_path, num_frames, delays_list, has_alpha, prefix
            )


def _merge_and_write_animated_gif_pillow(
    tmp_path: Path,
    output_path: Path,
    num_frames: int,
    delays_list: list[int],
    has_alpha: bool,
    prefix: str = "Merge",
) -> None:
    """
    Fallback GIF writer using Pillow when gifski is not available.

    Args:
        tmp_path: Path to temporary directory containing PNG frames.
        output_path: Path to the output GIF file.
        num_frames: Number of frames.
        delays_list: Per-frame delays in milliseconds.
        has_alpha: Whether frames have alpha channel.
        prefix: Progress prefix for tracking.
    """
    from PIL import Image
    import numpy as np

    frames: list[Image.Image] = []

    for i in range(num_frames):
        frame_path = tmp_path / f"frame_{i:06d}.png"
        if not frame_path.exists():
            continue

        img = Image.open(str(frame_path))

        if has_alpha and img.mode == "RGBA":
            # Handle RGBA -> palette with transparency
            alpha_data = np.array(img)[:, :, 3]
            alpha_mask = alpha_data < 128

            if np.any(alpha_mask):
                # Reserve 255 colors to leave room for transparent color at index 0
                img_p = img.quantize(colors=255, method=Image.Quantize.FASTOCTREE)
                palette = img_p.getpalette()
                px_data = np.array(img_p)

                # Shift all indices by 1 to make room for transparent at index 0
                px_data = px_data + 1
                px_data = np.clip(px_data, 0, 255).astype(np.uint8)
                px_data[alpha_mask] = 0

                new_img = Image.fromarray(px_data, mode="P")
                new_palette = [0, 0, 0] + palette[:765]
                new_img.putpalette(new_palette)
                frames.append(new_img)
            else:
                img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
                frames.append(img)
        else:
            img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
            frames.append(img)

    if not frames:
        print(f"Warning: No frames to write for animated GIF")
        return

    print(f"{prefix}: Saving animated GIF with Pillow fallback...")

    save_kwargs = {
        "save_all": True,
        "append_images": frames[1:] if len(frames) > 1 else [],
        "duration": delays_list[:len(frames)],
        "loop": 0,
        "optimize": False,
    }

    if has_alpha:
        save_kwargs["transparency"] = 0
        save_kwargs["disposal"] = 2

    frames[0].save(str(output_path), **save_kwargs)

    print(f"{prefix}: Animated GIF saved with {len(frames)} frames (Pillow fallback)")


def _merge_and_write_animated_ffmpeg(
    rgb_frames: list,
    alpha_clip: vs.VideoNode | None,
    output_path: Path,
    output_format: str,
    frame_delays: list[int] | int = 100,
    settings: WorkerSettings | None = None,
    prefix: str = "Merge",
) -> None:
    """
    Merge pre-extracted RGB frames with alpha clip and write animated WebP/AVIF/APNG using FFmpeg.

    Args:
        rgb_frames: List of RGB numpy arrays (H, W, 3) as uint8.
        alpha_clip: Optional VapourSynth clip (GRAY8 format) with SR-upscaled alpha.
        output_path: Path to the output file.
        output_format: "WEBP", "AVIF", or "APNG".
        frame_delays: Either a single int for uniform delay, or a list of ints.
        settings: WorkerSettings with encoder options for WebP/AVIF/APNG.
        prefix: Progress prefix for tracking.
    """
    import subprocess
    import tempfile
    import numpy as np
    from PIL import Image

    if not rgb_frames:
        print(f"Warning: No frames to write")
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

    # Create temporary directory for frames
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        concat_file = tmp_path / "concat.txt"

        print(f"{prefix}: Preparing {num_frames} frames (alpha: {has_alpha}, format: {output_format})...")

        concat_entries = []

        for i, rgb in enumerate(rgb_frames):
            frame_path = tmp_path / f"frame_{i:06d}.png"

            if has_alpha:
                # Get corresponding alpha frame from VS clip
                alpha_frame = alpha_clip.get_frame(i)
                alpha_data = np.asarray(alpha_frame[0]).copy()

                # Resize alpha if dimensions don't match
                if alpha_data.shape[0] != rgb.shape[0] or alpha_data.shape[1] != rgb.shape[1]:
                    print(f"  Warning: Alpha size mismatch at frame {i}")
                    alpha_img = Image.fromarray(alpha_data, mode="L")
                    alpha_img = alpha_img.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.LANCZOS)
                    alpha_data = np.array(alpha_img)

                rgba = np.concatenate([rgb, alpha_data[:, :, np.newaxis]], axis=-1)
                img = Image.fromarray(rgba, mode="RGBA")
                # Debug: verify first frame has alpha
                if i == 0:
                    print(f"  First frame: mode={img.mode}, has_transparency={np.any(alpha_data < 255)}")
            else:
                img = Image.fromarray(rgb, mode="RGB")

            img.save(str(frame_path), "PNG")

            # Calculate duration in seconds for this frame
            delay_sec = delays_list[i] / 1000.0
            concat_entries.append(f"file '{frame_path.name}'\nduration {delay_sec:.6f}")

            if (i + 1) % 10 == 0 or i == num_frames - 1:
                print(f"{prefix}: {i + 1}/{num_frames} frames prepared")

        # Add last frame again (required by FFmpeg concat demuxer)
        if concat_entries:
            last_frame = f"frame_{num_frames - 1:06d}.png"
            concat_entries.append(f"file '{last_frame}'")

        # Write concat file
        with open(concat_file, "w", encoding="utf-8") as f:
            f.write("\n".join(concat_entries))

        # Build FFmpeg command
        print(f"{prefix}: Encoding with FFmpeg ({output_format})...")

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
        ]

        if output_format.upper() == "WEBP":
            # Get settings or use defaults
            webp_quality = settings.webp_quality if settings else 90
            webp_lossless = settings.webp_lossless if settings else True
            webp_preset = settings.webp_preset if settings else "none"

            cmd.extend([
                "-c:v", "libwebp",
                "-lossless", "1" if webp_lossless else "0",
                "-quality", str(webp_quality),
                "-loop", "0",
            ])
            if webp_preset and webp_preset != "none":
                cmd.extend(["-preset", webp_preset])
            cmd.extend(["-f", "webp"])

        elif output_format.upper() == "APNG":
            # APNG encoding with FFmpeg apng encoder
            apng_pred = settings.apng_pred if settings else "mixed"

            cmd.extend([
                "-c:v", "apng",
                "-pred", apng_pred,
                "-plays", "0",  # Infinite loop
                "-f", "apng",
            ])

        elif output_format.upper() == "AVIF":
            # Use avifenc for AVIF output (proper alpha support)
            # Use 1000 as timescale (1ms precision)
            timescale = 1000
            print(f"{prefix}: Encoding with avifenc...")

            # Get settings or use defaults
            avif_speed = settings.avif_speed if settings else 6
            avif_quality = settings.avif_quality if settings else 80
            avif_quality_alpha = settings.avif_quality_alpha if settings else 90
            avif_lossless = settings.avif_lossless if settings else False

            cmd = [
                "avifenc",
                "--timescale", str(timescale),
                "-s", str(avif_speed),
            ]

            if avif_lossless:
                cmd.append("-l")  # Lossless mode
            else:
                cmd.extend(["-q", str(avif_quality)])  # Color quality
                if has_alpha:
                    cmd.extend(["--qalpha", str(avif_quality_alpha)])  # Alpha quality

            # Add frame durations (in timescale units = milliseconds)
            # Each --duration applies to the following input file
            for i in range(num_frames):
                frame_path = tmp_path / f"frame_{i:06d}.png"
                duration_ms = delays_list[i]
                cmd.extend(["--duration", str(duration_ms), str(frame_path)])

            # Add output file
            cmd.extend(["-o", str(output_path)])

            # Debug: print avifenc command (first few args)
            cmd_preview = cmd[:15]  # First 15 args include flags
            print(f"  avifenc cmd: {' '.join(cmd_preview)}...")

            # Run avifenc
            try:
                result = _run_subprocess_hidden(cmd)
                if result.returncode != 0:
                    print(f"avifenc encoding failed: {result.stderr}")
                    print(f"Command was: {' '.join(cmd[:20])}...")
                else:
                    print(f"{prefix}: AVIF saved with {num_frames} frames (avifenc)")
            except FileNotFoundError:
                print("Error: avifenc not found. Please ensure libavif tools are installed and in PATH.")
            except Exception as e:
                print(f"avifenc error: {e}")
            return  # Exit early for AVIF

        cmd.append(str(output_path))

        try:
            result = _run_subprocess_hidden(cmd)
            if result.returncode != 0:
                print(f"FFmpeg encoding failed: {result.stderr}")
            else:
                print(f"{prefix}: {output_format} saved with {num_frames} frames")
        except FileNotFoundError:
            print(f"Error: FFmpeg not found. Please ensure FFmpeg is installed and in PATH.")
        except Exception as e:
            print(f"FFmpeg error: {e}")


def _merge_and_write_animated_ffmpeg_pipe(
    rgb_frames: list,
    alpha_clip: vs.VideoNode | None,
    output_path: Path,
    output_format: str,
    frame_delays: list[int] | int = 100,
    settings: WorkerSettings | None = None,
    prefix: str = "Merge",
) -> None:
    """
    Merge pre-extracted RGB frames with alpha clip and write animated WebP/APNG using rawvideo pipe.

    This is an optimized version that pipes RGBA/RGB data directly to FFmpeg,
    eliminating the disk I/O overhead of temporary PNG files.

    Note: AVIF still uses temp PNG files because avifenc's --stdin only accepts Y4M
    (which doesn't support alpha channel).

    Args:
        rgb_frames: List of RGB numpy arrays (H, W, 3) as uint8.
        alpha_clip: Optional VapourSynth clip (GRAY8 format) with SR-upscaled alpha.
        output_path: Path to the output file.
        output_format: "WEBP" or "APNG" (AVIF falls back to temp PNG method).
        frame_delays: Either a single int for uniform delay, or a list of ints.
        settings: WorkerSettings with encoder options for WebP/APNG.
        prefix: Progress prefix for tracking.
    """
    import subprocess
    import numpy as np
    from PIL import Image

    if not rgb_frames:
        print(f"Warning: No frames to write")
        return

    # AVIF doesn't support rawvideo pipe with alpha, fall back to temp PNG method
    if output_format.upper() == "AVIF":
        _merge_and_write_animated_ffmpeg(
            rgb_frames, alpha_clip, output_path, output_format, frame_delays, settings, prefix
        )
        return

    num_frames = len(rgb_frames)
    has_alpha = alpha_clip is not None and len(alpha_clip) == num_frames

    # Get dimensions from first frame
    height, width = rgb_frames[0].shape[:2]

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

    # Calculate average FPS from delays
    avg_delay_ms = sum(delays_list) / len(delays_list) if delays_list else 100
    fps = 1000.0 / avg_delay_ms if avg_delay_ms > 0 else 10.0

    # Determine pixel format
    pix_fmt = "rgba" if has_alpha else "rgb24"

    # Build FFmpeg command for rawvideo input
    cmd = [
        "ffmpeg",
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", pix_fmt,
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",  # Read from stdin
    ]

    if output_format.upper() == "WEBP":
        # Get settings or use defaults
        webp_quality = settings.webp_quality if settings else 90
        webp_lossless = settings.webp_lossless if settings else True
        webp_preset = settings.webp_preset if settings else "none"

        cmd.extend([
            "-c:v", "libwebp",
            "-lossless", "1" if webp_lossless else "0",
            "-quality", str(webp_quality),
            "-loop", "0",
        ])
        if webp_preset and webp_preset != "none":
            cmd.extend(["-preset", webp_preset])
        cmd.extend(["-f", "webp"])

    elif output_format.upper() == "APNG":
        # APNG encoding with FFmpeg apng encoder
        apng_pred = settings.apng_pred if settings else "mixed"

        cmd.extend([
            "-c:v", "apng",
            "-pred", apng_pred,
            "-plays", "0",  # Infinite loop
            "-f", "apng",
        ])

    cmd.append(str(output_path))

    print(f"{prefix}: Encoding {num_frames} frames via pipe (pix_fmt={pix_fmt}, fps={fps:.2f})...")

    # Start FFmpeg process
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )

        # Write frames to pipe
        for i, rgb in enumerate(rgb_frames):
            if has_alpha:
                # Get alpha and pack as RGBA
                alpha_frame = alpha_clip.get_frame(i)
                alpha_data = np.asarray(alpha_frame[0]).copy()

                # Resize alpha if dimensions don't match
                if alpha_data.shape[0] != rgb.shape[0] or alpha_data.shape[1] != rgb.shape[1]:
                    if i == 0:
                        print(f"  Warning: Alpha size mismatch, resizing...")
                    alpha_img = Image.fromarray(alpha_data, mode="L")
                    alpha_img = alpha_img.resize((rgb.shape[1], rgb.shape[0]), Image.Resampling.LANCZOS)
                    alpha_data = np.array(alpha_img)

                # Pack as RGBA (interleaved)
                rgba = np.concatenate([rgb, alpha_data[:, :, np.newaxis]], axis=-1)
                process.stdin.write(rgba.tobytes())
            else:
                # RGB is already packed from rgb_frames
                process.stdin.write(rgb.tobytes())

            if (i + 1) % 10 == 0 or i == num_frames - 1:
                print(f"{prefix}: {i + 1}/{num_frames} frames piped")

        # Close stdin and wait for FFmpeg to finish
        process.stdin.close()
        stdout, stderr = process.communicate()

        if process.returncode != 0:
            print(f"FFmpeg encoding failed: {stderr.decode('utf-8', errors='replace')}")
            # Fall back to temp PNG method
            print("Falling back to temp PNG method...")
            _merge_and_write_animated_ffmpeg(
                rgb_frames, alpha_clip, output_path, output_format, frame_delays, settings, prefix
            )
        else:
            print(f"{prefix}: {output_format} saved with {num_frames} frames (rawvideo pipe)")

    except FileNotFoundError:
        print("Error: FFmpeg not found. Please ensure FFmpeg is installed and in PATH.")
    except Exception as e:
        print(f"FFmpeg pipe error: {e}")
        # Fall back to temp PNG method
        print("Falling back to temp PNG method...")
        _merge_and_write_animated_ffmpeg(
            rgb_frames, alpha_clip, output_path, output_format, frame_delays, settings, prefix
        )


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
    dest_stem, dest_dir_main = settings.compute_dest_stem_and_dir(input_path, output_dir)
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

    # Determine if we need to process alpha inline (for animated content)
    # Alpha-worker cannot read multi-frame GIFs with imwri.Read, so we handle it here
    # We use a two-pass approach to avoid doubling VRAM usage:
    # Pass 1: Process color frames, extract to memory, clear VRAM
    # Pass 2: Process alpha frames, merge with color, write output
    is_animated_output = is_animated_gif and imgformat in {"GIF", "WEBP", "AVIF"}
    process_alpha_inline = settings.use_alpha and is_animated_output

    # Debug: show alpha processing path
    print(f"Alpha debug: use_alpha={settings.use_alpha}, is_animated_gif={is_animated_gif}, "
          f"imgformat={imgformat}, is_animated_output={is_animated_output}, "
          f"process_alpha_inline={process_alpha_inline}")

    # Store dimensions for alpha processing (needed after clearing cache)
    main_width = 0
    main_height = 0

    try:
        # Build clip - either with SR upscaling or just load without upscaling
        if settings.upscale_enabled:
            # Full SR upscaling pipeline
            if frame_indices is not None and len(frame_indices) > 0:
                clip_sr = build_clip_with_frame_selection(input_path, settings, frame_indices)
            else:
                clip_sr = build_clip(input_path, settings)
        else:
            # No upscaling - just load the clip
            if frame_indices is not None and len(frame_indices) > 0:
                clip_sr = load_clip_no_upscale_with_frame_selection(input_path, settings, frame_indices)
            else:
                clip_sr = load_clip_no_upscale(input_path, settings)

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
                # Build alpha clip - with or without SR upscaling
                if settings.upscale_enabled:
                    print("Pass 2: Upscaling alpha with SR model...")
                else:
                    print("Pass 2: Loading alpha (no upscaling)...")
                try:
                    if settings.upscale_enabled:
                        alpha_clip = build_alpha_from_arrays(
                            alpha_masks, settings,
                            target_width=main_width,
                            target_height=main_height,
                        )
                    else:
                        alpha_clip = load_alpha_from_arrays_no_upscale(
                            alpha_masks, settings,
                            target_width=main_width,
                            target_height=main_height,
                        )

                    # Merge and write with SR-upscaled alpha using format-appropriate function
                    if imgformat == "GIF":
                        _merge_and_write_animated_gif(
                            rgb_frames, alpha_clip, tmp_main, frame_delays,
                            prefix="Merge + write", settings=settings
                        )
                    else:
                        # WebP, AVIF, or APNG - use pipe version for WebP/APNG
                        _merge_and_write_animated_ffmpeg_pipe(
                            rgb_frames, alpha_clip, tmp_main, imgformat, frame_delays,
                            settings=settings, prefix="Merge + write"
                        )

                    # Clean up alpha clip VRAM immediately after use
                    del alpha_clip
                    clear_cache()
                except Exception as e:
                    print(f"Warning: Alpha SR processing failed: {e}")
                    import traceback
                    traceback.print_exc()
                    if imgformat == "GIF":
                        _merge_and_write_animated_gif(
                            rgb_frames, None, tmp_main, frame_delays,
                            prefix="Write (no alpha)", settings=settings
                        )
                    else:
                        _merge_and_write_animated_ffmpeg_pipe(
                            rgb_frames, None, tmp_main, imgformat, frame_delays,
                            settings=settings, prefix="Write (no alpha)"
                        )
            else:
                # No alpha in source, write without
                if imgformat == "GIF":
                    _merge_and_write_animated_gif(
                        rgb_frames, None, tmp_main, frame_delays,
                        prefix="Write (no alpha)", settings=settings
                    )
                else:
                    _merge_and_write_animated_ffmpeg_pipe(
                        rgb_frames, None, tmp_main, imgformat, frame_delays,
                        settings=settings, prefix="Write (no alpha)"
                    )

            # Clean up rgb_frames to free memory
            del rgb_frames
        else:
            # Standard path: write directly
            _write_clip_imwri(
                clip_main, tmp_main, imgformat,
                prefix="Frames (main)",
                frame_delays=frame_delays,
                settings=settings,
            )

        # Process secondary output if enabled
        # Note: Secondary output is only supported for single-frame images
        if settings.use_secondary_output and tmp_secondary is not None:
            if is_animated_output:
                print("Note: Secondary output is not supported for animated content, skipping")
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
    dest_stem, dest_dir_main = settings.compute_dest_stem_and_dir(input_path, output_dir)
    dest_name = f"{dest_stem}{file_ext}"
    dest_path = dest_dir_main / dest_name

    if not dest_path.exists():
        print(f"[alpha-worker] Main output not found, skipping alpha: {dest_path}")
        return

    try:
        # Build alpha clip - with or without SR upscaling
        if settings.upscale_enabled:
            alpha_sr = build_alpha_hq(input_path, settings)
        else:
            alpha_sr = load_alpha_no_upscale(input_path, settings)

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
        prefix = "Merge color + HQ alpha" if settings.upscale_enabled else "Merge color + alpha"
        _write_clip_imwri(
            color_sr,
            dest_path,
            imgformat,
            alpha_clip=alpha_sr,
            prefix=prefix,
        )

    except Exception as e:
        print(f"[alpha-worker] Alpha processing failed: {e}")

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
