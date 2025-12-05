"""
VapourSynth super-resolution pipeline.

Contains the core SR processing logic using vsmlrt and TensorRT backend.
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import TYPE_CHECKING

# These imports are heavy and only loaded in worker mode
import vsmlrt
from vstools import vs, core, depth, initialize_clip, padder
from vsmlrt import Backend
from vskernels import Hermite, Lanczos
from vssource import BestSource

from ..core.utils import compute_padding
from ..core.constants import PADDING_ALIGNMENT, SUPPORTED_IMAGE_EXTENSIONS
from .settings import WorkerSettings

if TYPE_CHECKING:
    pass

# Global for timing from source creation onward
_process_start_time: float | None = None


def get_process_start_time() -> float | None:
    """Get the timestamp when processing started for the current file."""
    return _process_start_time


def _get_kernel(kernel_name: str):
    """Get the appropriate scaling kernel based on name."""
    if kernel_name.lower() == "hermite":
        return Hermite()
    else:
        return Lanczos()


def _build_backend(settings: WorkerSettings) -> Backend:
    """
    Build the vsmlrt TensorRT backend with configured settings.

    Backend shapes follow tile limits: opt/max = (tile_w_limit, tile_h_limit).
    """
    return Backend.TRT(
        fp16=settings.use_fp16,
        bf16=settings.use_bf16,
        tf32=settings.use_tf32,
        static_shape=False,
        builder_optimization_level=5,
        min_shapes=[128, 128],
        opt_shapes=[settings.tile_w_limit, settings.tile_h_limit],
        max_shapes=[settings.tile_w_limit, settings.tile_h_limit],
    )


def compute_prescale_dimensions(
    source_width: int,
    source_height: int,
    mode: str,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    """
    Compute pre-scale dimensions based on mode.

    Args:
        source_width: Width of the source.
        source_height: Height of the source.
        mode: "width", "height", or "2x".
        target_width: Target width for "width" mode.
        target_height: Target height for "height" mode.

    Returns:
        Tuple of (width, height) for pre-scaled output.
    """
    if mode == "2x":
        return max(1, int(source_width * 0.5 + 0.5)), max(1, int(source_height * 0.5 + 0.5))
    elif mode == "height":
        new_h = max(1, target_height)
        new_w = max(1, int(source_width * new_h / source_height + 0.5))
        return new_w, new_h
    else:  # "width" or default
        new_w = max(1, target_width)
        new_h = max(1, int(source_height * new_w / source_width + 0.5))
        return new_w, new_h


def apply_prescale(clip: vs.VideoNode, settings: WorkerSettings) -> vs.VideoNode:
    """
    Apply pre-scaling to a clip before upscaling.

    Args:
        clip: Input clip (already in RGBS format).
        settings: Worker settings with prescale configuration.

    Returns:
        Pre-scaled clip.
    """
    if not settings.prescale_enabled:
        return clip

    # Compute target dimensions
    target_w, target_h = compute_prescale_dimensions(
        clip.width, clip.height,
        settings.prescale_mode,
        settings.prescale_width,
        settings.prescale_height,
    )

    # Apply scaling using selected kernel
    kernel = _get_kernel(settings.kernel)
    clip = depth(clip, 32)
    return kernel.scale(clip, target_w, target_h, linear=True)


def build_clip(input_path: Path, settings: WorkerSettings) -> vs.VideoNode:
    """
    Build the VapourSynth processing chain for color super-resolution.

    Timing starts right before source creation.

    Args:
        input_path: Path to the input image/video file.
        settings: Worker settings from environment.

    Returns:
        Processed clip ready for output.
    """
    global _process_start_time
    _process_start_time = time.perf_counter()

    ext = input_path.suffix.lower()
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    if ext in img_exts:
        src = core.imwri.Read(str(input_path))
    else:
        src = BestSource.source(str(input_path))

    clip = initialize_clip(src)

    # Convert to RGBS (32-bit float RGB)
    clip = core.resize.Bicubic(clip, format=vs.RGBS, matrix_in=0, range_in=1, range=1)

    # Apply pre-scaling if enabled (before padding and upscaling)
    clip = apply_prescale(clip, settings)

    # Compute and apply padding to multiple of 64
    pad_left, pad_right, pad_top, pad_bottom = compute_padding(
        clip.width, clip.height, PADDING_ALIGNMENT
    )
    clip = padder.MIRROR(clip, pad_left, pad_right, pad_top, pad_bottom)

    # Clamp to valid range
    clip = core.std.Expr(clip, expr=["x 0 max 1 min"])

    # Compute tile sizes
    tile_h = min(settings.tile_h_limit, clip.height)
    tile_w = min(settings.tile_w_limit, clip.width)

    # Run through vsmlrt with TensorRT backend
    clip = vsmlrt.inference(
        clip,
        backend=_build_backend(settings),
        overlap=[16, 16],
        tilesize=[tile_w, tile_h],
        network_path=settings.onnx_path,
    )

    # Remove padding (scaled by model scale)
    scale = settings.model_scale
    clip = core.std.Crop(
        clip,
        left=pad_left * scale,
        right=pad_right * scale,
        top=pad_top * scale,
        bottom=pad_bottom * scale,
    )

    return clip


def build_alpha_hq(input_path: Path, settings: WorkerSettings) -> vs.VideoNode:
    """
    Build high-quality SR alpha channel.

    Steps:
    - Read the original source with alpha (PNG via imwri, GIF/WEBP via BestSource).
    - Extract alpha from frame properties.
    - Convert to RGBS, pad to multiples of 64.
    - Run through the SR model.
    - Remove padding (scaled by model scale).
    - Convert to GRAY8 for use as fpng alpha.

    Args:
        input_path: Path to the input file with alpha channel.
        settings: Worker settings from environment.

    Returns:
        GRAY8 alpha clip.
    """
    global _process_start_time
    _process_start_time = time.perf_counter()

    ext = input_path.suffix.lower()
    if ext == ".png":
        src = core.imwri.Read(str(input_path), alpha=True)
    elif ext in {".webp", ".gif"}:
        src = BestSource.source(str(input_path))
    else:
        raise ValueError("HQ alpha workflow only supports PNG / GIF / WEBP inputs")

    clip = initialize_clip(src)

    # Extract alpha from frame properties
    alpha = core.std.PropToClip(clip)

    # Convert alpha to RGBS for the model
    alpha = core.resize.Bicubic(
        alpha,
        format=vs.RGBS,
        transfer_in=13,
        primaries_in=1,
        range_in=1,
        range=1,
    )

    # Apply pre-scaling if enabled (before padding and upscaling)
    alpha = apply_prescale(alpha, settings)

    # Compute and apply padding
    pad_left, pad_right, pad_top, pad_bottom = compute_padding(
        alpha.width, alpha.height, PADDING_ALIGNMENT
    )
    alpha = padder.MIRROR(alpha, pad_left, pad_right, pad_top, pad_bottom)

    # Clamp range
    alpha = core.std.Expr(alpha, expr=["x 0 max 1 min"])

    # Compute tile sizes
    tile_h = min(settings.tile_h_limit, alpha.height)
    tile_w = min(settings.tile_w_limit, alpha.width)

    # Run through vsmlrt
    alpha = vsmlrt.inference(
        alpha,
        backend=_build_backend(settings),
        tilesize=[tile_w, tile_h],
        network_path=settings.onnx_path,
    )

    # Remove padding (scaled by model scale)
    scale = settings.model_scale
    alpha = core.std.Crop(
        alpha,
        left=pad_left * scale,
        right=pad_right * scale,
        top=pad_top * scale,
        bottom=pad_bottom * scale,
    )

    # Convert to GRAY8 for fpng alpha
    alpha = core.resize.Bicubic(
        alpha,
        format=vs.GRAY8,
        matrix=1,
        range_in=1,
        range=1,
    )

    return alpha


def apply_custom_resolution(
    clip: vs.VideoNode, width: int, height: int, kernel_name: str = "lanczos"
) -> vs.VideoNode:
    """
    Downscale a clip to a custom resolution using the specified kernel.

    Args:
        clip: Input clip.
        width: Target width.
        height: Target height.
        kernel_name: Kernel to use ("lanczos" or "hermite").

    Returns:
        Scaled clip.
    """
    kernel = _get_kernel(kernel_name)
    clip = depth(clip, 32)
    return kernel.scale(clip, width, height, linear=True)


def compute_secondary_dimensions(
    source_width: int,
    source_height: int,
    mode: str,
    target_width: int,
    target_height: int,
) -> tuple[int, int]:
    """
    Compute secondary output dimensions based on mode.

    Args:
        source_width: Width of the source (main output).
        source_height: Height of the source (main output).
        mode: "width", "height", or "2x".
        target_width: Target width for "width" mode.
        target_height: Target height for "height" mode.

    Returns:
        Tuple of (width, height) for secondary output.
    """
    if mode == "2x":
        return max(1, int(source_width * 0.5 + 0.5)), max(1, int(source_height * 0.5 + 0.5))
    elif mode == "height":
        new_h = max(1, target_height)
        new_w = max(1, int(source_width * new_h / source_height + 0.5))
        return new_w, new_h
    else:  # "width" or default
        new_w = max(1, target_width)
        new_h = max(1, int(source_height * new_w / source_width + 0.5))
        return new_w, new_h


def apply_sharpening(clip: vs.VideoNode, settings: WorkerSettings) -> vs.VideoNode:
    """
    Apply CAS (Contrast Adaptive Sharpening) to a clip if enabled.

    Args:
        clip: Input clip in RGBS format.
        settings: Worker settings with sharpening configuration.

    Returns:
        Sharpened clip if enabled, otherwise the original clip.
    """
    if settings.sharpen_enabled and settings.sharpen_value > 0:
        return core.cas.CAS(clip, sharpness=settings.sharpen_value)
    return clip


def clear_cache() -> None:
    """Clear VapourSynth cache and run garbage collection."""
    try:
        core.clear_cache()
    except Exception as e:
        print(f"Cache cleanup warning (non-fatal): {e}")
    gc.collect()
