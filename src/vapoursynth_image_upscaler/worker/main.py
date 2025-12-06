"""
Worker mode entry point.

This module handles the CLI interface for worker processes spawned by the GUI.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path


def is_worker_mode() -> bool:
    """Check if running in worker mode (--worker, --alpha-worker, or --batch-worker)."""
    return any(m in sys.argv[1:] for m in ("--worker", "--alpha-worker", "--batch-worker"))


def is_alpha_worker_mode() -> bool:
    """Check if running specifically in alpha worker mode."""
    return "--alpha-worker" in sys.argv[1:]


def is_batch_worker_mode() -> bool:
    """Check if running in batch worker mode."""
    return "--batch-worker" in sys.argv[1:]


def worker_main() -> None:
    """
    Worker entry point.

    Single file mode:
        python script.py --worker <input> <output_dir> <secondary_output_dir>

    Alpha-only mode:
        python script.py --alpha-worker <input> <output_dir> <secondary_output_dir>

    Batch mode:
        python script.py --batch-worker <manifest_json_path>
        The manifest contains file lists, output dirs, and settings.
    """
    # These imports are deferred to avoid loading heavy dependencies in GUI mode
    from vstools import core
    from .settings import WorkerSettings

    # Set cache size
    core.max_cache_size = 24000

    if is_batch_worker_mode():
        _batch_worker_main(core)
    else:
        _single_worker_main(core)


def _single_worker_main(core) -> None:
    """Handle single-file worker mode."""
    from .settings import WorkerSettings
    from .processor import process_one, process_one_alpha

    # Parse arguments
    args = [a for a in sys.argv[1:] if a not in ("--worker", "--alpha-worker")]
    if len(args) < 3:
        print("Worker usage: --worker/--alpha-worker <input> <output_dir> <secondary_output_dir>")
        sys.exit(1)

    input_path = Path(args[0]).resolve()
    output_dir = Path(args[1]).resolve()
    secondary_output_dir = Path(args[2]).resolve()

    # Load settings from environment
    settings = WorkerSettings.from_environment()

    try:
        if is_alpha_worker_mode():
            process_one_alpha(input_path, output_dir, secondary_output_dir, settings)
        else:
            process_one(input_path, output_dir, secondary_output_dir, settings)
    finally:
        # Final cleanup
        try:
            core.clear_cache()
        except Exception:
            pass
        gc.collect()

    sys.exit(0)


def _batch_worker_main(core) -> None:
    """Handle batch worker mode."""
    from .settings import WorkerSettings
    from .batch_processor import (
        process_batch,
        group_by_resolution_and_format,
        split_into_batches,
        MAX_BATCH_SIZE,
    )

    # Parse arguments
    args = [a for a in sys.argv[1:] if a != "--batch-worker"]
    if len(args) < 1:
        print("Batch worker usage: --batch-worker <manifest_json_path>")
        sys.exit(1)

    manifest_path = Path(args[0]).resolve()
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)

    # Load manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    files = [Path(p) for p in manifest["files"]]
    output_dirs = [Path(p) for p in manifest["output_dirs"]]
    secondary_dirs = [Path(p) for p in manifest["secondary_dirs"]]
    progress_file_path = manifest.get("progress_file")
    progress_file = Path(progress_file_path) if progress_file_path else None

    # Load settings from environment
    settings = WorkerSettings.from_environment()

    total_files = len(files)
    batch_offset = 0

    try:
        # Group files by resolution AND format
        groups = group_by_resolution_and_format(files)
        print(f"Processing {len(files)} files in {len(groups)} resolution/format group(s)")

        all_times: list[float] = []

        for (w, h, fmt), group_files in groups.items():
            # Split large groups into batches of MAX_BATCH_SIZE
            batches = split_into_batches(group_files, MAX_BATCH_SIZE)

            for batch_idx, batch_files in enumerate(batches):
                batch_label = f" (batch {batch_idx + 1}/{len(batches)})" if len(batches) > 1 else ""
                print(f"\nProcessing group {w}x{h} {fmt} ({len(batch_files)} files){batch_label}")

                # Get corresponding output/secondary dirs for this batch
                batch_output_dirs = []
                batch_secondary_dirs = []
                for bf in batch_files:
                    idx = files.index(bf)
                    batch_output_dirs.append(output_dirs[idx])
                    batch_secondary_dirs.append(secondary_dirs[idx])

                # Process this batch
                times = process_batch(
                    batch_files,
                    batch_output_dirs,
                    batch_secondary_dirs,
                    settings,
                    progress_file=progress_file,
                    batch_offset=batch_offset,
                    total_files=total_files,
                )
                all_times.extend(times)
                batch_offset += len(batch_files)

                # Clear VRAM between batches to prevent accumulation
                try:
                    core.clear_cache()
                except Exception:
                    pass
                gc.collect()

        # Write timing results back
        result_path = manifest_path.with_suffix(".result")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump({"times": all_times}, f)

    finally:
        # Final cleanup
        try:
            core.clear_cache()
        except Exception:
            pass
        gc.collect()

    sys.exit(0)
