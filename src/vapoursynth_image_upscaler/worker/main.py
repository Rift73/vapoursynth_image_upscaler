"""
Worker mode entry point.

This module handles the CLI interface for worker processes spawned by the GUI.
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path


def is_worker_mode() -> bool:
    """Check if running in worker mode (--worker or --alpha-worker)."""
    return "--worker" in sys.argv[1:] or "--alpha-worker" in sys.argv[1:]


def is_alpha_worker_mode() -> bool:
    """Check if running specifically in alpha worker mode."""
    return "--alpha-worker" in sys.argv[1:]


def worker_main() -> None:
    """
    Worker entry point.

    Main mode:
        python script.py --worker <input> <output_dir> <secondary_output_dir>

    Alpha-only mode:
        python script.py --alpha-worker <input> <output_dir> <secondary_output_dir>
    """
    # These imports are deferred to avoid loading heavy dependencies in GUI mode
    from vstools import core
    from .settings import WorkerSettings
    from .processor import process_one, process_one_alpha

    # Set cache size
    core.max_cache_size = 24000

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
