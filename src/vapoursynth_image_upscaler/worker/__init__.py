"""
Worker module for VapourSynth super-resolution processing.

This module is only imported when running in worker mode (--worker or --alpha-worker)
to keep the GUI startup light and avoid loading heavy VapourSynth dependencies.
"""

from .main import worker_main, is_worker_mode, is_alpha_worker_mode

__all__ = [
    "worker_main",
    "is_worker_mode",
    "is_alpha_worker_mode",
]
