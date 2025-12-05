"""
Progress bar utilities for worker mode.

Supports rich progress bars when available, with an ASCII fallback.
"""

from __future__ import annotations

import time
from typing import Iterator, TypeVar, Iterable

T = TypeVar("T")

# Try to import rich for fancy progress bars
_RICH_AVAILABLE = False
try:
    from rich.progress import (
        Progress,
        BarColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        TaskProgressColumn,
        TextColumn,
    )
    _RICH_AVAILABLE = True
except ModuleNotFoundError:
    pass


def track(iterable: Iterable[T], total: int | None = None, prefix: str = "") -> Iterator[T]:
    """
    Wrap an iterable with a progress bar.

    Uses rich if available, otherwise falls back to a simple ASCII progress bar.

    Args:
        iterable: The iterable to wrap.
        total: Total number of items. If None, will try to use len().
        prefix: Description prefix to show.

    Yields:
        Items from the iterable.
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore
        except TypeError:
            total = None

    if _RICH_AVAILABLE:
        yield from _track_rich(iterable, total, prefix)
    else:
        yield from _track_ascii(iterable, total, prefix)


def _track_rich(iterable: Iterable[T], total: int | None, description: str) -> Iterator[T]:
    """Rich progress bar implementation."""
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        transient=False,
    ) as progress:
        task_id = progress.add_task(description, total=total)
        for item in iterable:
            yield item
            progress.advance(task_id)


def _track_ascii(iterable: Iterable[T], total: int | None, prefix: str) -> Iterator[T]:
    """Simple ASCII progress bar fallback."""
    if total is None:
        # Can't show progress without knowing total
        yield from iterable
        return

    start_time = time.perf_counter()
    bar_width = 60

    def format_hms(seconds: float) -> str:
        return time.strftime("%H:%M:%S", time.gmtime(seconds))

    def print_bar(iteration: int) -> None:
        elapsed = time.perf_counter() - start_time
        elapsed_str = format_hms(elapsed)

        if iteration == 0:
            eta_str = "--:--:--"
        else:
            avg = elapsed / iteration
            remaining = avg * (total - iteration)
            eta_str = format_hms(remaining)

        percent = 100 * iteration / total
        filled = int(bar_width * iteration // total)
        bar = "█" * filled + "-" * (bar_width - filled)
        print(
            f"\r{prefix} |{bar}| {percent:.1f}% Elapsed: {elapsed_str} ETA: {eta_str} ",
            end="\r",
        )

    print_bar(0)
    for i, item in enumerate(iterable):
        yield item
        print_bar(i + 1)
    print()  # Newline after completion
