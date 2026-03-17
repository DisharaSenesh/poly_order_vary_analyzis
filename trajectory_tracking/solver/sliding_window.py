"""
Fixed-capacity sliding window for ``Measurement`` objects.

The window keeps the most recent *max_size* measurements.  Solving is
only attempted once the window contains at least *min_size* entries.
"""

from __future__ import annotations

from collections import deque
from typing import List

from trajectory_tracking.core.measurement import Measurement


class SlidingWindow:
    """FIFO window of measurements.

    Parameters
    ----------
    min_size : int
        Minimum number of measurements before solving can proceed.
    max_size : int
        Maximum window capacity; oldest entries are discarded first.
    """

    def __init__(self, min_size: int = 5, max_size: int = 20) -> None:
        self.min_size = min_size
        self.max_size = max_size
        self._buffer: deque[Measurement] = deque(maxlen=max_size)

    # ── mutators ────────────────────────────────────────────────────
    def push(self, m: Measurement) -> None:
        """Append a measurement (automatically evicts the oldest if full)."""
        self._buffer.append(m)

    def clear(self) -> None:
        """Remove all measurements."""
        self._buffer.clear()

    # ── queries ─────────────────────────────────────────────────────
    @property
    def ready(self) -> bool:
        """True when enough measurements have been accumulated."""
        return len(self._buffer) >= self.min_size

    @property
    def measurements(self) -> List[Measurement]:
        """Return a *copy* of the current window contents (oldest first)."""
        return list(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"SlidingWindow(len={len(self)}, "
            f"min={self.min_size}, max={self.max_size})"
        )
