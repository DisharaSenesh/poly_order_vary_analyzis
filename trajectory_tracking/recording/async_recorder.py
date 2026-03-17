"""
Asynchronous CSV recorder.

Uses a background thread with a ``queue.Queue`` so that logging never
blocks the real-time pipeline.  Four output files are maintained:

* ``measurements.csv``       – raw + processed measurement fields
* ``trajectory_estimates.csv`` – position / velocity / acceleration
* ``model_selection.csv``    – per-window model comparison
* ``error_metrics.csv``      – per-window aggregate metrics
"""

from __future__ import annotations

import csv
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from trajectory_tracking.core.measurement import Measurement
from trajectory_tracking.evaluation.error_metrics import WindowMetrics


class AsyncRecorder:
    """Non-blocking CSV logger backed by a daemon writer thread.

    Parameters
    ----------
    output_dir : str | Path
        Directory where CSV files are created.
    measurements_file : str
    trajectory_file : str
    model_selection_file : str
    error_metrics_file : str
    """

    def __init__(
        self,
        output_dir: str = "output",
        measurements_file: str = "measurements.csv",
        trajectory_file: str = "trajectory_estimates.csv",
        model_selection_file: str = "model_selection.csv",
        error_metrics_file: str = "error_metrics.csv",
    ) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._q: queue.Queue[Optional[Dict[str, Any]]] = queue.Queue()

        # File handles and writers are opened lazily by the worker thread.
        self._files: Dict[str, Any] = {}
        self._writers: Dict[str, csv.DictWriter] = {}

        self._file_names = {
            "measurements": measurements_file,
            "trajectory": trajectory_file,
            "model_selection": model_selection_file,
            "error_metrics": error_metrics_file,
        }

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ───────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker, drain the queue, close files."""
        if not self._running:
            return
        self._q.put(None)  # sentinel
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._running = False
        for fh in self._files.values():
            fh.close()
        self._files.clear()
        self._writers.clear()

    # ── public logging methods (non-blocking) ───────────────────────
    def log_measurement(self, m: Measurement) -> None:
        self._q.put({"_kind": "measurements", **m.to_dict()})

    def log_trajectory(self, m: Measurement) -> None:
        d = {
            "frame_id": m.frame_id,
            "timestamp": m.timestamp,
        }
        for axis, label in enumerate(("x", "y", "z")):
            d[f"pos_{label}"] = float(m.position[axis]) if m.position is not None else None
            d[f"vel_{label}"] = float(m.velocity[axis]) if m.velocity is not None else None
            d[f"acc_{label}"] = float(m.acceleration[axis]) if m.acceleration is not None else None
        d["chosen_model"] = m.chosen_model
        d["reprojection_error"] = m.reprojection_error
        self._q.put({"_kind": "trajectory", **d})

    def log_model_selection(
        self,
        frame_id: int,
        chosen_model: int,
        errors: Dict[int, float],
        scores: Optional[Dict[int, float]] = None,
    ) -> None:
        d: Dict[str, Any] = {"frame_id": frame_id, "chosen_model": chosen_model}
        for order, err in sorted(errors.items()):
            d[f"error_N{order}"] = err
        if scores:
            for order, score in sorted(scores.items()):
                d[f"model_score_N{order}"] = score
        self._q.put({"_kind": "model_selection", **d})

    def log_error_metrics(
        self,
        frame_id: int,
        wm: WindowMetrics,
        error_before_BA: Optional[float] = None,
        error_after_BA: Optional[float] = None,
        bundle_adjustment_ran: Optional[bool] = None,
        condition_number: float = 0.0,
        window_size: int = 0,
    ) -> None:
        d: Dict[str, Any] = {
            "frame_id": frame_id,
            "mean_reprojection_error": wm.mean_reprojection_error,
            "max_reprojection_error": wm.max_reprojection_error,
            "mean_ray_angle": wm.mean_ray_angle,
            "baseline_mm": wm.baseline_mm,
            "chosen_model": wm.chosen_model,
            "error_before_BA": error_before_BA,
            "error_after_BA": error_after_BA,
            "bundle_adjustment_ran": bundle_adjustment_ran,
            "condition_number": condition_number,
            "window_size": window_size,
        }
        self._q.put({"_kind": "error_metrics", **d})

    # ── background worker ───────────────────────────────────────────
    def _worker(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                # Drain remaining items.
                while not self._q.empty():
                    leftover = self._q.get_nowait()
                    if leftover is not None:
                        self._write(leftover)
                break
            self._write(item)

    def _write(self, item: Dict[str, Any]) -> None:
        kind = item.pop("_kind")
        fname = self._file_names[kind]

        if kind not in self._writers:
            path = self._dir / fname
            fh = open(path, "w", newline="")
            self._files[kind] = fh
            writer = csv.DictWriter(fh, fieldnames=list(item.keys()))
            writer.writeheader()
            self._writers[kind] = writer

        self._writers[kind].writerow(item)
        self._files[kind].flush()
