"""Camera baseline vs frame plot (diagnostic)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from trajectory_tracking.core.measurement import Measurement


def plot_baseline_vs_frame(
    measurements: List[Measurement],
    min_baseline_mm: float = 5.0,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 7),
    dpi: int = 150,
) -> None:
    """Plot the camera baseline (mm) for each frame.

    A horizontal dashed line marks the rejection threshold.

    Parameters
    ----------
    measurements : list[Measurement]
    min_baseline_mm : float
        Baseline threshold in mm (shown as a horizontal line).
    """
    data = [
        (m.frame_id, m.baseline_mm)
        for m in measurements
        if m.baseline_mm is not None
    ]
    if not data:
        return

    frames = [d[0] for d in data]
    baselines = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(frames, baselines, "o-", markersize=3, linewidth=1,
            color="steelblue", label="Baseline")
    ax.axhline(min_baseline_mm, color="r", linestyle="--", linewidth=1,
               label=f"Threshold = {min_baseline_mm} mm")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Camera baseline (mm)")
    ax.set_title("Camera Baseline vs Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved baseline vs frame → {output_path}")
    plt.close(fig)
