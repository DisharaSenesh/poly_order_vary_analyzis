"""Ray angle vs frame plot (debugging)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from trajectory_tracking.core.measurement import Measurement


def plot_ray_angle_vs_frame(
    measurements: List[Measurement],
    min_angle_threshold: float = 2.0,
    output_path: Optional[str] = None,
    figsize: tuple = (10, 7),
    dpi: int = 150,
) -> None:
    """Plot the inter-ray angle for each frame.

    A horizontal dashed line marks the rejection threshold.

    Parameters
    ----------
    measurements : list[Measurement]
    min_angle_threshold : float
        Angle threshold in degrees (shown as a horizontal line).
    """
    data = [
        (m.frame_id, m.ray_angle)
        for m in measurements
        if m.ray_angle is not None
    ]
    if not data:
        return

    frames = [d[0] for d in data]
    angles = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(frames, angles, "o-", markersize=3, linewidth=1, label="Ray angle")
    ax.axhline(min_angle_threshold, color="r", linestyle="--", linewidth=1,
               label=f"Threshold = {min_angle_threshold}°")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Ray angle (deg)")
    ax.set_title("Ray Angle vs Frame")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved ray angle vs frame → {output_path}")
    plt.close(fig)
