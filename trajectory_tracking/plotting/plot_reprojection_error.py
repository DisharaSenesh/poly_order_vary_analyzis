"""Combined reprojection error overview (line + histogram side-by-side)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from trajectory_tracking.core.measurement import Measurement


def plot_reprojection_overview(
    measurements: List[Measurement],
    output_path: Optional[str] = None,
    figsize: tuple = (14, 5),
    dpi: int = 150,
) -> None:
    errors = [
        (m.frame_id, m.reprojection_error)
        for m in measurements
        if m.reprojection_error is not None
    ]
    if not errors:
        return

    frames = [e[0] for e in errors]
    vals = [e[1] for e in errors]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    ax1.plot(frames, vals, "o-", markersize=3, linewidth=1)
    ax1.set_xlabel("Frame")
    ax1.set_ylabel("Reprojection error (deg)")
    ax1.set_title("Error vs Frame")
    ax1.grid(True, alpha=0.3)

    ax2.hist(vals, bins=30, edgecolor="black", alpha=0.7)
    ax2.axvline(np.mean(vals), color="r", linestyle="--",
                label=f"mean={np.mean(vals):.4f}°")
    ax2.set_xlabel("Reprojection error (deg)")
    ax2.set_ylabel("Count")
    ax2.set_title("Error Histogram")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved reprojection overview → {output_path}")
    plt.close(fig)
