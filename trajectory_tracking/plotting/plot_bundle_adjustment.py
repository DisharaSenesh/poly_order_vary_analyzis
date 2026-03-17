"""Bundle adjustment improvement plot (diagnostic)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from trajectory_tracking.core.measurement import Measurement


def plot_bundle_adjustment(
    measurements: List[Measurement],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 7),
    dpi: int = 150,
) -> None:
    """Plot reprojection error before and after bundle adjustment.

    Only frames where BA actually ran are shown.

    Parameters
    ----------
    measurements : list[Measurement]
    """
    data = [
        (m.frame_id, m.error_before_BA, m.error_after_BA)
        for m in measurements
        if m.error_before_BA is not None and m.error_after_BA is not None
    ]
    if not data:
        print("[plot] No bundle adjustment data to plot.")
        return

    frames = [d[0] for d in data]
    before = [d[1] for d in data]
    after = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(frames, before, "o-", markersize=4, linewidth=1.2,
            color="tomato", label="Before BA")
    ax.plot(frames, after, "s-", markersize=4, linewidth=1.2,
            color="seagreen", label="After BA")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Mean reprojection error (deg)")
    ax.set_title("Bundle Adjustment Improvement")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved bundle adjustment plot → {output_path}")
    plt.close(fig)
