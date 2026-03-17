"""3-D trajectory plot."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from trajectory_tracking.core.measurement import Measurement


def plot_3d_trajectory(
    measurements: List[Measurement],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 7),
    dpi: int = 150,
) -> None:
    """Plot estimated 3-D positions with camera centres.

    Parameters
    ----------
    measurements : list[Measurement]
        Must have ``position`` and ``camera_position`` populated.
    output_path : str | None
        If given, save the figure to this path.
    """
    positions = np.array([m.position for m in measurements if m.position is not None])
    cam_positions = np.array(
        [m.camera_position for m in measurements if m.camera_position is not None]
    )

    if len(positions) == 0:
        return

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        positions[:, 0], positions[:, 1], positions[:, 2],
        "r.-", label="Estimated trajectory", linewidth=1.5,
    )

    if len(cam_positions) > 0:
        ax.plot(
            cam_positions[:, 0], cam_positions[:, 1], cam_positions[:, 2],
            "b^-", markersize=4, label="Camera centres", alpha=0.6,
        )

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3-D Trajectory")
    ax.legend()

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved 3-D trajectory → {output_path}")
    plt.close(fig)
