"""
Detailed 3-D ray-geometry plot — shows individual ray–point residuals.

Draws a line from each camera centre through the estimated point along the
viewing ray, making triangulation quality immediately visible.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from trajectory_tracking.core.measurement import Measurement
from trajectory_tracking.utils.math_utils import normalize


def plot_ray_geometry_3d(
    measurements: List[Measurement],
    output_path: Optional[str] = None,
    figsize: tuple = (12, 9),
    dpi: int = 150,
) -> None:
    """3-D plot of rays from cameras towards the estimated object position.

    For each measurement the ray extends from the camera centre to the
    closest point on the ray to the estimated position, then a residual
    line is drawn from that closest point to the actual estimated position.
    """
    valid = [m for m in measurements if m.has_geometry() and m.position is not None]
    if not valid:
        return

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")

    for m in valid:
        C = m.camera_position
        d = m.ray_direction
        P = m.position

        # Project P onto the ray:  closest = C + t*d  where t = d·(P-C)
        t_proj = np.dot(d, P - C)
        closest = C + t_proj * d

        # Ray from C to closest point
        ax.plot(
            [C[0], closest[0]], [C[1], closest[1]], [C[2], closest[2]],
            "c-", alpha=0.5, linewidth=1,
        )
        # Residual from closest to estimated position
        ax.plot(
            [closest[0], P[0]], [closest[1], P[1]], [closest[2], P[2]],
            "r--", alpha=0.6, linewidth=0.8,
        )

    cam_pos = np.array([m.camera_position for m in valid])
    ax.plot(cam_pos[:, 0], cam_pos[:, 1], cam_pos[:, 2],
            "b^", markersize=5, label="Camera")

    pos = np.array([m.position for m in valid])
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
            "ro", markersize=4, label="Estimated position")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Ray Geometry — Residuals")
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved ray geometry → {output_path}")
    plt.close(fig)
