"""Reprojection error plots (vs frame and histogram)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from trajectory_tracking.core.measurement import Measurement


def plot_reprojection_error_vs_frame(
    measurements: List[Measurement],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 7),
    dpi: int = 150,
) -> None:
    data = [
        (m.frame_id, m.reprojection_error)
        for m in measurements
        if m.reprojection_error is not None
    ]
    if not data:
        return

    frames = [d[0] for d in data]
    errors = [d[1] for d in data]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(frames, errors, "o-", markersize=3, linewidth=1)
    ax.set_xlabel("Frame")
    ax.set_ylabel("Reprojection error (deg)")
    ax.set_title("Reprojection Error vs Frame")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved reprojection error vs frame → {output_path}")
    plt.close(fig)


def plot_reprojection_error_histogram(
    measurements: List[Measurement],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 7),
    dpi: int = 150,
) -> None:
    errors = [m.reprojection_error for m in measurements if m.reprojection_error is not None]
    if not errors:
        return

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.hist(errors, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Reprojection error (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Reprojection Error Histogram")
    ax.axvline(np.mean(errors), color="r", linestyle="--", label=f"mean={np.mean(errors):.4f}°")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved reprojection error histogram → {output_path}")
    plt.close(fig)
