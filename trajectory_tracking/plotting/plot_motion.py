"""Position / velocity / acceleration vs time plots."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

from trajectory_tracking.core.measurement import Measurement


def _plot_vector_vs_time(
    measurements: List[Measurement],
    attr: str,
    ylabel: str,
    title: str,
    output_path: Optional[str],
    figsize: tuple,
    dpi: int,
) -> None:
    data = [(m.timestamp, getattr(m, attr)) for m in measurements if getattr(m, attr) is not None]
    if not data:
        return

    t = np.array([d[0] for d in data])
    t -= t[0]
    vals = np.array([d[1] for d in data])

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for axis, label in enumerate(("X", "Y", "Z")):
        ax.plot(t, vals[:, axis], ".-", label=label)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved {title} → {output_path}")
    plt.close(fig)


def plot_position_vs_time(
    measurements: List[Measurement],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 7),
    dpi: int = 150,
) -> None:
    _plot_vector_vs_time(
        measurements, "position", "Position (mm)",
        "Position vs Time", output_path, figsize, dpi,
    )


def plot_velocity_vs_time(
    measurements: List[Measurement],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 7),
    dpi: int = 150,
) -> None:
    _plot_vector_vs_time(
        measurements, "velocity", "Velocity (mm/s)",
        "Velocity vs Time", output_path, figsize, dpi,
    )


def plot_acceleration_vs_time(
    measurements: List[Measurement],
    output_path: Optional[str] = None,
    figsize: tuple = (10, 7),
    dpi: int = 150,
) -> None:
    _plot_vector_vs_time(
        measurements, "acceleration", "Acceleration (mm/s²)",
        "Acceleration vs Time", output_path, figsize, dpi,
    )
