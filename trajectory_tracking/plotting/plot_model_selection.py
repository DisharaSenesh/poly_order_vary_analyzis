"""Model selection comparison bar chart."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_model_comparison(
    model_errors: Dict[int, float],
    chosen_model: int,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 5),
    dpi: int = 150,
) -> None:
    """Bar chart comparing reprojection error across motion-model orders.

    Parameters
    ----------
    model_errors : dict
        ``{order: mean_reprojection_error}``.
    chosen_model : int
        The selected model order (highlighted).
    """
    orders = sorted(model_errors.keys())
    errors = [model_errors[o] for o in orders]
    labels = [f"N={o}" for o in orders]
    colours = ["green" if o == chosen_model else "steelblue" for o in orders]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    bars = ax.bar(labels, errors, color=colours, edgecolor="black")
    ax.set_ylabel("Mean reprojection error (deg)")
    ax.set_title(f"Model Selection — chosen N={chosen_model}")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, err in zip(bars, errors):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{err:.4f}°",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved model comparison → {output_path}")
    plt.close(fig)
