"""Model-score comparison bar chart (with complexity penalty)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_model_score_comparison(
    model_scores: Dict[int, float],
    model_errors: Dict[int, float],
    chosen_model: int,
    output_path: Optional[str] = None,
    figsize: tuple = (8, 5),
    dpi: int = 150,
) -> None:
    """Bar chart comparing penalised model scores across motion-model orders.

    Shows the raw reprojection error as a hatched overlay to illustrate
    the penalty contribution.

    Parameters
    ----------
    model_scores : dict
        ``{order: penalised_score}``.
    model_errors : dict
        ``{order: mean_reprojection_error}``.
    chosen_model : int
        The selected model order (highlighted).
    """
    orders = sorted(model_scores.keys())
    scores = [model_scores[o] for o in orders]
    errors = [model_errors.get(o, 0.0) for o in orders]
    labels = [f"N={o}" for o in orders]
    colours = ["green" if o == chosen_model else "steelblue" for o in orders]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Total score bars
    bars = ax.bar(labels, scores, color=colours, edgecolor="black",
                  alpha=0.8, label="Score (error + penalty)")
    # Raw error overlay
    ax.bar(labels, errors, color="none", edgecolor="black",
           hatch="//", alpha=0.5, label="Raw error")

    ax.set_ylabel("Score / Error")
    ax.set_title(f"Model Score Comparison — chosen N={chosen_model}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{s:.4f}",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path)
        print(f"[plot] Saved model score comparison → {output_path}")
    plt.close(fig)
