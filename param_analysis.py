#!/usr/bin/env python3
"""
Parameter Sensitivity Study — Analysis & Plotting
===================================================

Reads:
    results/param_sweep/sweep_results.csv
    results/param_sweep/sweep_per_frame.csv

Writes plots to:
    results/param_sweep/plots/
        Plot 1 — OPAT line plots (3 files)
            opat_ray_angle.png
            opat_baseline.png
            opat_window_size.png
        Plot 2 — Error-bar plots over factorial data (3 files)
            errorbar_ray_angle.png
            errorbar_baseline.png
            errorbar_window_size.png
        Plot 3 — 2-D interaction heatmaps  X & Z error (6 files)
            heatmap_ray_vs_base_xerr.png   heatmap_ray_vs_base_zerr.png
            heatmap_ray_vs_win_xerr.png    heatmap_ray_vs_win_zerr.png
            heatmap_base_vs_win_xerr.png   heatmap_base_vs_win_zerr.png
        Plot 4 — Solved-frame count heatmaps (3 files)
            heatmap_ray_vs_base_nsolved.png
            heatmap_ray_vs_win_nsolved.png
            heatmap_base_vs_win_nsolved.png
        Plot 5 — Best / worst summary dashboard (1 file)
            best_worst_summary.png

Usage
-----
    python3 param_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")                      # headless — no display needed
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT        = Path(__file__).resolve().parent
SWEEP_DIR    = _ROOT / "results" / "param_sweep_sin_1"
RESULTS_CSV  = SWEEP_DIR / "sweep_results.csv"
PF_CSV       = SWEEP_DIR / "sweep_per_frame.csv"
PLOT_DIR     = SWEEP_DIR / "plots"

# results/param_sweep_acc_1

DPI          = 150
FIGSIZE_STD  = (9, 5)
FIGSIZE_HEAT = (8, 6)
FIGSIZE_DASH = (14, 8)

# Colour scheme
C_X   = "#1f77b4"   # blue  — X error
C_Z   = "#d62728"   # red   — Z error
C_SOL = "#2ca02c"   # green — n_solved


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not RESULTS_CSV.exists():
        sys.exit(
            f"[ERROR] sweep_results.csv not found at {RESULTS_CSV}\n"
            "        Run  python3 param_sweep.py  first."
        )
    df   = pd.read_csv(RESULTS_CSV)
    df_pf = pd.read_csv(PF_CSV) if PF_CSV.exists() else pd.DataFrame()
    print(f"Loaded {len(df)} summary rows from {RESULTS_CSV}")
    return df, df_pf


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, name: str) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    p = PLOT_DIR / name
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {p.relative_to(_ROOT)}")


def _heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    row_labels,
    col_labels,
    title: str,
    cmap: str = "viridis_r",
    fmt: str = ".0f",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """Draw an annotated heatmap on *ax*."""
    im = ax.imshow(
        data, aspect="auto", cmap=cmap,
        vmin=vmin if vmin is not None else np.nanmin(data),
        vmax=vmax if vmax is not None else np.nanmax(data),
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, shrink=0.85)

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels([str(c) for c in col_labels], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([str(r) for r in row_labels], fontsize=7)
    ax.set_title(title, fontsize=10, fontweight="bold")

    # Annotate cells with dynamic font sizing based on grid size
    # Always annotate now (removed the size check that skipped small grids)
    n_cells = data.shape[0] * data.shape[1]
    if n_cells <= 100:
        font_size = 7
    elif n_cells <= 256:
        font_size = 5
    else:
        font_size = 4

    norm_data = (data - np.nanmin(data)) / max(np.nanmax(data) - np.nanmin(data), 1e-9)
    for ri in range(data.shape[0]):
        for ci in range(data.shape[1]):
            val = data[ri, ci]
            if np.isnan(val):
                txt = "—"
            else:
                txt = f"{val:{fmt}}"
            colour = "white" if norm_data[ri, ci] > 0.55 else "black"
            ax.text(ci, ri, txt, ha="center", va="center",
                    fontsize=font_size, color=colour)


# ---------------------------------------------------------------------------
# Plot 1 — OPAT line plots
# ---------------------------------------------------------------------------

def plot1_opat(df: pd.DataFrame) -> None:
    print("\n[Plot 1] OPAT line plots")

    specs = [
        # (opat_flag_col, x_col,           xlabel,                  filename,           x_default)
        ("opat_ray",    "ray_angle_deg",  "min_ray_angle_deg (°)",  "opat_ray_angle.png",   3),
        ("opat_base",   "baseline_mm",    "min_baseline_mm (mm)",   "opat_baseline.png",    9),
        ("opat_window", "window_size",    "sliding_window.max_size","opat_window_size.png", 50),
    ]

    for flag, x_col, xlabel, fname, x_default in specs:
        sub = df[df[flag]].copy().sort_values(x_col)
        if sub.empty:
            print(f"  [SKIP] no rows with {flag}=True")
            continue

        fig, ax = plt.subplots(figsize=FIGSIZE_STD)

        # X error
        ax.plot(sub[x_col], sub["x_err_mean"], color=C_X, marker="o",
                markersize=4, linewidth=1.5, label="X error mean")
        ax.fill_between(
            sub[x_col],
            sub["x_err_mean"] - sub["x_err_std"],
            sub["x_err_mean"] + sub["x_err_std"],
            alpha=0.20, color=C_X, label="X error ±1σ",
        )

        # Z error
        ax.plot(sub[x_col], sub["z_err_mean"], color=C_Z, marker="s",
                markersize=4, linewidth=1.5, label="Z error mean")
        ax.fill_between(
            sub[x_col],
            sub["z_err_mean"] - sub["z_err_std"],
            sub["z_err_mean"] + sub["z_err_std"],
            alpha=0.20, color=C_Z, label="Z error ±1σ",
        )

        # Default marker
        ax.axvline(x_default, color="grey", linestyle="--",
                   linewidth=1, alpha=0.7, label=f"default={x_default}")

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Absolute error (mm)", fontsize=11)
        ax.set_title(f"OPAT — effect of {xlabel}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, fname)


# ---------------------------------------------------------------------------
# Plot 2 — Error-bar plots (aggregated over factorial data)
# ---------------------------------------------------------------------------

def plot2_errorbars(df: pd.DataFrame) -> None:
    print("\n[Plot 2] Error-bar plots (factorial data)")

    specs = [
        ("ray_angle_deg",  "min_ray_angle_deg (°)",   "errorbar_ray_angle.png"),
        ("baseline_mm",    "min_baseline_mm (mm)",     "errorbar_baseline.png"),
        ("window_size",    "sliding_window.max_size",  "errorbar_window_size.png"),
    ]

    for x_col, xlabel, fname in specs:
        grp = df.groupby(x_col).agg(
            x_mean=("x_err_mean", "mean"),
            x_std=("x_err_mean", "std"),
            x_min=("x_err_min",  "min"),
            x_max=("x_err_max",  "max"),
            z_mean=("z_err_mean", "mean"),
            z_std=("z_err_mean", "std"),
            z_min=("z_err_min",  "min"),
            z_max=("z_err_max",  "max"),
        ).reset_index().sort_values(x_col)

        xs = grp[x_col].values

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

        for ax, mean_col, std_col, min_col, max_col, colour, label in [
            (axes[0], "x_mean", "x_std", "x_min", "x_max", C_X, "X error"),
            (axes[1], "z_mean", "z_std", "z_min", "z_max", C_Z, "Z error"),
        ]:
            means  = grp[mean_col].values
            stds   = grp[std_col].fillna(0).values
            mins   = grp[min_col].values
            maxs   = grp[max_col].values

            # Min–max whiskers
            ax.errorbar(
                xs, means,
                yerr=[means - mins, maxs - means],
                fmt="none", ecolor=colour, alpha=0.3,
                capsize=3, linewidth=1, label="min–max",
            )
            # ±1σ bars
            ax.errorbar(
                xs, means,
                yerr=stds,
                fmt="o", color=colour, markersize=5,
                elinewidth=2, capsize=4, label="mean ±1σ",
            )

            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel("Absolute error (mm)", fontsize=11)
            ax.set_title(f"{label} vs {xlabel}", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Error-bar plot — {xlabel}", fontsize=13, fontweight="bold")
        fig.tight_layout()
        _save(fig, fname)


# ---------------------------------------------------------------------------
# Plot 3 — 2-D interaction heatmaps (X and Z error)
# ---------------------------------------------------------------------------

def plot3_heatmaps(df: pd.DataFrame) -> None:
    print("\n[Plot 3] 2-D interaction heatmaps (X and Z errors)")

    pairs = [
        # (row_param,       col_param,       margin_param,    row_label,         col_label)
        ("ray_angle_deg",  "baseline_mm",   "window_size",   "ray_angle_deg",   "baseline_mm"),
        ("ray_angle_deg",  "window_size",   "baseline_mm",   "ray_angle_deg",   "window_size"),
        ("baseline_mm",    "window_size",   "ray_angle_deg", "baseline_mm",     "window_size"),
    ]

    for row_p, col_p, margin_p, row_label, col_label in pairs:
        pivot_x = (
            df.groupby([row_p, col_p])["x_err_mean"]
            .mean()
            .unstack(col_p)
        )
        pivot_z = (
            df.groupby([row_p, col_p])["z_err_mean"]
            .mean()
            .unstack(col_p)
        )

        slug = f"{row_p[:3]}_vs_{col_p[:3]}"

        for pivot, err_label, colour, fname_sfx in [
            (pivot_x, "Mean X error (mm)", "Blues",  f"heatmap_{slug}_xerr.png"),
            (pivot_z, "Mean Z error (mm)", "Reds",   f"heatmap_{slug}_zerr.png"),
        ]:
            fig, ax = plt.subplots(figsize=FIGSIZE_HEAT)
            data = pivot.values
            _heatmap(
                ax, data,
                row_labels=pivot.index.tolist(),
                col_labels=pivot.columns.tolist(),
                title=f"{err_label}\n({row_label} vs {col_label},  marginalized over {margin_p})",
                cmap=f"{colour}_r",
                fmt=".1f",
            )
            ax.set_ylabel(row_label, fontsize=10)
            ax.set_xlabel(col_label, fontsize=10)
            fig.tight_layout()
            _save(fig, fname_sfx)


# ---------------------------------------------------------------------------
# Plot 4 — Solved-frame count heatmaps
# ---------------------------------------------------------------------------

def plot4_nsolved_heatmaps(df: pd.DataFrame) -> None:
    print("\n[Plot 4] Solved-frame count heatmaps")

    pairs = [
        ("ray_angle_deg",  "baseline_mm",   "window_size",   "ray_angle_deg",   "baseline_mm"),
        ("ray_angle_deg",  "window_size",   "baseline_mm",   "ray_angle_deg",   "window_size"),
        ("baseline_mm",    "window_size",   "ray_angle_deg", "baseline_mm",     "window_size"),
    ]

    for row_p, col_p, margin_p, row_label, col_label in pairs:
        pivot = (
            df.groupby([row_p, col_p])["n_solved"]
            .mean()
            .unstack(col_p)
        )
        slug  = f"{row_p[:3]}_vs_{col_p[:3]}"
        fname = f"heatmap_{slug}_nsolved.png"

        fig, ax = plt.subplots(figsize=FIGSIZE_HEAT)
        _heatmap(
            ax, pivot.values,
            row_labels=pivot.index.tolist(),
            col_labels=pivot.columns.tolist(),
            title=(
                f"Mean solved-frame count\n"
                f"({row_label} vs {col_label},  marginalized over {margin_p})"
            ),
            cmap="Greens",
            fmt=".1f",
        )
        ax.set_ylabel(row_label, fontsize=10)
        ax.set_xlabel(col_label, fontsize=10)
        fig.tight_layout()
        _save(fig, fname)


# ---------------------------------------------------------------------------
# Plot 5 — Best / worst summary dashboard
# ---------------------------------------------------------------------------

def plot5_best_worst(df: pd.DataFrame) -> None:
    print("\n[Plot 5] Best / worst summary dashboard")

    # Combined error: sum of mean X and mean Z errors (ignoring NaN rows)
    valid = df.dropna(subset=["x_err_mean", "z_err_mean"]).copy()
    valid["combined_err"] = valid["x_err_mean"] + valid["z_err_mean"]
    valid = valid.sort_values("combined_err")

    top5  = valid.head(5).reset_index(drop=True)
    bot5  = valid.tail(5).sort_values("combined_err", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=FIGSIZE_DASH)
    gs  = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35)

    ax_top_tbl  = fig.add_subplot(gs[0, 0])
    ax_bot_tbl  = fig.add_subplot(gs[1, 0])
    ax_bar      = fig.add_subplot(gs[:, 1])

    # ── Helper: render a dataframe as a table ─────────────────────────────
    def _table(ax: plt.Axes, data: pd.DataFrame, title: str, colour: str) -> None:
        ax.axis("off")
        
        # Determine which columns to display based on what's available
        core_cols = ["ray_angle_deg", "baseline_mm", "window_size"]
        frame_cols = []
        if "n_total_frames" in data.columns:
            frame_cols.append("n_total_frames")
        if "n_filtered_frames" in data.columns:
            frame_cols.append("n_filtered_frames")
        error_cols = ["n_solved", "x_err_mean", "z_err_mean", "combined_err"]
        
        cols = core_cols + frame_cols + error_cols
        cell_text = []
        for _, row in data[cols].iterrows():
            row_text = [
                f"{row['ray_angle_deg']:.0f}",
                f"{row['baseline_mm']:.0f}",
                f"{row['window_size']:.0f}",
            ]
            if "n_total_frames" in data.columns:
                row_text.append(f"{row['n_total_frames']:.0f}")
            if "n_filtered_frames" in data.columns:
                row_text.append(f"{row['n_filtered_frames']:.0f}")
            row_text.extend([
                f"{row['n_solved']:.0f}",
                f"{row['x_err_mean']:.2f}",
                f"{row['z_err_mean']:.2f}",
                f"{row['combined_err']:.2f}",
            ])
            cell_text.append(row_text)
        
        col_labels = ["ray°", "base mm", "win"]
        if "n_total_frames" in data.columns:
            col_labels.append("n_total")
        if "n_filtered_frames" in data.columns:
            col_labels.append("n_filt")
        col_labels.extend(["n_solved", "X̄ err", "Z̄ err", "combined"])
        
        tbl = ax.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7)
        tbl.scale(1, 1.3)
        ax.set_title(title, fontsize=10, fontweight="bold", pad=4, color=colour)

    _table(ax_top_tbl, top5,  "Top-5 best configurations",   "darkgreen")
    _table(ax_bot_tbl, bot5,  "Top-5 worst configurations",  "darkred")

    # ── Bar chart comparison ──────────────────────────────────────────────
    def _cfg_label(row: pd.Series) -> str:
        return (
            f"ray={row['ray_angle_deg']:.0f}\n"
            f"base={row['baseline_mm']:.0f}\n"
            f"win={row['window_size']:.0f}"
        )

    all10   = pd.concat([top5, bot5], ignore_index=True)
    labels  = [_cfg_label(r) for _, r in all10.iterrows()]
    x_errs  = all10["x_err_mean"].values
    z_errs  = all10["z_err_mean"].values

    pos     = np.arange(len(all10))
    w       = 0.38

    ax_bar.barh(pos + w / 2, x_errs, height=w, color=C_X, label="X error mean")
    ax_bar.barh(pos - w / 2, z_errs, height=w, color=C_Z, label="Z error mean")

    ax_bar.set_yticks(pos)
    ax_bar.set_yticklabels(labels, fontsize=7)
    ax_bar.axvline(0, color="black", linewidth=0.5)
    ax_bar.axhline(4.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.6,
                   label="best|worst boundary")

    # Shade best / worst regions
    n = len(all10)
    ax_bar.axhspan(-0.5, 4.5, alpha=0.05, color="green")
    ax_bar.axhspan(4.5, n - 0.5, alpha=0.05, color="red")

    ax_bar.set_xlabel("Absolute error (mm)", fontsize=10)
    ax_bar.set_title("Best vs Worst — X & Z error comparison", fontsize=10,
                      fontweight="bold")
    ax_bar.legend(fontsize=8, loc="lower right")
    ax_bar.grid(True, axis="x", alpha=0.3)
    ax_bar.invert_yaxis()

    fig.suptitle("Parameter Sensitivity Study — Best / Worst Summary",
                 fontsize=13, fontweight="bold")
    _save(fig, "best_worst_summary.png")


# ---------------------------------------------------------------------------
# Plot 6 — Error vs number of solved frames
# ---------------------------------------------------------------------------

def plot6_error_vs_nsolved(df_pf: pd.DataFrame) -> None:
    print("\n[Plot 6] Error vs number of solved frames")

    if df_pf.empty:
        print("  [SKIP] No per-frame data available")
        return

    # Aggregate n_solved per config
    config_nsolved = (
        df_pf.groupby(["ray_angle_deg", "baseline_mm", "window_size"])
        .size()
        .reset_index(name="n_solved")
    )

    # Per-config mean errors
    config_errors = (
        df_pf.groupby(["ray_angle_deg", "baseline_mm", "window_size"])
        .agg(
            x_err_mean=("err_x", "mean"),
            z_err_mean=("err_z", "mean"),
        )
        .reset_index()
    )

    config_data = config_nsolved.merge(config_errors, on=["ray_angle_deg", "baseline_mm", "window_size"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # X error vs n_solved
    axes[0].scatter(config_data["n_solved"], config_data["x_err_mean"],
                   alpha=0.6, s=50, color=C_X, edgecolors="black", linewidth=0.5)
    axes[0].set_xlabel("Number of solved frames", fontsize=11)
    axes[0].set_ylabel("Mean X error (mm)", fontsize=11)
    axes[0].set_title("X Error vs Solved-Frame Count", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)

    # Z error vs n_solved
    axes[1].scatter(config_data["n_solved"], config_data["z_err_mean"],
                   alpha=0.6, s=50, color=C_Z, edgecolors="black", linewidth=0.5)
    axes[1].set_xlabel("Number of solved frames", fontsize=11)
    axes[1].set_ylabel("Mean Z error (mm)", fontsize=11)
    axes[1].set_title("Z Error vs Solved-Frame Count", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Verification: Low errors are NOT caused by few frames", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "plot6_error_vs_nsolved.png")


# ---------------------------------------------------------------------------
# Plot 7 — Reprojection error by parameter
# ---------------------------------------------------------------------------

def plot7_reproj_error(df: pd.DataFrame) -> None:
    print("\n[Plot 7] Reprojection error (solver metric)")

    # Check if we have reprojection error data
    if "reproj_err_mean" not in df.columns or df["reproj_err_mean"].isna().all():
        print("  [SKIP] No reprojection error data available")
        return

    specs = [
        ("ray_angle_deg",  "min_ray_angle_deg (°)",   "reproj_vs_ray_angle.png"),
        ("baseline_mm",    "min_baseline_mm (mm)",     "reproj_vs_baseline.png"),
        ("window_size",    "sliding_window.max_size",  "reproj_vs_window_size.png"),
    ]

    for x_col, xlabel, fname in specs:
        grp = (
            df.groupby(x_col)
            .agg(
                reproj_mean=("reproj_err_mean", "mean"),
                reproj_std=("reproj_err_mean", "std"),
                reproj_min=("reproj_err_min", "min"),
                reproj_max=("reproj_err_max", "max"),
            )
            .reset_index()
            .sort_values(x_col)
        )

        xs = grp[x_col].values
        means = grp["reproj_mean"].values
        stds = grp["reproj_std"].fillna(0).values
        mins = grp["reproj_min"].values
        maxs = grp["reproj_max"].values

        fig, ax = plt.subplots(figsize=FIGSIZE_STD)

        # Min–max whiskers
        ax.errorbar(
            xs, means,
            yerr=[means - mins, maxs - means],
            fmt="none", ecolor="purple", alpha=0.3,
            capsize=3, linewidth=1, label="min–max",
        )
        # ±1σ bars
        ax.errorbar(
            xs, means,
            yerr=stds,
            fmt="o", color="purple", markersize=6,
            elinewidth=2, capsize=4, label="mean ±1σ",
        )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Reprojection error (angular, degrees)", fontsize=11)
        ax.set_title(
            f"Solver's optimization metric (reprojection error)\nvs {xlabel}",
            fontsize=11, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        _save(fig, fname)


# ---------------------------------------------------------------------------
# Plot 8 — Condition number by parameter
# ---------------------------------------------------------------------------

def plot8_condition_number(df: pd.DataFrame) -> None:
    print("\n[Plot 8] System matrix condition number (numerical stability)")

    # Check if we have condition number data
    if "cond_mean" not in df.columns or df["cond_mean"].isna().all():
        print("  [SKIP] No condition number data available")
        print("        (Requires newer pipeline with condition number logging)")
        return

    specs = [
        ("ray_angle_deg",  "min_ray_angle_deg (°)",   "cond_vs_ray_angle.png"),
        ("baseline_mm",    "min_baseline_mm (mm)",     "cond_vs_baseline.png"),
    ]

    for x_col, xlabel, fname in specs:
        grp = (
            df.groupby(x_col)
            .agg(
                cond_mean=("cond_mean", "mean"),
                cond_std=("cond_std", "std"),
                cond_min=("cond_min", "min"),
                cond_max=("cond_max", "max"),
            )
            .reset_index()
            .sort_values(x_col)
        )

        xs = grp[x_col].values
        means = grp["cond_mean"].values
        stds = grp["cond_std"].fillna(0).values
        mins = grp["cond_min"].values
        maxs = grp["cond_max"].values

        fig, ax = plt.subplots(figsize=FIGSIZE_STD)

        # Min–max whiskers (log scale)
        ax.errorbar(
            xs, means,
            yerr=[means - mins, maxs - means],
            fmt="none", ecolor="orange", alpha=0.3,
            capsize=3, linewidth=1, label="min–max",
        )
        # ±1σ bars
        ax.errorbar(
            xs, means,
            yerr=stds,
            fmt="o", color="orange", markersize=6,
            elinewidth=2, capsize=4, label="mean ±1σ",
        )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Condition number, cond(A)", fontsize=11)
        ax.set_yscale("log")
        ax.set_title(
            f"System matrix conditioning (numerical stability)\nvs {xlabel}\n"
            f"Large cond(A) = ill-conditioned → error spikes",
            fontsize=10, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")
        fig.tight_layout()
        _save(fig, fname)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    df, df_pf = _load()

    plot1_opat(df)
    plot2_errorbars(df)
    plot3_heatmaps(df)
    plot4_nsolved_heatmaps(df)
    plot5_best_worst(df)
    plot6_error_vs_nsolved(df_pf)
    plot7_reproj_error(df)
    plot8_condition_number(df)

    print(f"\nAll plots written to {PLOT_DIR.relative_to(_ROOT)}/")


if __name__ == "__main__":
    main()
