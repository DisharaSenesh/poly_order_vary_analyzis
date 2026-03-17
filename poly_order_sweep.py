#!/usr/bin/env python3
"""
Polynomial Order Analysis — Parameter Sweep
============================================

Measures how changing fixed polynomial order N (1–10) affects mean X-axis
error for three object motion types (linear, sinusoidal, circular).

For each (dataset, poly_order) combination the script sweeps
min_baseline_mm and min_ray_angle_deg while holding window_max=50
and min_window=3(N+1).  The primary metric is mean |est_x − gt_x|.

After the sweep, the script:
  1. Generates annotated heatmaps (PNG + PDF) for every (dataset, order).
  2. Selects the optimal (baseline, angle) per heatmap.
  3. Reruns the pipeline at those optimal settings and saves geometry plots.
  4. Writes a consolidated results table (CSV).
  5. Renders a final HTML/PDF report from the existing template.

Usage
-----
    # Full sweep (sequential)
    python3 poly_order_sweep.py

    # Parallel (N workers)
    python3 poly_order_sweep.py --workers 6

    # Smoke test (tiny grid)
    python3 poly_order_sweep.py --smoke

    # Resume interrupted sweep (skips existing rows in CSV)
    python3 poly_order_sweep.py --resume
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trajectory_tracking.replay.replay_dataset import replay_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_YAML = _ROOT / "trajectory_tracking" / "configs" / "default.yaml"
OUT_ROOT     = _ROOT / "results" / "poly_order_analysis"

DATASETS: List[Dict[str, str]] = [
    {"label": "linear",     "path": str(_ROOT / "datasets" / "T2_141237_linear_ovr1.csv")},
    {"label": "sinusoidal", "path": str(_ROOT / "datasets" / "T4_143521_sinusoidal_ovr1.csv")},
    {"label": "circular",   "path": str(_ROOT / "datasets" / "T5_144016_circular_ovr1.csv")},
]

GT_X_REFERENCE: float = 1034.74927  # mm — validation reference only

POLY_ORDERS:  List[int]   = list(range(1, 11))       # 1..10
BASELINES:    List[float] = [round(x, 1) for x in np.arange(0.5, 10.5, 0.5)]  # 0.5..10.0
RAY_ANGLES:   List[float] = [round(x, 1) for x in np.arange(0.5, 10.5, 0.5)]  # 0.5..10.0
WINDOW_MAX:   int = 50

# Smoke-test grid
SMOKE_ORDERS:    List[int]   = [1, 3]
SMOKE_BASELINES: List[float] = [1.0, 5.0]
SMOKE_ANGLES:    List[float] = [1.0, 5.0]


def min_window_for_order(n: int) -> int:
    """Compute minimum window size: 3(N+1)."""
    return 3 * (n + 1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_base_cfg() -> dict:
    with open(DEFAULT_YAML) as fh:
        return yaml.safe_load(fh)


def _write_temp_cfg(
    poly_order: int,
    baseline: float,
    ray_angle: float,
    scratch_dir: str,
    tmp_dir: str,
) -> str:
    """Write override YAML config and return path."""
    cfg = _load_base_cfg()
    cfg["solver"]["manual_order"]               = int(poly_order)
    cfg["solver"]["lambda_penalty"]             = 0.0
    cfg["solver"]["bundle_adjustment_interval"] = 0
    cfg["sliding_window"]["min_size"]           = min_window_for_order(poly_order)
    cfg["sliding_window"]["max_size"]           = WINDOW_MAX
    cfg["filtering"]["min_ray_angle_deg"]       = float(ray_angle)
    cfg["filtering"]["min_baseline_mm"]         = float(baseline)
    cfg["plotting"]                             = {"enabled": False}
    cfg["logging"]["output_dir"]                = scratch_dir

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=tmp_dir,
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _agg(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {"mean": float("nan"), "std": float("nan"),
                "min":  float("nan"), "max": float("nan")}
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min":  float(a.min()),  "max": float(a.max())}


# ---------------------------------------------------------------------------
# Single-config runner (module-level for pickling)
# ---------------------------------------------------------------------------
RunArgs = Tuple[str, str, int, float, float]  # label, csv_path, order, baseline, angle


def _run_one(args: RunArgs) -> Tuple[List[dict], dict]:
    """Run one (label, csv, order, baseline, angle) combo.

    Returns (per_frame_rows, summary_row).
    """
    label, csv_path, poly_order, baseline, ray_angle = args

    tmp_dir = str(OUT_ROOT / "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    scratch = tempfile.mkdtemp(dir=tmp_dir)
    cfg_path = _write_temp_cfg(poly_order, baseline, ray_angle, scratch, tmp_dir)

    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            measurements = replay_csv(csv_path, cfg_path)
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass
        shutil.rmtree(scratch, ignore_errors=True)

    # Collect per-frame errors using gt_x from CSV
    per_frame: List[dict] = []
    x_errs:    List[float] = []
    y_errs:    List[float] = []
    z_errs:    List[float] = []
    n_total = len(measurements)

    for m in measurements:
        if m.position is None or m.gt_position is None:
            continue
        ex = abs(float(m.position[0]) - float(m.gt_position[0]))
        ey = abs(float(m.position[1]) - float(m.gt_position[1]))
        ez = abs(float(m.position[2]) - float(m.gt_position[2]))
        x_errs.append(ex)
        y_errs.append(ey)
        z_errs.append(ez)

        per_frame.append({
            "dataset":         label,
            "poly_order":      poly_order,
            "min_baseline_mm": baseline,
            "min_ray_angle":   ray_angle,
            "frame_id":        m.frame_id,
            "timestamp":       m.timestamp,
            "est_x":           float(m.position[0]),
            "est_y":           float(m.position[1]),
            "est_z":           float(m.position[2]),
            "gt_x":            float(m.gt_position[0]),
            "gt_y":            float(m.gt_position[1]),
            "gt_z":            float(m.gt_position[2]),
            "err_x":           ex,
            "err_y":           ey,
            "err_z":           ez,
        })

    ax = _agg(x_errs)
    ay = _agg(y_errs)
    az = _agg(z_errs)
    n_solved = len(x_errs)

    summary: dict = {
        "dataset":            label,
        "poly_order":         poly_order,
        "min_window":         min_window_for_order(poly_order),
        "window_max":         WINDOW_MAX,
        "min_baseline_mm":    baseline,
        "min_ray_angle":      ray_angle,
        "n_total":            n_total,
        "n_solved":           n_solved,
        "solve_ratio":        n_solved / n_total if n_total > 0 else 0.0,
        "mean_x_error_mm":    ax["mean"],
        "std_x_error_mm":     ax["std"],
        "min_x_error_mm":     ax["min"],
        "max_x_error_mm":     ax["max"],
        "mean_y_error_mm":    ay["mean"],
        "mean_z_error_mm":    az["mean"],
    }
    return per_frame, summary


# ═══════════════════════════════════════════════════════════════════
#  SWEEP ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════
def run_sweep(
    smoke: bool = False,
    workers: int = 1,
    resume: bool = False,
) -> pd.DataFrame:
    """Execute the parameter sweep and persist results."""
    summaries_dir = OUT_ROOT / "summaries"
    per_frame_dir = OUT_ROOT / "per_frame"
    summaries_dir.mkdir(parents=True, exist_ok=True)
    per_frame_dir.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "tmp").mkdir(parents=True, exist_ok=True)

    orders    = SMOKE_ORDERS    if smoke else POLY_ORDERS
    baselines = SMOKE_BASELINES if smoke else BASELINES
    angles    = SMOKE_ANGLES    if smoke else RAY_ANGLES

    # Build full run list
    all_args: List[RunArgs] = []
    for ds in DATASETS:
        for order, base, angle in product(orders, baselines, angles):
            all_args.append((ds["label"], ds["path"], order, base, angle))

    # Resume support: skip already-completed combos
    summary_csv = summaries_dir / "sweep_summary.csv"
    done_keys = set()
    if resume and summary_csv.exists():
        df_done = pd.read_csv(summary_csv)
        for _, row in df_done.iterrows():
            done_keys.add((
                row["dataset"], int(row["poly_order"]),
                float(row["min_baseline_mm"]), float(row["min_ray_angle"]),
            ))
        print(f"Resume: {len(done_keys)} combos already completed, skipping.")
        all_args = [a for a in all_args if (a[0], a[2], a[3], a[4]) not in done_keys]

    total = len(all_args)
    print(f"Polynomial Order Sweep: {total} runs")
    print(f"  orders:    {orders}")
    print(f"  baselines: {baselines[0]}–{baselines[-1]} ({len(baselines)} levels)")
    print(f"  angles:    {angles[0]}–{angles[-1]} ({len(angles)} levels)")
    print(f"  datasets:  {[d['label'] for d in DATASETS]}")
    print(f"  workers:   {workers}")
    print()

    summary_rows: List[dict] = []
    per_frame_rows: List[dict] = []
    t_start = time.time()

    def _progress(i: int, summary: dict) -> None:
        elapsed = time.time() - t_start
        rate = i / elapsed if elapsed > 0 else 0
        eta = (total - i) / rate if rate > 0 else float("inf")
        print(
            f"  [{i:>5}/{total}] "
            f"{summary['dataset']:>11} N={summary['poly_order']:<2} "
            f"base={summary['min_baseline_mm']:>5.1f} "
            f"angle={summary['min_ray_angle']:>5.1f} "
            f"solved={summary['n_solved']:>4} "
            f"x̄_err={summary['mean_x_error_mm']:>9.3f} "
            f"ETA {eta:>5.0f}s",
            flush=True,
        )

    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, (frames, summary) in enumerate(
                pool.imap(_run_one, all_args), start=1
            ):
                summary_rows.append(summary)
                per_frame_rows.extend(frames)
                _progress(i, summary)
    else:
        for i, args in enumerate(all_args, start=1):
            frames, summary = _run_one(args)
            summary_rows.append(summary)
            per_frame_rows.extend(frames)
            _progress(i, summary)

    # Merge with any previous resume data
    df_summary = pd.DataFrame(summary_rows)
    if resume and summary_csv.exists():
        df_old = pd.read_csv(summary_csv)
        df_summary = pd.concat([df_old, df_summary], ignore_index=True)

    df_summary.to_csv(summary_csv, index=False)
    print(f"\n[sweep] Summary saved → {summary_csv}")

    df_per_frame = pd.DataFrame(per_frame_rows)
    pf_path = per_frame_dir / "sweep_per_frame.csv"
    if resume and pf_path.exists():
        df_old_pf = pd.read_csv(pf_path)
        df_per_frame = pd.concat([df_old_pf, df_per_frame], ignore_index=True)
    df_per_frame.to_csv(pf_path, index=False)
    print(f"[sweep] Per-frame saved → {pf_path}")

    # Cleanup tmp
    shutil.rmtree(OUT_ROOT / "tmp", ignore_errors=True)

    return df_summary


# ═══════════════════════════════════════════════════════════════════
#  HEATMAP GENERATOR
# ═══════════════════════════════════════════════════════════════════
def generate_heatmaps(df: pd.DataFrame) -> pd.DataFrame:
    """Generate annotated heatmaps for every (dataset, poly_order) pair.

    Returns a DataFrame of optimal cells (one per heatmap).
    """
    heatmap_dir = OUT_ROOT / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    optimal_rows: List[dict] = []

    for (label, order), grp in df.groupby(["dataset", "poly_order"]):
        pivot = grp.pivot_table(
            index="min_ray_angle",
            columns="min_baseline_mm",
            values="mean_x_error_mm",
            aggfunc="first",
        )
        pivot_solved = grp.pivot_table(
            index="min_ray_angle",
            columns="min_baseline_mm",
            values="n_solved",
            aggfunc="first",
        )

        fig, ax = plt.subplots(figsize=(14, 10))

        # Mask cells with no solved frames
        masked = np.ma.masked_invalid(pivot.values)

        # Use a perceptual light-to-dark colormap
        cmap = plt.cm.YlOrRd.copy()
        cmap.set_bad(color="lightgray", alpha=0.5)

        im = ax.imshow(
            masked,
            aspect="auto",
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
        )
        cbar = fig.colorbar(im, ax=ax, label="Mean X Error (mm)", shrink=0.8)

        # Axis labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns], rotation=45, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.1f}" for v in pivot.index], fontsize=7)
        ax.set_xlabel("Min Baseline (mm)")
        ax.set_ylabel("Min Ray Angle (°)")
        ax.set_title(
            f"Mean X Error — {label.capitalize()} Motion, Poly Order N={order}\n"
            f"(min_window={min_window_for_order(order)}, window_max={WINDOW_MAX})",
            fontsize=11,
        )

        # Annotate cells with values
        for i in range(masked.shape[0]):
            for j in range(masked.shape[1]):
                solved = pivot_solved.values[i, j]
                if np.isnan(pivot.values[i, j]) or solved == 0:
                    ax.text(j, i, "NA", ha="center", va="center",
                            fontsize=5, color="gray", fontstyle="italic")
                else:
                    val = pivot.values[i, j]
                    text_color = "white" if val > (masked.max() * 0.6) else "black"
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=5, color=text_color, fontweight="bold")

        plt.tight_layout()

        # Save PNG and PDF
        base_name = f"{label}_order_{order:02d}_heatmap"
        png_path = heatmap_dir / f"{base_name}.png"
        pdf_path = heatmap_dir / f"{base_name}.pdf"
        fig.savefig(str(png_path), dpi=200)
        fig.savefig(str(pdf_path))
        plt.close(fig)
        print(f"[heatmap] {png_path.name}")

        # Find optimal cell: lowest mean X error, then highest n_solved,
        # then lowest baseline, then lowest angle
        valid = grp[grp["n_solved"] > 0].dropna(subset=["mean_x_error_mm"])
        if valid.empty:
            optimal_rows.append({
                "dataset": label, "poly_order": order,
                "min_mean_error_mm": float("nan"),
                "opt_baseline_mm": float("nan"), "opt_ray_angle": float("nan"),
                "n_solved": 0, "n_total": 0, "solve_ratio": 0.0,
                "min_window": min_window_for_order(order),
                "heatmap_png": str(png_path), "heatmap_pdf": str(pdf_path),
            })
            continue

        valid_sorted = valid.sort_values(
            by=["mean_x_error_mm", "n_solved", "min_baseline_mm", "min_ray_angle"],
            ascending=[True, False, True, True],
        )
        best = valid_sorted.iloc[0]
        optimal_rows.append({
            "dataset":           label,
            "poly_order":        int(order),
            "min_mean_error_mm": float(best["mean_x_error_mm"]),
            "opt_baseline_mm":   float(best["min_baseline_mm"]),
            "opt_ray_angle":     float(best["min_ray_angle"]),
            "n_solved":          int(best["n_solved"]),
            "n_total":           int(best["n_total"]),
            "solve_ratio":       float(best["solve_ratio"]),
            "min_window":        min_window_for_order(int(order)),
            "heatmap_png":       str(png_path),
            "heatmap_pdf":       str(pdf_path),
        })

    df_optimal = pd.DataFrame(optimal_rows)
    opt_path = OUT_ROOT / "summaries" / "optimal_cells.csv"
    df_optimal.to_csv(opt_path, index=False)
    print(f"\n[heatmap] Optimal cells saved → {opt_path}")
    return df_optimal


# ═══════════════════════════════════════════════════════════════════
#  OPTIMAL RERUN + GEOMETRY PLOTS
# ═══════════════════════════════════════════════════════════════════
def rerun_optimal(df_optimal: pd.DataFrame) -> pd.DataFrame:
    """Rerun pipeline at optimal settings and save geometry plots."""
    from trajectory_tracking.plotting.plot_geometry import plot_camera_ray_geometry

    runs_dir = OUT_ROOT / "optimal_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    geometry_paths: List[dict] = []

    for _, row in df_optimal.iterrows():
        label = row["dataset"]
        order = int(row["poly_order"])

        if np.isnan(row["opt_baseline_mm"]):
            print(f"[rerun] Skipping {label} N={order} — no valid solution")
            geometry_paths.append({"dataset": label, "poly_order": order,
                                   "geometry_png": "", "geometry_pdf": ""})
            continue

        baseline = float(row["opt_baseline_mm"])
        angle    = float(row["opt_ray_angle"])

        # Find dataset CSV path
        csv_path = next(d["path"] for d in DATASETS if d["label"] == label)

        # Create a config for this run
        run_dir = runs_dir / f"{label}_order_{order:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cfg = _load_base_cfg()
        cfg["solver"]["manual_order"]               = order
        cfg["solver"]["lambda_penalty"]             = 0.0
        cfg["solver"]["bundle_adjustment_interval"] = 0
        cfg["sliding_window"]["min_size"]           = min_window_for_order(order)
        cfg["sliding_window"]["max_size"]           = WINDOW_MAX
        cfg["filtering"]["min_ray_angle_deg"]       = angle
        cfg["filtering"]["min_baseline_mm"]         = baseline
        cfg["plotting"]                             = {"enabled": False}
        cfg["logging"]["output_dir"]                = str(run_dir / "output")

        cfg_path = str(run_dir / "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        print(f"[rerun] {label} N={order} baseline={baseline} angle={angle} ...")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            measurements = replay_csv(csv_path, cfg_path)

        # Save geometry plot with GT overlay
        title = (
            f"Camera–Ray–Trajectory · {label.capitalize()} · N={order}\n"
            f"baseline={baseline:.1f}mm  angle={angle:.1f}°  "
            f"mean_err_x={row['min_mean_error_mm']:.2f}mm"
        )
        png_path = str(run_dir / f"{label}_order_{order:02d}_geometry.png")
        plot_camera_ray_geometry(
            measurements,
            output_path=png_path,
            title=title,
            save_pdf=True,
            figsize=(14, 10),
            dpi=200,
        )
        pdf_path = str(Path(png_path).with_suffix(".pdf"))
        geometry_paths.append({
            "dataset": label, "poly_order": order,
            "geometry_png": png_path, "geometry_pdf": pdf_path,
        })

    df_geo = pd.DataFrame(geometry_paths)
    df_optimal = df_optimal.merge(df_geo, on=["dataset", "poly_order"], how="left")
    return df_optimal


# ═══════════════════════════════════════════════════════════════════
#  REPORT RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════
def build_report_table(df_optimal: pd.DataFrame) -> pd.DataFrame:
    """Build the requested results table.

    Columns: poly_order | object_motion | minimum_mean_error |
             min_baseline | min_ray_angle | min_window | n_solved | n_total | solve_ratio
    """
    report = df_optimal[[
        "poly_order", "dataset", "min_mean_error_mm",
        "opt_baseline_mm", "opt_ray_angle",
        "min_window", "n_solved", "n_total", "solve_ratio",
    ]].copy()
    report.columns = [
        "poly_order", "object_motion", "minimum_mean_error",
        "min_baseline", "min_ray_angle",
        "min_window", "n_solved", "n_total", "solve_ratio",
    ]
    report = report.sort_values(["poly_order", "object_motion"]).reset_index(drop=True)

    report_path = OUT_ROOT / "summaries" / "report_results.csv"
    report.to_csv(report_path, index=False)
    print(f"[report] Results table saved → {report_path}")
    return report


# ═══════════════════════════════════════════════════════════════════
#  HTML REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════
def _img_to_base64(path: str) -> str:
    """Read an image file and return its base64 data URI."""
    if not path or not Path(path).exists():
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    suffix = Path(path).suffix.lstrip(".")
    mime = "image/png" if suffix == "png" else f"image/{suffix}"
    return f"data:{mime};base64,{data}"


def generate_html_report(
    df_optimal: pd.DataFrame,
    df_report: pd.DataFrame,
    df_summary: pd.DataFrame,
) -> str:
    """Generate a self-contained HTML report and save it."""
    report_dir = OUT_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)

    ds_labels = ["linear", "sinusoidal", "circular"]
    orders = sorted(df_optimal["poly_order"].unique())

    today = time.strftime("%Y-%m-%d")

    # Build per-dataset trend data: best error at each order
    trend_data = {}
    for label in ds_labels:
        sub = df_optimal[df_optimal["dataset"] == label].sort_values("poly_order")
        trend_data[label] = list(zip(sub["poly_order"].tolist(), sub["min_mean_error_mm"].tolist()))

    # ------- HTML Construction -------
    html_parts = []
    html_parts.append("""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Polynomial Order Analysis — Trajectory Reconstruction</title>
<style>
  :root {
    --bg: #f4f1eb; --surface: #faf8f3; --ink: #1a1a1a; --ink-dim: #555;
    --ink-faint: #999; --accent: #1a4f8a; --accent2: #c94f1a; --accent3: #1a8a4f;
    --grid: #d4cfc5; --cell-bg: #fff; --cell-border: #c8c2b8;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', sans-serif; background: var(--bg);
    color: var(--ink); padding: 30px 40px; min-height: 100vh;
  }
  .report-header {
    border-top: 3px solid var(--ink); border-bottom: 1px solid var(--ink);
    padding: 14px 0 10px; margin-bottom: 22px;
    display: flex; justify-content: space-between; align-items: flex-end;
  }
  .report-header h1 { font-size: 16px; font-weight: 600; letter-spacing: 0.03em; }
  .report-header p { font-size: 11px; color: var(--ink-dim); margin-top: 3px; }
  .header-tags { display: flex; flex-direction: column; align-items: flex-end; gap: 3px; }
  .param-tag {
    font-size: 9px; color: var(--ink-dim); border: 1px solid #aaa; padding: 2px 7px;
    border-radius: 2px; background: var(--surface);
  }
  .section-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-faint);
    margin: 22px 0 10px; display: flex; align-items: center; gap: 8px;
  }
  .section-label::after { content:''; flex:1; height:1px; background:var(--grid); }

  table.results { width: 100%; border-collapse: collapse; margin-bottom: 20px; font-size: 11px; }
  table.results th {
    background: var(--ink); color: #fff; padding: 7px 10px; text-align: left; font-weight: 600;
  }
  table.results td { padding: 6px 10px; border-bottom: 1px solid var(--cell-border); }
  table.results tr:hover td { background: #f0ede6; }
  .best-row td { background: #e8f5e8 !important; font-weight: 600; }

  .grid-container {
    display: grid; grid-template-columns: 88px repeat(3, 1fr); gap: 1px;
    border: 1px solid var(--cell-border); background: var(--cell-border); margin-bottom: 22px;
  }
  .grid-cell { background: var(--cell-bg); padding: 8px; }
  .col-hdr {
    background: var(--ink); color: #fff; padding: 8px 10px; text-align: center;
    font-size: 11px; font-weight: 600;
  }
  .col-hdr .sub { font-size: 9px; font-weight: 400; opacity: 0.65; display: block; margin-top: 2px; }
  .row-hdr {
    background: var(--surface); display: flex; flex-direction: column;
    align-items: center; justify-content: center; padding: 10px 4px; gap: 3px;
    border-right: 1px solid var(--cell-border);
  }
  .row-hdr .order-num { font-size: 16px; font-weight: 600; }
  .row-hdr .order-label { font-size: 7px; color: var(--ink-faint); text-transform: uppercase; }
  .corner { background: var(--ink); padding: 6px; display:flex;flex-direction:column;justify-content:flex-end; }
  .corner span { font-size: 7px; color: rgba(255,255,255,0.45); text-transform: uppercase; }

  .cell-img { width: 100%; max-height: 260px; object-fit: contain; border: 1px solid #ddd; margin-bottom: 4px; }
  .cell-caption {
    font-size: 8px; color: var(--ink-faint); text-align: center; margin-top: 2px;
  }

  .trend-section { display: flex; gap: 20px; margin-bottom: 22px; flex-wrap: wrap; }
  .trend-img { max-width: 600px; width: 100%; border: 1px solid #ddd; }
  .summary-box {
    flex: 1; min-width: 300px; padding: 14px; background: var(--surface);
    border: 1px solid var(--cell-border); border-radius: 3px;
  }
  .summary-box h3 { font-size: 12px; margin-bottom: 8px; }

  .report-footer {
    border-top: 1px solid var(--ink); padding-top: 8px; margin-top: 30px;
    display: flex; justify-content: space-between; font-size: 9px; color: var(--ink-faint);
  }
  @media print { body { padding: 15px; } .grid-container { break-inside: avoid; } }
</style>
</head>
<body>
""")

    # Header
    html_parts.append(f"""
<div class="report-header">
  <div>
    <h1>Polynomial Order Analysis &mdash; Trajectory Reconstruction</h1>
    <p>Effect of polynomial order (N=1&ndash;10) on mean X-axis error &middot; linear / sinusoidal / circular motion</p>
  </div>
  <div class="header-tags">
    <span class="param-tag">Max window size = {WINDOW_MAX}</span>
    <span class="param-tag">Baselines: {BASELINES[0]}&ndash;{BASELINES[-1]} mm</span>
    <span class="param-tag">Ray angles: {RAY_ANGLES[0]}&ndash;{RAY_ANGLES[-1]}&deg;</span>
    <span class="param-tag">Date: {today}</span>
  </div>
</div>
""")

    # ── Per-order results grid ──
    html_parts.append('<div class="section-label">Per-Polynomial-Order Results</div>')
    html_parts.append('<div class="grid-container">')
    html_parts.append('<div class="grid-cell corner"><span>Poly<br>Order</span></div>')
    for label in ds_labels:
        html_parts.append(
            f'<div class="grid-cell col-hdr">{label.capitalize()}'
            f'<span class="sub">{label} motion</span></div>'
        )

    for order in orders:
        # Row header
        html_parts.append(
            f'<div class="grid-cell row-hdr">'
            f'<div class="order-num">N={order}</div>'
            f'<div class="order-label">min_win={min_window_for_order(order)}</div></div>'
        )
        for label in ds_labels:
            row = df_optimal[
                (df_optimal["dataset"] == label) & (df_optimal["poly_order"] == order)
            ]
            if row.empty:
                html_parts.append('<div class="grid-cell">No data</div>')
                continue
            row = row.iloc[0]
            heatmap_b64 = _img_to_base64(row.get("heatmap_png", ""))
            geo_b64 = _img_to_base64(row.get("geometry_png", ""))
            err_str = f"{row['min_mean_error_mm']:.2f}" if not np.isnan(row["min_mean_error_mm"]) else "NA"

            cell_html = '<div class="grid-cell">'
            if heatmap_b64:
                cell_html += f'<img class="cell-img" src="{heatmap_b64}" alt="heatmap"/>'
            cell_html += f'<div class="cell-caption">best err={err_str}mm · '
            cell_html += f'base={row["opt_baseline_mm"]:.1f} · angle={row["opt_ray_angle"]:.1f}°</div>'
            if geo_b64:
                cell_html += f'<img class="cell-img" src="{geo_b64}" alt="geometry"/>'
            cell_html += '</div>'
            html_parts.append(cell_html)

    html_parts.append('</div><!-- end grid -->')

    # ── Error trend plot ──
    html_parts.append('<div class="section-label">Mean X Error vs Polynomial Order</div>')
    trend_png = _generate_trend_plot(trend_data)
    trend_b64 = _img_to_base64(str(trend_png))

    html_parts.append('<div class="trend-section">')
    if trend_b64:
        html_parts.append(f'<img class="trend-img" src="{trend_b64}" alt="trend"/>')

    # Best-per-dataset summary
    html_parts.append('<div class="summary-box"><h3>Optimal Order per Motion Type</h3>')
    html_parts.append('<table class="results"><thead><tr>'
                      '<th>Motion</th><th>Best N*</th><th>Min Error (mm)</th>'
                      '<th>Baseline</th><th>Ray Angle</th></tr></thead><tbody>')
    for label in ds_labels:
        sub = df_optimal[df_optimal["dataset"] == label]
        if sub.empty or sub["min_mean_error_mm"].isna().all():
            continue
        best = sub.loc[sub["min_mean_error_mm"].idxmin()]
        html_parts.append(
            f'<tr><td>{label.capitalize()}</td><td>N={int(best["poly_order"])}</td>'
            f'<td>{best["min_mean_error_mm"]:.2f}</td>'
            f'<td>{best["opt_baseline_mm"]:.1f} mm</td>'
            f'<td>{best["opt_ray_angle"]:.1f}°</td></tr>'
        )
    html_parts.append('</tbody></table></div></div>')

    # ── Full results table ──
    html_parts.append('<div class="section-label">Full Results Table</div>')
    html_parts.append('<table class="results"><thead><tr>')
    for col in ["Poly Order", "Object Motion", "Min Mean Error (mm)",
                "Min Baseline (mm)", "Min Ray Angle (°)",
                "Min Window", "Solved", "Total", "Solve Ratio"]:
        html_parts.append(f'<th>{col}</th>')
    html_parts.append('</tr></thead><tbody>')

    # Find best per dataset for highlighting
    best_per_ds = {}
    for label in ds_labels:
        sub = df_report[df_report["object_motion"] == label]
        if not sub.empty and not sub["minimum_mean_error"].isna().all():
            best_per_ds[label] = sub["minimum_mean_error"].min()

    for _, r in df_report.iterrows():
        is_best = (r["object_motion"] in best_per_ds and
                   not np.isnan(r["minimum_mean_error"]) and
                   abs(r["minimum_mean_error"] - best_per_ds[r["object_motion"]]) < 1e-6)
        cls = ' class="best-row"' if is_best else ''
        err_str = f'{r["minimum_mean_error"]:.2f}' if not np.isnan(r["minimum_mean_error"]) else "NA"
        html_parts.append(
            f'<tr{cls}><td>N={int(r["poly_order"])}</td>'
            f'<td>{r["object_motion"].capitalize()}</td>'
            f'<td>{err_str}</td>'
            f'<td>{r["min_baseline"]:.1f}</td>'
            f'<td>{r["min_ray_angle"]:.1f}</td>'
            f'<td>{int(r["min_window"])}</td>'
            f'<td>{int(r["n_solved"])}</td>'
            f'<td>{int(r["n_total"])}</td>'
            f'<td>{r["solve_ratio"]:.2%}</td></tr>'
        )

    html_parts.append('</tbody></table>')

    # Footer
    html_parts.append(f"""
<div class="report-footer">
  <span>Dynamic Object Trajectory Reconstruction &middot; Polynomial Order Analysis</span>
  <span>Generated: {today}</span>
</div>
</body></html>""")

    html_content = "\n".join(html_parts)
    html_path = report_dir / "polynomial_order_analysis_report.html"
    with open(html_path, "w") as f:
        f.write(html_content)
    print(f"[report] HTML report saved → {html_path}")

    # Try PDF conversion
    _try_pdf_export(html_path)

    return str(html_path)


def _generate_trend_plot(
    trend_data: Dict[str, List[Tuple[int, float]]],
) -> Path:
    """Generate a mean-error-vs-polynomial-order trend plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = {"linear": "#1a4f8a", "sinusoidal": "#c94f1a", "circular": "#1a8a4f"}
    markers = {"linear": "o", "sinusoidal": "s", "circular": "D"}

    for label, data in trend_data.items():
        orders_list = [d[0] for d in data]
        errors = [d[1] for d in data]
        ax.plot(orders_list, errors,
                color=colors.get(label, "gray"),
                marker=markers.get(label, "o"),
                linewidth=2, markersize=6,
                label=f"{label.capitalize()}")

        # Mark the minimum
        valid = [(o, e) for o, e in zip(orders_list, errors) if not np.isnan(e)]
        if valid:
            best_order, best_err = min(valid, key=lambda x: x[1])
            ax.axvline(best_order, color=colors.get(label, "gray"),
                       linestyle="--", alpha=0.4, linewidth=1)
            ax.annotate(f"N*={best_order}", (best_order, best_err),
                        textcoords="offset points", xytext=(8, 8),
                        fontsize=8, color=colors.get(label, "gray"))

    ax.set_xlabel("Polynomial Order")
    ax.set_ylabel("Best Mean X Error (mm)")
    ax.set_title("Best Achievable Mean X Error vs Polynomial Order")
    ax.set_xticks(range(1, 11))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    trend_path = OUT_ROOT / "report" / "trend_error_vs_order.png"
    trend_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(trend_path), dpi=200)
    fig.savefig(str(trend_path.with_suffix(".pdf")))
    plt.close(fig)
    print(f"[report] Trend plot saved → {trend_path}")
    return trend_path


def _try_pdf_export(html_path: Path) -> None:
    """Attempt PDF export using available tools."""
    pdf_path = html_path.with_suffix(".pdf")

    # Try weasyprint first
    try:
        from weasyprint import HTML as WeasyprintHTML
        WeasyprintHTML(filename=str(html_path)).write_pdf(str(pdf_path))
        print(f"[report] PDF report saved → {pdf_path}")
        return
    except ImportError:
        pass

    # Try wkhtmltopdf
    import subprocess
    try:
        subprocess.run(
            ["wkhtmltopdf", "--quiet", str(html_path), str(pdf_path)],
            check=True, capture_output=True, timeout=60,
        )
        print(f"[report] PDF report saved → {pdf_path}")
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    print("[report] PDF conversion skipped — install weasyprint or wkhtmltopdf for PDF output")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Polynomial Order Analysis — Parameter Sweep",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--smoke", action="store_true", help="Tiny grid for testing")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted sweep")
    parser.add_argument("--skip-sweep", action="store_true",
                        help="Skip sweep, regenerate plots/report from existing CSV")
    args = parser.parse_args()

    summary_csv = OUT_ROOT / "summaries" / "sweep_summary.csv"

    if args.skip_sweep and summary_csv.exists():
        print("[main] Loading existing sweep results...")
        df_summary = pd.read_csv(summary_csv)
    else:
        df_summary = run_sweep(smoke=args.smoke, workers=args.workers, resume=args.resume)

    print(f"\n{'='*72}")
    print(f"Sweep complete: {len(df_summary)} rows")
    print(f"{'='*72}\n")

    # Phase 6: Heatmaps
    print("── Phase 6: Generating heatmaps ──────────────────")
    df_optimal = generate_heatmaps(df_summary)

    # Phase 7: Optimal reruns
    print("\n── Phase 7: Rerunning optimal configurations ─────")
    df_optimal = rerun_optimal(df_optimal)

    # Phase 8: Results table
    print("\n── Phase 8: Building report table ────────────────")
    df_report = build_report_table(df_optimal)
    print(df_report.to_string(index=False))

    # Phase 9: HTML report
    print("\n── Phase 9: Generating HTML report ───────────────")
    generate_html_report(df_optimal, df_report, df_summary)

    print(f"\n{'='*72}")
    print(f"All outputs saved to: {OUT_ROOT}")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
