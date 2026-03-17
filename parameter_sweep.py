#!/usr/bin/env python3
"""
Parameter Sweep — Find Optimal X-Constant Trajectory
=====================================================

The object moves only in Y,Z w.r.t. the robot base, so the reconstructed
X-position should be nearly constant.  This script sweeps pipeline parameters
to find combinations that produce the flattest X-trajectory (lowest X std-dev).

Parameter grid
--------------
    solver.manual_order         1, 2, 3, 4, 5, 6
    sliding_window.min_size     5, 10, 15
    sliding_window.max_size     15, 20, 30, 40
    filtering.min_ray_angle_deg 0.5, 1.0, 2.0, 3.0, 5.0
    filtering.min_baseline_mm   2.0, 5.0, 10.0
    solver.lambda_penalty       0.0, 0.01, 0.05, 0.1

Invalid combinations (min_size > max_size) are pruned automatically.

Outputs
-------
    results/parameter_sweep/sweep_results.csv
        One row per (dataset, param-combo), sorted by x_std ascending.
    Console: top-20 combos, best combo per dataset.

Usage
-----
    # Full sweep (all datasets)
    python3 parameter_sweep.py

    # Quick sweep (~100 combos, single dataset)
    python3 parameter_sweep.py --quick

    # Specify datasets
    python3 parameter_sweep.py --datasets datasets/converted/T7_155924_linear_ovr3_acc_no.csv

    # Parallel workers
    python3 parameter_sweep.py --workers 4

    # Dry run (1 combo, 1 dataset)
    python3 parameter_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
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
OUT_DIR      = _ROOT / "results" / "parameter_sweep"

# Default datasets to sweep over (linear trajectories — X should be constant)
DEFAULT_DATASETS: List[str] = [
    str(_ROOT / "datasets" / "converted" / "T4_143521_sinusoidal_ovr1.csv"),
    str(_ROOT / "datasets" / "converted" / "T4_143646_sinusoidal_ovr3.csv"),
]

# datasets/converted/T4_143521_sinusoidal_ovr1.csv
# datasets/converted/T4_143646_sinusoidal_ovr3.csv

# ---------------------------------------------------------------------------
# Full parameter grid
# ---------------------------------------------------------------------------
MANUAL_ORDERS:    List[int]   = [1, 2, 3, 4, 5, 6]
MIN_SIZES:        List[int]   = [5, 10, 15]
MAX_SIZES:        List[int]   = [15, 20, 30, 40]
RAY_ANGLES:       List[float] = [0.5, 1.0, 2.0, 3.0, 5.0]
BASELINES:        List[float] = [2.0, 5.0, 10.0]
LAMBDA_PENALTIES: List[float] = [0.0, 0.01, 0.05, 0.1]

# Quick grid (smaller, for fast iteration)
QUICK_ORDERS:     List[int]   = [1, 2, 3, 4]
QUICK_MIN_SIZES:  List[int]   = [5, 10]
QUICK_MAX_SIZES:  List[int]   = [20, 40]
QUICK_RAY_ANGLES: List[float] = [1.0, 3.0]
QUICK_BASELINES:  List[float] = [2.0, 10.0]
QUICK_LAMBDAS:    List[float] = [0.0, 0.05]


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
ParamCombo = Tuple[int, int, int, float, float, float]  # order, min, max, ray, base, lam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_base_cfg() -> dict:
    with open(DEFAULT_YAML) as fh:
        return yaml.safe_load(fh)


def _build_grid(quick: bool) -> List[ParamCombo]:
    """Build the parameter grid, pruning invalid min_size > max_size combos."""
    if quick:
        orders   = QUICK_ORDERS
        mins     = QUICK_MIN_SIZES
        maxs     = QUICK_MAX_SIZES
        rays     = QUICK_RAY_ANGLES
        bases    = QUICK_BASELINES
        lambdas  = QUICK_LAMBDAS
    else:
        orders   = MANUAL_ORDERS
        mins     = MIN_SIZES
        maxs     = MAX_SIZES
        rays     = RAY_ANGLES
        bases    = BASELINES
        lambdas  = LAMBDA_PENALTIES

    grid: List[ParamCombo] = []
    for order, mn, mx, ray, base, lam in product(orders, mins, maxs, rays, bases, lambdas):
        if mn > mx:
            continue  # invalid: min_size > max_size
        grid.append((order, mn, mx, ray, base, lam))
    return grid


def _write_temp_cfg(
    combo: ParamCombo,
    scratch_dir: str,
) -> str:
    """Write an overridden YAML config and return its path."""
    order, min_sz, max_sz, ray_angle, baseline, lam = combo

    cfg = _load_base_cfg()
    cfg["solver"]["manual_order"]                = int(order)
    cfg["solver"]["lambda_penalty"]              = float(lam)
    cfg["solver"]["bundle_adjustment_interval"]  = 0          # disable BA for speed
    cfg["sliding_window"]["min_size"]            = int(min_sz)
    cfg["sliding_window"]["max_size"]            = int(max_sz)
    cfg["filtering"]["min_ray_angle_deg"]        = float(ray_angle)
    cfg["filtering"]["min_baseline_mm"]          = float(baseline)
    cfg["plotting"]                              = {"enabled": False}
    cfg["logging"]["output_dir"]                 = scratch_dir

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(OUT_DIR),
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _agg(vals: List[float]) -> Dict[str, float]:
    """Return mean/std/min/max or NaN for an empty list."""
    if not vals:
        return {"mean": float("nan"), "std": float("nan"),
                "min":  float("nan"), "max": float("nan")}
    a = np.array(vals, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std()),
            "min":  float(a.min()),  "max": float(a.max())}


# ---------------------------------------------------------------------------
# Single-config runner  (module-level for picklability)
# ---------------------------------------------------------------------------

def _run_one(args: Tuple[ParamCombo, str]) -> dict:
    """Run pipeline for one (param-combo, dataset) pair.

    Returns a summary dict with X-flatness metrics.
    """
    combo, csv_path = args
    order, min_sz, max_sz, ray_angle, baseline, lam = combo
    dataset_name = Path(csv_path).stem

    scratch = tempfile.mkdtemp(dir=str(OUT_DIR))
    cfg_path = _write_temp_cfg(combo, scratch)

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

    # ── Collect X-positions and metrics from solved frames ──
    x_positions:  List[float] = []
    reproj_errs:  List[float] = []

    for m in measurements:
        if m.position is None:
            continue
        x_positions.append(float(m.position[0]))
        if m.reprojection_error is not None:
            reproj_errs.append(float(m.reprojection_error))

    n_solved = len(x_positions)

    ax  = _agg(x_positions)
    ar  = _agg(reproj_errs)

    # X-flatness score: std-dev of X positions (lower = flatter = better)
    x_std   = ax["std"]
    x_range = ax["max"] - ax["min"] if n_solved > 0 else float("nan")

    return {
        "dataset":              dataset_name,
        "manual_order":         order,
        "min_size":             min_sz,
        "max_size":             max_sz,
        "min_ray_angle_deg":    ray_angle,
        "min_baseline_mm":      baseline,
        "lambda_penalty":       lam,
        "n_solved":             n_solved,
        "x_std":                x_std,
        "x_mean":               ax["mean"],
        "x_min":                ax["min"],
        "x_max":                ax["max"],
        "x_range":              x_range,
        "reproj_err_mean":      ar["mean"],
        "reproj_err_std":       ar["std"],
        "reproj_err_min":       ar["min"],
        "reproj_err_max":       ar["max"],
    }


# ---------------------------------------------------------------------------
# Sweep orchestrator
# ---------------------------------------------------------------------------

def run_sweep(
    datasets: List[str],
    quick: bool = False,
    dry_run: bool = False,
    workers: int = 1,
) -> pd.DataFrame:
    """Execute the parameter sweep and persist results."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = _build_grid(quick=quick)

    # Build (combo, dataset) run list
    all_args: List[Tuple[ParamCombo, str]] = [
        (combo, ds) for combo in grid for ds in datasets
    ]

    if dry_run:
        all_args = all_args[:1]

    total = len(all_args)
    print(f"Parameter sweep: {len(grid)} parameter combos × {len(datasets)} dataset(s)")
    print(f"  Total runs        : {total}")
    print(f"  manual_order      : {QUICK_ORDERS if quick else MANUAL_ORDERS}")
    print(f"  min_size          : {QUICK_MIN_SIZES if quick else MIN_SIZES}")
    print(f"  max_size          : {QUICK_MAX_SIZES if quick else MAX_SIZES}")
    print(f"  min_ray_angle_deg : {QUICK_RAY_ANGLES if quick else RAY_ANGLES}")
    print(f"  min_baseline_mm   : {QUICK_BASELINES if quick else BASELINES}")
    print(f"  lambda_penalty    : {QUICK_LAMBDAS if quick else LAMBDA_PENALTIES}")
    print(f"  Workers           : {workers}")
    print()

    summary_rows: List[dict] = []
    t_start = time.time()

    if workers > 1:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, summary in enumerate(pool.imap(_run_one, all_args), start=1):
                summary_rows.append(summary)
                _print_progress(i, total, summary, t_start)
    else:
        for i, args in enumerate(all_args, start=1):
            summary = _run_one(args)
            summary_rows.append(summary)
            _print_progress(i, total, summary, t_start)

    # ── Build DataFrame and sort by X flatness ──
    df = pd.DataFrame(summary_rows)
    col_order = [
        "dataset",
        "manual_order", "min_size", "max_size",
        "min_ray_angle_deg", "min_baseline_mm", "lambda_penalty",
        "n_solved", "x_std", "x_mean", "x_min", "x_max", "x_range",
        "reproj_err_mean", "reproj_err_std", "reproj_err_min", "reproj_err_max",
    ]
    df = df[col_order].sort_values("x_std", ascending=True, na_position="last")

    # ── Save ──
    results_path = OUT_DIR / "sweep_results.csv"
    df.to_csv(results_path, index=False)

    elapsed_total = time.time() - t_start
    print(f"\n{'=' * 72}")
    print(f"Sweep complete in {elapsed_total:.1f}s  ({total} runs)")
    print(f"Results saved to: {results_path}")

    # ── Print top-20 ──
    _print_top_results(df)

    # ── Print best per dataset ──
    _print_best_per_dataset(df)

    return df


def _print_progress(i: int, total: int, summary: dict, t_start: float) -> None:
    elapsed = time.time() - t_start
    rate = i / elapsed if elapsed > 0 else 0
    eta = (total - i) / rate if rate > 0 else float("inf")
    print(
        f"  [{i:>5}/{total}] "
        f"ds={summary['dataset'][:20]:<20s}  "
        f"ord={summary['manual_order']}  "
        f"win=[{summary['min_size']},{summary['max_size']}]  "
        f"ray={summary['min_ray_angle_deg']:.1f}  "
        f"base={summary['min_baseline_mm']:.0f}  "
        f"λ={summary['lambda_penalty']:.2f}  "
        f"solved={summary['n_solved']:>3}  "
        f"x_std={summary['x_std']:>8.3f}  "
        f"ETA {eta:>5.0f}s",
        flush=True,
    )


def _print_top_results(df: pd.DataFrame, n: int = 20) -> None:
    """Print the top-n parameter combos ranked by lowest X std-dev."""
    # Filter to rows with at least some solved frames
    valid = df[df["n_solved"] > 0].head(n)
    if valid.empty:
        print("\nNo solved frames in any configuration!")
        return

    print(f"\n{'=' * 72}")
    print(f"TOP-{n} PARAMETER SETS (lowest X std-dev = flattest X trajectory)")
    print(f"{'=' * 72}")
    display_cols = [
        "dataset", "manual_order", "min_size", "max_size",
        "min_ray_angle_deg", "min_baseline_mm", "lambda_penalty",
        "n_solved", "x_std", "x_range", "reproj_err_mean",
    ]
    print(valid[display_cols].to_string(index=False))


def _print_best_per_dataset(df: pd.DataFrame) -> None:
    """Print the single best combo for each dataset."""
    valid = df[df["n_solved"] > 0]
    if valid.empty:
        return

    print(f"\n{'=' * 72}")
    print("BEST PARAMETERS PER DATASET")
    print(f"{'=' * 72}")

    for ds_name, group in valid.groupby("dataset"):
        best = group.iloc[0]  # already sorted by x_std
        print(f"\n  Dataset: {ds_name}")
        print(f"    manual_order      : {int(best['manual_order'])}")
        print(f"    min_size          : {int(best['min_size'])}")
        print(f"    max_size          : {int(best['max_size'])}")
        print(f"    min_ray_angle_deg : {best['min_ray_angle_deg']}")
        print(f"    min_baseline_mm   : {best['min_baseline_mm']}")
        print(f"    lambda_penalty    : {best['lambda_penalty']}")
        print(f"    ──────────────────────────────────")
        print(f"    n_solved          : {int(best['n_solved'])}")
        print(f"    X std-dev (mm)    : {best['x_std']:.4f}")
        print(f"    X range (mm)      : {best['x_range']:.4f}")
        print(f"    X mean (mm)       : {best['x_mean']:.4f}")
        print(f"    reproj_err mean   : {best['reproj_err_mean']:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sweep to find optimal X-constant trajectory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a smaller parameter grid (~100 combos) for fast iteration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single configuration as a smoke test.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Number of parallel worker processes (default: 1).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Paths to dataset CSV files. Defaults to two linear datasets.",
    )
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS

    # Validate dataset paths
    for ds in datasets:
        if not Path(ds).exists():
            print(f"ERROR: dataset not found: {ds}", file=sys.stderr)
            sys.exit(1)

    run_sweep(
        datasets=datasets,
        quick=args.quick,
        dry_run=args.dry_run,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
