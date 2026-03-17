#!/usr/bin/env python3
"""
Parameter Sensitivity Study — Sweep Runner
==========================================

Phase 1 — OPAT  (one-parameter-at-a-time, 16+16+6 = 38 unique configs)
Phase 2 — Full Factorial  (16 × 16 × 6 = 1 536 configs)

OPAT configs are a strict subset of the factorial grid, so we run the full
1 536-combination sweep and annotate each row with OPAT membership flags.

Outputs
-------
results/param_sweep/sweep_results.csv
    One row per (ray_angle, baseline, window) configuration:
    aggregated mean / std / min / max X and Z errors + n_solved.

results/param_sweep/sweep_per_frame.csv
    One row per solved measurement per configuration.

Usage
-----
    # Smoke test (single run)
    python3 param_sweep.py --dry-run

    # Full sweep, sequential
    python3 param_sweep.py

    # Full sweep, parallel  (N = number of worker processes)
    python3 param_sweep.py --workers 6
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
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Project root — ensures trajectory_tracking package is importable
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from trajectory_tracking.replay.replay_dataset import replay_csv  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_YAML = _ROOT / "trajectory_tracking" / "configs" / "default.yaml"
# CSV_PATH     = str(_ROOT / "datasets" / "converted" /"T8_160240_sinusoidal_ovr1_acc_no.csv")
OUT_DIR      = _ROOT / "results" / "param_sweep_sin_1"
CSV_PATH     = str(_ROOT / "datasets" / "143521_ovr1_sinusoidal_kf_interp1500.csv")


# datasets/converted/T7_155924_linear_ovr3_acc_no.csv
# Ground truth position (X and Z are constant; Y changes)
GT_X: float = 1034.74927   # mm
GT_Z: float =  664.122498  # mm
# GT_Z: float =  580  # mm

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------
RAY_ANGLES:   List[float] = list(np.arange(0, 5, 0.1))        # 0 – 15  (16 levels)
BASELINES:    List[float] = list(np.arange(0, 10, 0.1))        # 0 – 15  (16 levels)
WINDOW_SIZES: List[int] = [50] #          (6  levels)

# OPAT membership sets (used for annotation only — no extra runs)
_OPAT_RAY_SET:    Set[Tuple[float, float, int]] = {(r, 9, 50) for r in RAY_ANGLES}
_OPAT_BASE_SET:   Set[Tuple[float, float, int]] = {(3, b, 50) for b in BASELINES}
_OPAT_WINDOW_SET: Set[Tuple[float, float, int]] = {(3, 9, w)  for w in WINDOW_SIZES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_base_cfg() -> dict:
    with open(DEFAULT_YAML) as fh:
        return yaml.safe_load(fh)


def _write_temp_cfg(
    ray_angle: float,
    baseline: float,
    window_size: int,
    scratch_dir: str,
) -> str:
    """Write an overridden YAML config to a NamedTemporaryFile and return its path."""
    cfg = _load_base_cfg()
    cfg["filtering"]["min_ray_angle_deg"]           = float(ray_angle)
    cfg["filtering"]["min_baseline_mm"]             = float(baseline)
    cfg["sliding_window"]["max_size"]               = int(window_size)
    cfg["solver"]["bundle_adjustment_interval"]     = 0       # BA disabled
    cfg["plotting"]                                 = {"enabled": False}
    cfg["logging"]["output_dir"]                    = scratch_dir

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=str(OUT_DIR)
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
# Single-config runner  (must be a module-level function to be picklable)
# ---------------------------------------------------------------------------

def _run_one(args: Tuple[int, int, int]) -> Tuple[List[dict], dict]:
    """Run pipeline for one (ray_angle, baseline, window_size) combination.

    Returns
    -------
    per_frame_rows : list[dict]
        One dict per solved measurement.
    summary_row : dict
        Aggregated statistics across all solved measurements.
    """
    ray_angle, baseline, window_size = args

    scratch = tempfile.mkdtemp(dir=str(OUT_DIR))
    cfg_path = _write_temp_cfg(ray_angle, baseline, window_size, scratch)

    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            measurements = replay_csv(CSV_PATH, cfg_path)
    finally:
        try:
            os.unlink(cfg_path)
        except OSError:
            pass
        shutil.rmtree(scratch, ignore_errors=True)

    solved = [m for m in measurements if m.position is not None]

    per_frame: List[dict] = []
    x_errs:    List[float] = []
    z_errs:    List[float] = []
    reproj_errs: List[float] = []
    cond_nums: List[float] = []

    for m in measurements:
        if m.position is None:
            continue
        ex = abs(float(m.position[0]) - GT_X)
        ez = abs(float(m.position[2]) - GT_Z)
        x_errs.append(ex)
        z_errs.append(ez)
        
        # Collect reprojection error if available
        if m.reprojection_error is not None:
            reproj_errs.append(float(m.reprojection_error))
        
        per_frame.append({
            "ray_angle_deg": ray_angle,
            "baseline_mm":   baseline,
            "window_size":   window_size,
            "frame_id":      m.frame_id,
            "pos_x":         float(m.position[0]),
            "pos_z":         float(m.position[2]),
            "err_x":         ex,
            "err_z":         ez,
            "reprojection_error": float(m.reprojection_error) if m.reprojection_error is not None else None,
        })

    # Try to read condition numbers from logged error_metrics.csv
    error_metrics_path = Path(scratch) / "error_metrics.csv"
    if error_metrics_path.exists():
        try:
            df_metrics = pd.read_csv(error_metrics_path)
            if "condition_number" in df_metrics.columns:
                cond_nums = df_metrics["condition_number"].dropna().tolist()
        except Exception:
            pass

    ax = _agg(x_errs)
    az = _agg(z_errs)
    ar = _agg(reproj_errs)
    ac = _agg(cond_nums)

    key = (ray_angle, baseline, window_size)
    summary: dict = {
        "ray_angle_deg": ray_angle,
        "baseline_mm":   baseline,
        "window_size":   window_size,
        "n_solved":      len(solved),
        "x_err_mean":    ax["mean"],
        "x_err_std":     ax["std"],
        "x_err_min":     ax["min"],
        "x_err_max":     ax["max"],
        "z_err_mean":    az["mean"],
        "z_err_std":     az["std"],
        "z_err_min":     az["min"],
        "z_err_max":     az["max"],
        "reproj_err_mean": ar["mean"],
        "reproj_err_std":  ar["std"],
        "reproj_err_min":  ar["min"],
        "reproj_err_max":  ar["max"],
        "cond_mean":     ac["mean"],
        "cond_std":      ac["std"],
        "cond_min":      ac["min"],
        "cond_max":      ac["max"],
        # OPAT membership flags
        "opat_ray":    key in _OPAT_RAY_SET,
        "opat_base":   key in _OPAT_BASE_SET,
        "opat_window": key in _OPAT_WINDOW_SET,
    }
    return per_frame, summary


# ---------------------------------------------------------------------------
# Sweep orchestrator
# ---------------------------------------------------------------------------

def run_sweep(dry_run: bool = False, workers: int = 1) -> None:
    """Execute the full parameter sweep and persist results to CSV."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build full factorial run list
    all_args: List[Tuple[int, int, int]] = [
        (r, b, w)
        for r, b, w in product(RAY_ANGLES, BASELINES, WINDOW_SIZES)
    ]

    if dry_run:
        all_args = all_args[:1]

    total = len(all_args)
    opat_count = sum(
        1 for (r, b, w) in all_args
        if (r, b, w) in _OPAT_RAY_SET
        or (r, b, w) in _OPAT_BASE_SET
        or (r, b, w) in _OPAT_WINDOW_SET
    )
    print(
        f"Parameter sweep: {total} total runs  "
        f"({opat_count} OPAT-annotated, {total - opat_count} factorial-only)"
    )
    print(f"  ray_angle_deg : {RAY_ANGLES[0]}–{RAY_ANGLES[-1]}  ({len(RAY_ANGLES)} levels)")
    print(f"  baseline_mm   : {BASELINES[0]}–{BASELINES[-1]}  ({len(BASELINES)} levels)")
    print(f"  window_size   : {WINDOW_SIZES}  ({len(WINDOW_SIZES)} levels)")
    print(f"  Workers       : {workers}")
    print()

    summary_rows:    List[dict] = []
    per_frame_rows:  List[dict] = []
    t_start = time.time()

    if workers > 1:
        # ── Parallel execution ────────────────────────────────────────────
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for i, (frames, summary) in enumerate(
                pool.imap(_run_one, all_args), start=1
            ):
                summary_rows.append(summary)
                per_frame_rows.extend(frames)

                elapsed = time.time() - t_start
                rate    = i / elapsed if elapsed > 0 else 0
                eta     = (total - i) / rate if rate > 0 else float("inf")
                print(
                    f"  [{i:>5}/{total}] "
                    f"ray={summary['ray_angle_deg']:>2}  "
                    f"base={summary['baseline_mm']:>2}  "
                    f"win={summary['window_size']:>3}  "
                    f"solved={summary['n_solved']:>3}  "
                    f"x̄={summary['x_err_mean']:>8.2f}  "
                    f"z̄={summary['z_err_mean']:>8.2f}  "
                    f"ETA {eta:>5.0f}s",
                    flush=True,
                )
    else:
        # ── Sequential execution ──────────────────────────────────────────
        for i, args in enumerate(all_args, start=1):
            frames, summary = _run_one(args)
            summary_rows.append(summary)
            per_frame_rows.extend(frames)

            elapsed = time.time() - t_start
            rate    = i / elapsed if elapsed > 0 else 0
            eta     = (total - i) / rate if rate > 0 else float("inf")
            print(
                f"  [{i:>5}/{total}] "
                f"ray={summary['ray_angle_deg']:>2}  "
                f"base={summary['baseline_mm']:>2}  "
                f"win={summary['window_size']:>3}  "
                f"solved={summary['n_solved']:>3}  "
                f"x̄={summary['x_err_mean']:>8.2f}  "
                f"z̄={summary['z_err_mean']:>8.2f}  "
                f"ETA {eta:>5.0f}s",
                flush=True,
            )

    # ── Persist results ───────────────────────────────────────────────────
    results_path   = OUT_DIR / "sweep_results.csv"
    per_frame_path = OUT_DIR / "sweep_per_frame.csv"

    df_results   = pd.DataFrame(summary_rows)
    df_per_frame = pd.DataFrame(per_frame_rows)

    # Consistent column order for sweep_results
    col_order = [
        "ray_angle_deg", "baseline_mm", "window_size",
        "n_solved",
        "x_err_mean", "x_err_std", "x_err_min", "x_err_max",
        "z_err_mean", "z_err_std", "z_err_min", "z_err_max",
        "reproj_err_mean", "reproj_err_std", "reproj_err_min", "reproj_err_max",
        "cond_mean", "cond_std", "cond_min", "cond_max",
        "opat_ray", "opat_base", "opat_window",
    ]
    df_results = df_results[col_order]

    df_results.to_csv(results_path,   index=False)
    df_per_frame.to_csv(per_frame_path, index=False)

    elapsed_total = time.time() - t_start
    print()
    print(f"Sweep complete in {elapsed_total:.1f}s")
    print(f"  {results_path}")
    print(f"  {per_frame_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter sensitivity sweep for the trajectory pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
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
        help=(
            "Number of parallel worker processes (default: 1). "
            "Set to 4–8 for a significant speed-up on multi-core machines."
        ),
    )
    args = parser.parse_args()
    run_sweep(dry_run=args.dry_run, workers=args.workers)


if __name__ == "__main__":
    main()
