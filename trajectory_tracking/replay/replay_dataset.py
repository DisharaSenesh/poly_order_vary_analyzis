"""
Offline replay of a CSV dataset through the full pipeline.

Usage
-----
    python -m trajectory_tracking.replay.replay_dataset datasets/example.csv

or from the project root:

    python replay/replay_dataset.py datasets/example.csv

The CSV must have columns::

    frame_id, time, u, v, X, Y, Z, A, B, C
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Ensure that the package root is importable when run as a script.
_THIS_DIR = Path(__file__).resolve().parent
_PKG_ROOT = _THIS_DIR.parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from trajectory_tracking.core.measurement import Measurement
from trajectory_tracking.sync.measurement_sync import MeasurementSync
from trajectory_tracking.pipeline_runner import PipelineRunner


def replay_csv(
    csv_path: str,
    config_path: str = "trajectory_tracking/configs/default.yaml",
) -> List[Measurement]:
    """Read a CSV dataset and process every row through the pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    list[Measurement]
        Processed measurements with trajectory estimates.
    """
    df = pd.read_csv(csv_path)
    required = {"frame_id", "time", "u", "v", "X", "Y", "Z", "A", "B", "C"}
    if not required.issubset(set(df.columns)):
        raise ValueError(
            f"CSV must contain columns {required}; got {set(df.columns)}"
        )

    has_gt = {"gt_x", "gt_y", "gt_z"}.issubset(set(df.columns))

    runner = PipelineRunner(config_path=config_path)
    sync = MeasurementSync()

    all_measurements: List[Measurement] = []

    for _, row in df.iterrows():
        gt_kwargs = {}
        if has_gt:
            gt_kwargs = dict(
                gt_x=float(row["gt_x"]),
                gt_y=float(row["gt_y"]),
                gt_z=float(row["gt_z"]),
            )
        m = sync.from_csv_row(
            frame_id=int(row["frame_id"]),
            timestamp=float(row["time"]),
            u=float(row["u"]),
            v=float(row["v"]),
            X=float(row["X"]),
            Y=float(row["Y"]),
            Z=float(row["Z"]),
            A=float(row["A"]),
            B=float(row["B"]),
            C=float(row["C"]),
            **gt_kwargs,
        )
        runner.process(m)
        all_measurements.append(m)

    runner.finalise(all_measurements)
    return all_measurements


# ── CLI entry point ─────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a CSV dataset through the trajectory pipeline."
    )
    parser.add_argument("csv", help="Path to the input CSV file.")
    parser.add_argument(
        "--config",
        default="trajectory_tracking/configs/default.yaml",
        help="Path to YAML config (default: trajectory_tracking/configs/default.yaml).",
    )
    args = parser.parse_args()

    measurements = replay_csv(args.csv, args.config)
    solved = [m for m in measurements if m.position is not None]
    print(f"\nProcessed {len(measurements)} measurements, solved {len(solved)}.")
    if solved:
        last = solved[-1]
        print(f"Last position estimate: {last.position}")
        print(f"Last velocity estimate: {last.velocity}")
        print(f"Chosen model: N={last.chosen_model}")


if __name__ == "__main__":
    main()
