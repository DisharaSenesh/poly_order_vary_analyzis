"""
main.py — Entry point for dynamic object trajectory reconstruction.

Supports two execution modes:

  **Online**  (``--mode online``)
      Reads from a live camera and KUKA robot controller.

  **Offline** (``--mode offline --dataset <path>``)
      Replays a CSV dataset through the pipeline.

# Offline replay
python3 trajectory_tracking/main.py --mode offline --dataset datasets/circle_detection_dataset_2_converted.csv

# Online (live camera + KUKA)
python3 trajectory_tracking/main.py --mode online --config trajectory_tracking/configs/default.yaml

Usage
-----
    python main.py --config configs/default.yaml
    python main.py --config configs/default.yaml --mode offline --dataset datasets/example.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure package root is on sys.path when run from inside the package.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))


def run_online(config_path: str) -> None:
    """Online mode — live camera + KUKA controller."""
    from trajectory_tracking.pipeline_runner import PipelineRunner
    from trajectory_tracking.sensors.aruco_read import AprilTagTracker
    from trajectory_tracking.sensors.RobotControl import KUKAControl
    from trajectory_tracking.sync.measurement_sync import MeasurementSync
    from trajectory_tracking.core.measurement import Measurement

    import yaml
    import numpy as np

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    runner = PipelineRunner(config_path=config_path)
    sync = MeasurementSync()

    # ── Sensors ─────────────────────────────────────────────────────
    tag_cfg = cfg.get("apriltag", {})
    cam_cfg = cfg.get("camera", {})
    tracker = AprilTagTracker(
        target_id=tag_cfg.get("target_id", 5),
        camera_index=cam_cfg.get("index", 0),
        show_video=True,
    )
    tracker.start()

    robot_cfg = cfg.get("robot", {})
    robot = KUKAControl(ip=robot_cfg["ip"], port=robot_cfg["port"])
    connected = robot.connect()
    if not connected:
        print("[main] Robot connection failed — aborting.")
        tracker.stop()
        return

    all_measurements: list[Measurement] = []

    print("\n── Online pipeline running (Ctrl+C to stop) ──\n")
    try:
        while True:
            centre = tracker.get_centre()
            if centre is None:
                time.sleep(0.01)
                continue

            pose = robot.read_pose()
            if pose is None:
                time.sleep(0.01)
                continue

            m = sync.from_online(pixel_uv=centre, robot_pose=pose)
            metrics = runner.process(m)
            all_measurements.append(m)

            if metrics is not None:
                print(
                    f"[frame {m.frame_id}] pos={m.position}  "
                    f"reproj={metrics.mean_reprojection_error:.4f}°  "
                    f"model=N{metrics.chosen_model}"
                )

            time.sleep(0.02)  # ~50 Hz max

    except KeyboardInterrupt:
        print("\n── Stopping ──")

    tracker.stop()
    robot.close()
    runner.finalise(all_measurements)


def run_offline(config_path: str, dataset_path: str) -> None:
    """Offline mode — replay CSV dataset."""
    from trajectory_tracking.replay.replay_dataset import replay_csv

    measurements = replay_csv(dataset_path, config_path)
    solved = [m for m in measurements if m.position is not None]
    print(f"\nProcessed {len(measurements)} rows, {len(solved)} solved.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic object trajectory reconstruction pipeline."
    )
    parser.add_argument(
        "--config",
        default="trajectory_tracking/configs/default.yaml",
        help="Path to YAML configuration file.",
    )
    parser.add_argument(
        "--mode",
        choices=["online", "offline"],
        default="offline",
        help="Execution mode (default: offline).",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to CSV dataset (required for offline mode).",
    )
    args = parser.parse_args()

    if args.mode == "offline":
        if args.dataset is None:
            parser.error("--dataset is required for offline mode.")
        run_offline(args.config, args.dataset)
    else:
        run_online(args.config)


if __name__ == "__main__":
    main()
