"""
Central pipeline runner.

Orchestrates every stage:

1. Geometry processing   → camera pose + viewing ray
2. Geometry filtering    → ray-angle gating
3. Sliding window        → buffer management
4. Trajectory solver     → polynomial fit + model selection
5. Bundle adjustment     → periodic polynomial refinement  (NEW)
6. Evaluation            → reprojection error, metrics
7. Logging               → asynchronous CSV writing
8. Plotting              → diagnostic & result figures  (at finalise)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from trajectory_tracking.core.measurement import Measurement
from trajectory_tracking.evaluation.error_metrics import WindowMetrics, compute_window_metrics
from trajectory_tracking.filtering.ray_angle_filter import RayAngleFilter
from trajectory_tracking.geometry.camera_pose import compute_camera_pose, get_hand_eye_transform
from trajectory_tracking.geometry.kuka_rotation import kuka_rotation_matrix
from trajectory_tracking.geometry.ray_builder import build_ray
from trajectory_tracking.geometry.reprojection import reprojection_error_angular
from trajectory_tracking.recording.async_recorder import AsyncRecorder
from trajectory_tracking.solver.model_selection import select_best_model, ModelSelectionResult
from trajectory_tracking.solver.sliding_window import SlidingWindow
from trajectory_tracking.utils.math_utils import intrinsics_matrix


class PipelineRunner:
    """Stateful pipeline that processes one ``Measurement`` at a time.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    """

    def __init__(self, config_path: str = "trajectory_tracking/configs/default.yaml") -> None:
        self.cfg = self._load_config(config_path)

        # ── pre-compute constants ───────────────────────────────────
        cam = self.cfg["camera"]
        self.K = intrinsics_matrix(cam["fx"], cam["fy"], cam["cx"], cam["cy"])
        self.dist_coeffs = np.array(cam["distortion"], dtype=np.float64)
        self.img_width: int = int(cam.get("width", 1280))
        self.img_height: int = int(cam.get("height", 960))

        he = self.cfg["hand_eye"]
        self.R_cam_tool, self.t_cam_tool = get_hand_eye_transform(he)

        # ── modules ─────────────────────────────────────────────────
        filt = self.cfg["filtering"]
        self.ray_filter = RayAngleFilter(
            min_angle_deg=filt["min_ray_angle_deg"],
            min_baseline_mm=filt.get("min_baseline_mm", 5.0),
        )

        sw = self.cfg["sliding_window"]
        self.window = SlidingWindow(min_size=sw["min_size"], max_size=sw["max_size"])

        self.model_orders: List[int] = self.cfg["solver"]["model_orders"]
        self.manual_order: Optional[int] = self.cfg["solver"].get("manual_order", None)
        self.lambda_penalty: float = self.cfg["solver"].get("lambda_penalty", 0.05)
        self.ba_interval: int = self.cfg["solver"].get("bundle_adjustment_interval", 25)

        # ── logging ─────────────────────────────────────────────────
        log_cfg = self.cfg["logging"]
        self.recorder = AsyncRecorder(
            output_dir=log_cfg["output_dir"],
            measurements_file=log_cfg["measurements_file"],
            trajectory_file=log_cfg["trajectory_file"],
            model_selection_file=log_cfg["model_selection_file"],
            error_metrics_file=log_cfg["error_metrics_file"],
        )
        self.recorder.start()

        # ── state ───────────────────────────────────────────────────
        self._latest_metrics: Optional[WindowMetrics] = None
        self._latest_result: Optional[ModelSelectionResult] = None
        # Counter of measurements successfully accepted into the sliding
        # window.  Used for BA scheduling instead of raw frame_id so that
        # sparse or non-sequential frame IDs are handled correctly.
        self.processed_frame_count: int = 0

    # ── config loader ───────────────────────────────────────────────
    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ════════════════════════════════════════════════════════════════
    #  MAIN PROCESSING STEP  (called once per measurement)
    # ════════════════════════════════════════════════════════════════
    def process(self, m: Measurement) -> Optional[WindowMetrics]:
        """Run a single measurement through the entire pipeline.

        Returns *WindowMetrics* when the solver ran, otherwise ``None``.
        """

        # ── 0. Pixel coordinate validation ─────────────────────────
        if not self._validate_pixel(m):
            return None

        # ── 1. Geometry processing ──────────────────────────────────
        self._compute_geometry(m)

        # ── 2. Geometry filtering ───────────────────────────────────
        if not self.ray_filter.check(m, self.window.measurements):
            print(
                f"[frame {m.frame_id}] SKIP: ray filter rejected "
                f"(baseline={m.baseline_mm:.1f} mm  "
                f"ray_angle={m.median_ray_angle:.2f} deg)"
            )
            return None

        # ── 3. Push measurement into the sliding window ─────────────
        self.window.push(m)
        self.processed_frame_count += 1

        if not self.window.ready:
            # Window not yet full enough to solve — log raw measurement now
            # (trajectory fields will remain None).
            self.recorder.log_measurement(m)
            return None

        # ── 4. Trajectory solver + model selection ──────────────────
        meas = self.window.measurements
        if self.manual_order is not None:
            print(
                f"[frame {m.frame_id}] Solver: MANUAL mode N={self.manual_order}  "
                f"window={len(meas)}  baseline={m.baseline_mm:.1f} mm  "
                f"ray_angle={m.median_ray_angle:.2f} deg"
            )
            result = select_best_model(
                meas,
                orders=[self.manual_order],
                lambda_penalty=0.0,
            )
        else:
            print(
                f"[frame {m.frame_id}] Solver: AUTO mode orders={self.model_orders}  "
                f"window={len(meas)}  baseline={m.baseline_mm:.1f} mm  "
                f"ray_angle={m.median_ray_angle:.2f} deg"
            )
            result = select_best_model(
                meas,
                orders=self.model_orders,
                lambda_penalty=self.lambda_penalty,
            )

        # ── 4a. Degeneracy check ─────────────────────────────────────
        if result.is_degenerate:
            print(
                f"[frame {m.frame_id}] SKIP: degenerate system "
                f"(cond={result.condition_number:.2e}) — keeping previous estimate."
            )
            # Log the measurement without trajectory fields (they stay None).
            self.recorder.log_measurement(m)
            return None

        # Store model scores on the current measurement
        for order, score in result.scores.items():
            setattr(m, f"model_score_N{order}", score)

        # ── 4b. Periodic bundle adjustment ───────────────────────────
        # Trigger on accepted-frame count so sparse frame_id sequences
        # are handled correctly.  A ba_interval of 0 disables BA entirely.
        if (
            self.ba_interval > 0
            and (self.processed_frame_count % self.ba_interval == 0)
            and len(meas) >= self.window.min_size
        ):
            result = self._run_bundle_adjustment(meas, result, m)

        self._latest_result = result

        # ── 5. Evaluation metrics ────────────────────────────────────
        # Populates m.position, m.velocity (mm/s), m.acceleration (mm/s²),
        # m.chosen_model, and m.reprojection_error.
        metrics = compute_window_metrics(meas, result)
        self._latest_metrics = metrics

        # ── 6. Logging (after all fields are populated) ──────────────
        # Log the measurement AFTER solving so trajectory fields are present.
        self.recorder.log_measurement(m)

        # Log only the current (latest) frame to prevent duplicate rows
        # in trajectory_estimates.csv.
        self.recorder.log_trajectory(m)

        self.recorder.log_model_selection(
            frame_id=m.frame_id,
            chosen_model=result.chosen_order,
            errors=result.errors,
            scores=result.scores,
        )
        self.recorder.log_error_metrics(
            frame_id=m.frame_id,
            wm=metrics,
            error_before_BA=m.error_before_BA,
            error_after_BA=m.error_after_BA,
            bundle_adjustment_ran=m.bundle_adjustment_ran,
            condition_number=result.condition_number,
            window_size=len(meas),
        )

        return metrics

    # ── pixel validation ────────────────────────────────────────────
    def _validate_pixel(self, m: Measurement) -> bool:
        """Return ``True`` if pixel coordinates are valid.

        Rejects measurements where u/v is None, both are zero (a common
        sentinel for missing detections), or either is outside the image
        bounds defined in the camera config (keys ``width`` / ``height``).
        """
        u, v = m.u, m.v

        if u is None or v is None:
            print(f"[frame {m.frame_id}] SKIP: pixel coords are None")
            return False

        if u == 0.0 and v == 0.0:
            print(f"[frame {m.frame_id}] SKIP: pixel (0, 0) sentinel — skipping")
            return False

        if not (0.0 <= u < self.img_width) or not (0.0 <= v < self.img_height):
            print(
                f"[frame {m.frame_id}] SKIP: pixel out of bounds "
                f"(u={u:.1f}, v={v:.1f}  image={self.img_width}x{self.img_height})"
            )
            return False

        return True

    # ── geometry computation ────────────────────────────────────────
    def _compute_geometry(self, m: Measurement) -> None:
        """Populate camera_position and ray_direction on *m*."""
        assert m.robot_xyz is not None and m.robot_abc is not None

        A, B, C = m.robot_abc[0], m.robot_abc[1], m.robot_abc[2]
        R_tool_base = kuka_rotation_matrix(A, B, C)
        t_tool_base = m.robot_xyz

        R_cam_base, t_cam_base = compute_camera_pose(
            R_tool_base, t_tool_base, self.R_cam_tool, self.t_cam_tool,
        )

        m.camera_position = t_cam_base
        m.ray_direction = build_ray(
            m.u, m.v, self.K, R_cam_base, self.dist_coeffs,
        )

    # ════════════════════════════════════════════════════════════════
    #  BUNDLE ADJUSTMENT  (lightweight polynomial refinement)
    # ════════════════════════════════════════════════════════════════
    def _run_bundle_adjustment(
        self,
        measurements: List[Measurement],
        result: ModelSelectionResult,
        current_measurement: Measurement,
    ) -> ModelSelectionResult:
        """Refine trajectory polynomial coefficients via least-squares.

        Only the polynomial coefficients θ are optimised; camera poses
        are held fixed.  Uses ``scipy.optimize.least_squares`` with the
        **trf** method.

        Updates coefficients only if optimisation succeeds *and* the
        resulting error is no worse than before.
        """
        from scipy.optimize import least_squares as scipy_least_squares

        order = result.chosen_order
        n_coeffs = order + 1
        theta0 = result.coeffs.copy()
        times = result.times.copy()

        # Extract geometry arrays from measurements
        cam_positions = np.array([m.camera_position for m in measurements])
        ray_directions = np.array([m.ray_direction for m in measurements])

        # ── Error before BA ─────────────────────────────────────────
        def _evaluate_trajectory(theta: np.ndarray) -> np.ndarray:
            """Evaluate polynomial position at each time step."""
            a = [theta[3 * k : 3 * k + 3] for k in range(n_coeffs)]
            M = len(times)
            positions = np.zeros((M, 3), dtype=np.float64)
            for i, t in enumerate(times):
                pos = np.zeros(3, dtype=np.float64)
                for k in range(n_coeffs):
                    pos += a[k] * (t ** k)
                positions[i] = pos
            return positions

        def _residual_fn(theta: np.ndarray) -> np.ndarray:
            """Vector of angular reprojection errors for least_squares."""
            positions = _evaluate_trajectory(theta)
            residuals = np.zeros(len(measurements), dtype=np.float64)
            for i in range(len(measurements)):
                residuals[i] = reprojection_error_angular(
                    positions[i], cam_positions[i], ray_directions[i],
                )
            return residuals

        # ── Compute pre-BA error ────────────────────────────────────
        err_before = float(np.mean(_residual_fn(theta0)))
        current_measurement.error_before_BA = err_before

        # ── Run optimisation ────────────────────────────────────────
        try:
            opt_result = scipy_least_squares(
                _residual_fn,
                x0=theta0,
                method="trf",
            )
        except Exception:
            current_measurement.error_after_BA = err_before
            current_measurement.bundle_adjustment_ran = False
            return result

        # ── Compute post-BA error ───────────────────────────────────
        err_after = float(np.mean(_residual_fn(opt_result.x)))
        current_measurement.error_after_BA = err_after

        # ── Accept only if improvement ──────────────────────────────
        if opt_result.success and err_after <= err_before:
            current_measurement.bundle_adjustment_ran = True

            # Rebuild trajectories with refined coefficients
            new_theta = opt_result.x
            a = [new_theta[3 * k : 3 * k + 3] for k in range(n_coeffs)]

            M = len(times)
            positions = np.zeros((M, 3), dtype=np.float64)
            velocities = np.zeros((M, 3), dtype=np.float64)
            accelerations = np.zeros((M, 3), dtype=np.float64)

            for i, t in enumerate(times):
                pos = np.zeros(3, dtype=np.float64)
                for k in range(n_coeffs):
                    pos += a[k] * (t ** k)
                positions[i] = pos

                vel = np.zeros(3, dtype=np.float64)
                for k in range(1, n_coeffs):
                    vel += k * a[k] * (t ** (k - 1))
                velocities[i] = vel

                acc = np.zeros(3, dtype=np.float64)
                for k in range(2, n_coeffs):
                    acc += k * (k - 1) * a[k] * (t ** (k - 2))
                accelerations[i] = acc

            # Update result in-place
            result.coeffs = new_theta
            result.trajectories = {
                "position": positions,
                "velocity": velocities,
                "acceleration": accelerations,
            }
        else:
            current_measurement.bundle_adjustment_ran = False

        return result

    # ════════════════════════════════════════════════════════════════
    #  FINALISE  (called once after all measurements are processed)
    # ════════════════════════════════════════════════════════════════
    def finalise(self, all_measurements: List[Measurement]) -> None:
        """Stop logging and generate all plots."""
        self.recorder.stop()

        plot_cfg = self.cfg.get("plotting", {})
        if not plot_cfg.get("enabled", True):
            return

        plot_dir = plot_cfg.get("output_dir", "output/plots")
        figsize = tuple(plot_cfg.get("figsize", [10, 7]))
        dpi = plot_cfg.get("dpi", 150)

        self._generate_plots(all_measurements, plot_dir, figsize, dpi)

    def _generate_plots(
        self,
        measurements: List[Measurement],
        plot_dir: str,
        figsize: tuple,
        dpi: int,
    ) -> None:
        from trajectory_tracking.plotting.plot_geometry import (
            plot_camera_ray_geometry,
        )

        p = Path(plot_dir)

        print("\n── Generating geometry plot ──────────────────────")

        plot_camera_ray_geometry(
            measurements,
            output_path=str(p / "camera_ray_geometry.png"),
            figsize=(12, 9),
            dpi=dpi,
        )

        print("── Plot complete ────────────────────────\n")
