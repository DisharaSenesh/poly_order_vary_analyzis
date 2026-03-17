"""
Evaluation metrics computed after each solver step.

Populates each ``Measurement`` in the window with:
* ``reprojection_error``
* ``position`` / ``velocity`` / ``acceleration``
* ``chosen_model``

Also computes aggregate metrics for logging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from trajectory_tracking.core.measurement import Measurement
from trajectory_tracking.geometry.reprojection import reprojection_error_angular
from trajectory_tracking.solver.model_selection import ModelSelectionResult
from trajectory_tracking.utils.math_utils import angle_between


@dataclass
class WindowMetrics:
    """Aggregate metrics for the current solver window."""

    mean_reprojection_error: float
    max_reprojection_error: float
    mean_ray_angle: float
    baseline_mm: float
    chosen_model: int
    model_errors: Dict[int, float]
    # Condition number of the linear system for the chosen model order.
    condition_number: float = 0.0
    # Number of measurements in the window when this result was computed.
    window_size: int = 0


def compute_window_metrics(
    measurements: List[Measurement],
    result: ModelSelectionResult,
) -> WindowMetrics:
    """Populate measurements and return aggregate metrics.

    Parameters
    ----------
    measurements : list[Measurement]
        Window measurements (geometry already computed).
    result : ModelSelectionResult
        Output from ``select_best_model``.

    Returns
    -------
    WindowMetrics
    """
    positions = result.trajectories["position"]
    velocities = result.trajectories["velocity"]
    accelerations = result.trajectories["acceleration"]

    # ── Velocity / acceleration unit correction ───────────────────────
    # The solver normalises time to [0, 1] over the window span T so that
    # d_theta/d_t_norm = T · d_theta/d_t_real.  Dividing here converts the
    # normalised coefficients back to physically meaningful mm/s (velocity)
    # and mm/s² (acceleration) without touching the solver internals.
    t0 = measurements[0].timestamp
    t_last = measurements[-1].timestamp
    T = t_last - t0
    if abs(T) < 1e-15:
        T = 1.0  # degenerate window — avoid divide-by-zero

    reproj_errors: List[float] = []
    ray_angles: List[float] = []

    for i, m in enumerate(measurements):
        m.position = positions[i].copy()
        m.velocity = (velocities[i] / T).copy()              # mm/s
        m.acceleration = (accelerations[i] / (T * T)).copy() # mm/s²
        m.chosen_model = result.chosen_order

        err = reprojection_error_angular(positions[i], m.camera_position, m.ray_direction)
        m.reprojection_error = err
        reproj_errors.append(err)

        if m.ray_angle is not None:
            ray_angles.append(m.ray_angle)

    # Baseline: distance between first and last camera centre.
    c0 = measurements[0].camera_position
    c1 = measurements[-1].camera_position
    baseline = float(np.linalg.norm(c1 - c0)) if c0 is not None and c1 is not None else 0.0

    return WindowMetrics(
        mean_reprojection_error=float(np.mean(reproj_errors)),
        max_reprojection_error=float(np.max(reproj_errors)),
        mean_ray_angle=float(np.mean(ray_angles)) if ray_angles else 0.0,
        baseline_mm=baseline,
        chosen_model=result.chosen_order,
        model_errors=result.errors,
        condition_number=result.condition_number,
        window_size=len(measurements),
    )
