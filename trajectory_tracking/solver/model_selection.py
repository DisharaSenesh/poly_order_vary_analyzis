"""
Motion-model selection via complexity-penalised reprojection error.

Evaluates polynomial orders *N = 0 … 6* and picks the one with the
lowest **penalised score**::

    score = mean_reprojection_error + λ × parameter_count

where ``parameter_count = 3 × (order + 1)``.

When ``lambda_penalty = 0`` the behaviour is identical to the original
minimum-error selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from trajectory_tracking.core.measurement import Measurement
from trajectory_tracking.geometry.reprojection import reprojection_error_angular
from trajectory_tracking.solver.build_system import build_polynomial_system
from trajectory_tracking.solver.trajectory_solver import solve_trajectory

# Condition number threshold above which the linear system is considered
# geometrically degenerate (underdetermined or near-singular).
_COND_THRESHOLD: float = 1e8


def _min_measurements_for_order(order: int) -> int:
    """Return the minimum number of measurements required for *order*.

    Rules:
        N = 0, 1, 2  →  5 measurements
        N = 3, 4     → 10 measurements
        N = 5, 6     → 15 measurements
    """
    if order <= 2:
        return 5
    if order <= 4:
        return 10
    return 15


@dataclass
class ModelSelectionResult:
    """Container for the model-comparison outcome."""

    chosen_order: int
    errors: Dict[int, float]   # order → mean reprojection error (deg)
    coeffs: np.ndarray         # coefficients of the chosen model
    times: np.ndarray
    trajectories: Dict[str, np.ndarray]
    # Complexity-penalised scores (order → score).
    scores: Dict[int, float] = field(default_factory=dict)
    # Condition number of the linear system for the chosen model.
    condition_number: float = 0.0
    # True when all candidate models produced an ill-conditioned system.
    is_degenerate: bool = False


def _evaluate_model(
    measurements: List[Measurement],
    order: int,
) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, np.ndarray]], float]:
    """Fit and evaluate a single model order.

    Returns (mean_error, coeffs, times, trajectories, condition_number).

    When ``cond(A) > _COND_THRESHOLD`` the system is considered degenerate:
    ``mean_error`` is returned as ``np.inf`` and coeffs/times/trajectories
    are ``None``.
    """
    # ── Minimum-measurements check (cheap, before building the matrix) ─
    if len(measurements) < _min_measurements_for_order(order):
        return np.inf, None, None, None, 0.0

    # ── Degeneracy check (does not touch solve_trajectory internals) ─
    A, _, _ = build_polynomial_system(measurements, order)
    cond = float(np.linalg.cond(A))
    if cond > _COND_THRESHOLD:
        return np.inf, None, None, None, cond

    coeffs, times, trajs = solve_trajectory(measurements, order)
    positions = trajs["position"]

    errors: List[float] = []
    for i, m in enumerate(measurements):
        err = reprojection_error_angular(positions[i], m.camera_position, m.ray_direction)
        errors.append(err)
    return float(np.mean(errors)), coeffs, times, trajs, cond


def select_best_model(
    measurements: List[Measurement],
    orders: List[int] = [0, 1, 2],
    lambda_penalty: float = 0.0,
) -> ModelSelectionResult:
    """Try each polynomial order and select the best by penalised score.

    Parameters
    ----------
    measurements : list[Measurement]
        Window of measurements with geometry already computed.
    orders : list[int]
        Candidate polynomial orders to evaluate.
    lambda_penalty : float
        Complexity penalty weight.  ``score = error + λ × 3(order+1)``.
        Default ``0.0`` preserves the original minimum-error behaviour.

    Returns
    -------
    ModelSelectionResult
    """
    best_order: int = orders[0]
    best_score: float = np.inf
    best_coeffs: Optional[np.ndarray] = None
    best_times: Optional[np.ndarray] = None
    best_trajs: Optional[Dict[str, np.ndarray]] = None
    all_errors: Dict[int, float] = {}
    model_scores: Dict[int, float] = {}
    model_conds: Dict[int, float] = {}

    for order in orders:
        try:
            mean_err, coeffs, times, trajs, cond = _evaluate_model(measurements, order)
        except np.linalg.LinAlgError:
            mean_err = np.inf
            coeffs, times, trajs, cond = None, None, None, 0.0

        all_errors[order] = mean_err
        model_conds[order] = cond

        # Penalised score (np.inf stays np.inf regardless of penalty).
        param_count = 3 * (order + 1)
        score = mean_err + lambda_penalty * param_count
        model_scores[order] = score

        if score < best_score:
            best_score = score
            best_order = order
            best_coeffs = coeffs
            best_times = times
            best_trajs = trajs

    best_cond = model_conds.get(best_order, 0.0)

    # All models degenerate — no valid trajectory can be produced.
    if best_coeffs is None:
        empty_traj: Dict[str, np.ndarray] = {
            "position": np.zeros((0, 3)),
            "velocity": np.zeros((0, 3)),
            "acceleration": np.zeros((0, 3)),
        }
        return ModelSelectionResult(
            chosen_order=best_order,
            errors=all_errors,
            coeffs=np.zeros(3 * (best_order + 1), dtype=np.float64),
            times=np.array([], dtype=np.float64),
            trajectories=empty_traj,
            scores=model_scores,
            condition_number=best_cond,
            is_degenerate=True,
        )

    return ModelSelectionResult(
        chosen_order=best_order,
        errors=all_errors,
        coeffs=best_coeffs,
        times=best_times,          # type: ignore[arg-type]
        trajectories=best_trajs,   # type: ignore[arg-type]
        scores=model_scores,
        condition_number=best_cond,
        is_degenerate=False,
    )
