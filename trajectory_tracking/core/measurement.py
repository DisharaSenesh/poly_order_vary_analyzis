"""
Central data class that flows through every pipeline stage.

Every module reads from / writes to the same ``Measurement`` instance,
ensuring a single source of truth for each observation.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Measurement:
    """One synchronised camera + robot observation.

    Attributes
    ----------
    frame_id : int
        Sequential frame counter.
    timestamp : float
        Wall-clock time of the observation (seconds).
    u, v : float
        Detected tag centre in pixel coordinates.
    robot_xyz : np.ndarray | None
        Tool position [X, Y, Z] in mm (base frame).
    robot_abc : np.ndarray | None
        Tool orientation [A, B, C] in degrees (KUKA convention).

    camera_position : np.ndarray | None
        Camera optical-centre in the base frame (mm).
    ray_direction : np.ndarray | None
        Unit viewing-ray direction in the base frame.
    ray_angle : float | None
        Angle (degrees) between pairs of rays used for triangulation.
    median_ray_angle : float | None
        Median angle (degrees) between the new ray and all rays in window.
    baseline_mm : float | None
        Camera baseline (mm) between the new and last camera position.

    position : np.ndarray | None
        Estimated 3-D position of the tracked object (mm).
    velocity : np.ndarray | None
        Estimated velocity vector (mm s⁻¹).
    acceleration : np.ndarray | None
        Estimated acceleration vector (mm s⁻²).

    reprojection_error : float | None
        Angular reprojection error (degrees or mm, depending on metric).
    chosen_model : int | None
        Motion-model order selected for this window (0, 1, or 2).

    model_score_N0 : float | None
        Complexity-penalised score for model order 0.
    model_score_N1 : float | None
        Complexity-penalised score for model order 1.
    model_score_N2 : float | None
        Complexity-penalised score for model order 2.

    error_before_BA : float | None
        Mean angular reprojection error before bundle adjustment.
    error_after_BA : float | None
        Mean angular reprojection error after bundle adjustment.
    bundle_adjustment_ran : bool | None
        Whether bundle adjustment was executed for this frame.
    """

    # --- raw observation ---
    frame_id: int = 0
    timestamp: float = 0.0

    u: float = 0.0
    v: float = 0.0

    robot_xyz: Optional[np.ndarray] = None
    robot_abc: Optional[np.ndarray] = None

    # --- geometry (filled by pipeline) ---
    camera_position: Optional[np.ndarray] = None
    ray_direction: Optional[np.ndarray] = None
    ray_angle: Optional[float] = None
    median_ray_angle: Optional[float] = None
    baseline_mm: Optional[float] = None

    # --- trajectory estimates (filled by solver) ---
    position: Optional[np.ndarray] = None
    velocity: Optional[np.ndarray] = None
    acceleration: Optional[np.ndarray] = None

    # --- evaluation (filled by evaluation stage) ---
    reprojection_error: Optional[float] = None
    chosen_model: Optional[int] = None

    # --- model selection diagnostics ---
    model_score_N0: Optional[float] = None
    model_score_N1: Optional[float] = None
    model_score_N2: Optional[float] = None

    # --- ground truth (evaluation only, not used by solver) ---
    gt_position: Optional[np.ndarray] = None

    # --- bundle adjustment diagnostics ---
    error_before_BA: Optional[float] = None
    error_after_BA: Optional[float] = None
    bundle_adjustment_ran: Optional[bool] = None

    # -----------------------------------------------------------------
    # Convenience helpers
    # -----------------------------------------------------------------
    def pixel(self) -> np.ndarray:
        """Return pixel coordinates as a length-2 array."""
        return np.array([self.u, self.v], dtype=np.float64)

    def has_geometry(self) -> bool:
        """True once camera_position and ray_direction have been computed."""
        return self.camera_position is not None and self.ray_direction is not None

    def to_dict(self) -> dict:
        """Serialise to a flat dictionary suitable for CSV / DataFrame."""
        d: dict = {}
        d["frame_id"] = self.frame_id
        d["timestamp"] = self.timestamp
        d["u"] = self.u
        d["v"] = self.v

        for axis, label in enumerate(("X", "Y", "Z")):
            d[f"robot_{label}"] = (
                float(self.robot_xyz[axis]) if self.robot_xyz is not None else None
            )
        for axis, label in enumerate(("A", "B", "C")):
            d[f"robot_{label}"] = (
                float(self.robot_abc[axis]) if self.robot_abc is not None else None
            )

        for axis, label in enumerate(("x", "y", "z")):
            d[f"cam_pos_{label}"] = (
                float(self.camera_position[axis])
                if self.camera_position is not None
                else None
            )
        for axis, label in enumerate(("x", "y", "z")):
            d[f"ray_dir_{label}"] = (
                float(self.ray_direction[axis])
                if self.ray_direction is not None
                else None
            )

        d["ray_angle"] = self.ray_angle
        d["median_ray_angle"] = self.median_ray_angle
        d["baseline_mm"] = self.baseline_mm

        for axis, label in enumerate(("x", "y", "z")):
            d[f"pos_{label}"] = (
                float(self.position[axis]) if self.position is not None else None
            )
        for axis, label in enumerate(("x", "y", "z")):
            d[f"vel_{label}"] = (
                float(self.velocity[axis]) if self.velocity is not None else None
            )
        for axis, label in enumerate(("x", "y", "z")):
            d[f"acc_{label}"] = (
                float(self.acceleration[axis])
                if self.acceleration is not None
                else None
            )

        for axis, label in enumerate(("x", "y", "z")):
            d[f"gt_{label}"] = (
                float(self.gt_position[axis])
                if self.gt_position is not None
                else None
            )

        d["reprojection_error"] = self.reprojection_error
        d["chosen_model"] = self.chosen_model

        # --- diagnostic fields ---
        d["model_score_N0"] = self.model_score_N0
        d["model_score_N1"] = self.model_score_N1
        d["model_score_N2"] = self.model_score_N2
        d["error_before_BA"] = self.error_before_BA
        d["error_after_BA"] = self.error_after_BA
        d["bundle_adjustment_ran"] = self.bundle_adjustment_ran
        return d

    def __repr__(self) -> str:  # noqa: D105
        pos_str = (
            f"[{self.position[0]:.1f}, {self.position[1]:.1f}, {self.position[2]:.1f}]"
            if self.position is not None
            else "None"
        )
        return (
            f"Measurement(frame={self.frame_id}, t={self.timestamp:.4f}, "
            f"px=({self.u:.1f},{self.v:.1f}), pos={pos_str})"
        )
