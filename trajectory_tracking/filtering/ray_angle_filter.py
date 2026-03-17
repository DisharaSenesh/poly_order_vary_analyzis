"""
Reject measurements whose inter-ray angle or camera baseline is too small.

Uses **median ray-angle comparison** across all rays in the window plus a
**camera baseline check** to decide whether the new measurement provides
sufficient geometric diversity for triangulation.

Previous behaviour:  compared new ray to only the *first* ray in the window.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from trajectory_tracking.core.measurement import Measurement
from trajectory_tracking.utils.math_utils import angle_between


class RayAngleFilter:
    """Drop measurements with insufficient triangulation baseline.

    Parameters
    ----------
    min_angle_deg : float
        Minimum *median* angle (degrees) between the new ray and all rays
        currently in the window.
    min_baseline_mm : float
        Minimum Euclidean distance (mm) between the new camera position
        and the most recent camera position in the window.
    max_baseline_mm : float
        Maximum Euclidean distance (mm) between the new camera position
        and the most recent camera position in the window.
    """

    def __init__(
        self,
        min_angle_deg: float = 5.0,
        min_baseline_mm: float = 10.0,
        max_baseline_mm: float = 150000.0,
    ) -> None:
        self.min_angle_deg = min_angle_deg
        self.min_baseline_mm = min_baseline_mm
        self.max_baseline_mm = max_baseline_mm

    def check(
        self,
        new_measurement: Measurement,
        window: List[Measurement],
    ) -> bool:
        """Return ``True`` if the measurement should be **kept**.

        Populates ``new_measurement.median_ray_angle`` and
        ``new_measurement.baseline_mm`` with the computed diagnostics.
        Also sets ``new_measurement.ray_angle`` for backward compatibility.
        """
        if not window:
            # First measurement is always accepted.
            new_measurement.ray_angle = 0.0
            new_measurement.median_ray_angle = 0.0
            new_measurement.baseline_mm = 0.0
            return True

        assert new_measurement.ray_direction is not None
        assert new_measurement.camera_position is not None

        # ── Camera baseline (distance to most recent measurement) ───
        last = window[-1]
        assert last.camera_position is not None
        baseline = float(np.linalg.norm(
            new_measurement.camera_position - last.camera_position
        ))
        new_measurement.baseline_mm = baseline

        # ── Window has < 2 entries: only check baseline ─────────────
        if len(window) < 2:
            new_measurement.ray_angle = 0.0
            new_measurement.median_ray_angle = 0.0
            return baseline >= self.min_baseline_mm

        # ── Compute angle between new ray and every ray in window ───
        angles_deg: List[float] = []
        for m in window:
            assert m.ray_direction is not None
            angle_rad = angle_between(m.ray_direction, new_measurement.ray_direction)
            angles_deg.append(float(np.rad2deg(angle_rad)))

        median_angle = float(np.median(angles_deg))

        # Store diagnostics
        new_measurement.ray_angle = median_angle        # backward compat
        new_measurement.median_ray_angle = median_angle

        # ── Accept only if both criteria are met ────────────────────
        return (median_angle >= self.min_angle_deg) and (baseline >= self.min_baseline_mm) and (baseline <= self.max_baseline_mm)   
