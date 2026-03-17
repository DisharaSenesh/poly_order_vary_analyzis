"""
Measurement synchronisation.

In **online** mode the synchroniser polls the camera tracker and robot
controller, pairs the most recent readings, and emits a ``Measurement``.

In **offline** mode a CSV row is converted directly.
"""

from __future__ import annotations

import time
from typing import Optional

from typing import Optional

import numpy as np

from trajectory_tracking.core.measurement import Measurement


class MeasurementSync:
    """Pair camera and robot readings into ``Measurement`` objects."""

    def __init__(self) -> None:
        self._frame_counter: int = 0

    def from_online(
        self,
        pixel_uv: tuple[float, float],
        robot_pose: np.ndarray,
    ) -> Measurement:
        """Create a measurement from live sensor data.

        Parameters
        ----------
        pixel_uv : (u, v)
            Detected tag centre in pixels.
        robot_pose : np.ndarray (6,)
            ``[X, Y, Z, A, B, C]`` from the robot controller.
        """
        m = Measurement(
            frame_id=self._frame_counter,
            timestamp=time.time(),
            u=pixel_uv[0],
            v=pixel_uv[1],
            robot_xyz=robot_pose[:3].copy(),
            robot_abc=robot_pose[3:].copy(),
        )
        self._frame_counter += 1
        return m

    def from_csv_row(
        self,
        frame_id: int,
        timestamp: float,
        u: float,
        v: float,
        X: float,
        Y: float,
        Z: float,
        A: float,
        B: float,
        C: float,
        gt_x: Optional[float] = None,
        gt_y: Optional[float] = None,
        gt_z: Optional[float] = None,
    ) -> Measurement:
        """Create a measurement from a CSV dataset row."""
        gt_pos = None
        if gt_x is not None and gt_y is not None and gt_z is not None:
            gt_pos = np.array([gt_x, gt_y, gt_z], dtype=np.float64)
        return Measurement(
            frame_id=frame_id,
            timestamp=timestamp,
            u=u,
            v=v,
            robot_xyz=np.array([X, Y, Z], dtype=np.float64),
            robot_abc=np.array([A, B, C], dtype=np.float64),
            gt_position=gt_pos,
        )
