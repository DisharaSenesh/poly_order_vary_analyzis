"""
Camera pose computation in the robot base frame.

The hand–eye calibration defines the *camera → tool* transform.  Combined
with the current tool → base transform the camera pose in the base frame
is obtained as:

    R_cam_base = R_tool_base @ R_cam_tool
    t_cam_base = R_tool_base @ t_cam_tool + t_tool_base

The camera optical centre (used as the ray origin) is::

    C = t_cam_base
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from trajectory_tracking.utils.math_utils import Rx, Ry, Rz


# ─── hand-eye transform ────────────────────────────────────────────
def get_hand_eye_transform(
    he: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the camera → tool rotation and translation.

    The hand-eye config contains *rx, ry, rz* (degrees) and *tx, ty, tz*
    (mm).  The rotation order used in the reference calibration is
    ``Rx @ Ry @ Rz``.

    Parameters
    ----------
    he : dict
        Keys ``rx, ry, rz, tx, ty, tz``.

    Returns
    -------
    R_cam_tool : np.ndarray   (3×3)
    t_cam_tool : np.ndarray   (3,)
    """
    rx = np.deg2rad(he["rx"])
    ry = np.deg2rad(he["ry"])
    rz = np.deg2rad(he["rz"])

    R_cam_tool = Rx(rx) @ Ry(ry) @ Rz(rz)
    t_cam_tool = np.array([he["tx"], he["ty"], he["tz"]], dtype=np.float64)
    return R_cam_tool, t_cam_tool


# ─── camera pose in base frame ─────────────────────────────────────
def compute_camera_pose(
    R_tool_base: np.ndarray,
    t_tool_base: np.ndarray,
    R_cam_tool: np.ndarray,
    t_cam_tool: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute camera pose in the robot base frame.

    Parameters
    ----------
    R_tool_base : np.ndarray  (3×3)
    t_tool_base : np.ndarray  (3,)
    R_cam_tool  : np.ndarray  (3×3)
    t_cam_tool  : np.ndarray  (3,)

    Returns
    -------
    R_cam_base  : np.ndarray  (3×3)
    t_cam_base  : np.ndarray  (3,)   — camera optical centre in base frame.
    """
    R_cam_base = R_tool_base @ R_cam_tool
    t_cam_base = R_tool_base @ t_cam_tool + t_tool_base
    return R_cam_base, t_cam_base
