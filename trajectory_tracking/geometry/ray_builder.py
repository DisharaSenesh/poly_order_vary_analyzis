"""
Viewing-ray construction from pixel coordinates.

Steps
-----
1. Optionally undistort the centre pixel with ``cv2.undistortPoints``.
2. Convert to normalised camera coordinates::

       x = (u − cx) / fx
       y = (v − cy) / fy
       d_cam = [x, y, 1]

3. Normalise, then transform to base frame::

       d_base = R_cam_base @ d_cam / ||R_cam_base @ d_cam||
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from trajectory_tracking.utils.math_utils import normalize


def undistort_pixel(
    u: float,
    v: float,
    K: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Undistort a single pixel and return normalised camera-frame ray.

    If *dist_coeffs* is ``None`` or all-zero, simple pin-hole back-projection
    is used instead of ``cv2.undistortPoints``.

    Returns
    -------
    d_cam : np.ndarray (3,)
        ``[x, y, 1]``  — normalised, **not** unit-length (caller should call
        :func:`normalize` if needed).
    """
    if dist_coeffs is not None and np.any(dist_coeffs != 0):
        src = np.array([[[u, v]]], dtype=np.float64)
        dst = cv2.undistortPoints(src, K, dist_coeffs)
        x = float(dst[0, 0, 0])
        y = float(dst[0, 0, 1])
    else:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        x = (u - cx) / fx
        y = (v - cy) / fy

    return np.array([x, y, 1.0], dtype=np.float64)


def build_ray(
    u: float,
    v: float,
    K: np.ndarray,
    R_cam_base: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a unit viewing-ray in the base frame.

    Parameters
    ----------
    u, v : float
        Pixel centre coordinates.
    K : np.ndarray (3×3)
        Camera intrinsic matrix.
    R_cam_base : np.ndarray (3×3)
        Camera → base rotation (= R_tool_base @ R_cam_tool).
    dist_coeffs : np.ndarray | None
        Distortion coefficients (k1 k2 p1 p2 k3).

    Returns
    -------
    d_base : np.ndarray (3,)
        Unit direction vector in the base frame.
    """
    d_cam = undistort_pixel(u, v, K, dist_coeffs)
    d_cam = normalize(d_cam)
    d_base = R_cam_base @ d_cam
    d_base = normalize(d_base)
    return d_base
