"""
Low-level mathematical helpers used across the pipeline.

All angles are in **radians** unless stated otherwise.
"""

from __future__ import annotations

import numpy as np


# ── rotation primitives ──────────────────────────────────────────────
def Rx(angle: float) -> np.ndarray:
    """Elementary rotation about the X axis (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1.0,  0.0, 0.0],
        [0.0,   c,  -s ],
        [0.0,   s,   c ],
    ], dtype=np.float64)


def Ry(angle: float) -> np.ndarray:
    """Elementary rotation about the Y axis (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c,  0.0,  s],
        [0.0, 1.0, 0.0],
        [-s,  0.0,  c],
    ], dtype=np.float64)


def Rz(angle: float) -> np.ndarray:
    """Elementary rotation about the Z axis (radians)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c,  -s, 0.0],
        [ s,   c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# ── skew-symmetric (hat) map ────────────────────────────────────────
def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3×3 skew-symmetric matrix of vector *v*."""
    vx, vy, vz = v.ravel()[:3]
    return np.array([
        [ 0.0, -vz,  vy],
        [ vz,  0.0, -vx],
        [-vy,  vx,  0.0],
    ], dtype=np.float64)


# ── unit-vector helpers ──────────────────────────────────────────────
def normalize(v: np.ndarray) -> np.ndarray:
    """Return *v / ||v||*.  Returns zeros for zero-length vectors."""
    n = np.linalg.norm(v)
    if n < 1e-15:
        return np.zeros_like(v)
    return v / n


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Angle in **radians** between two vectors."""
    a_n = normalize(a)
    b_n = normalize(b)
    dot = np.clip(np.dot(a_n, b_n), -1.0, 1.0)
    return float(np.arccos(dot))


# ── intrinsics ───────────────────────────────────────────────────────
def intrinsics_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a 3×3 camera intrinsic matrix."""
    return np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
