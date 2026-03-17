"""
Reprojection-error computation.

Given an estimated 3-D point **X**, a camera centre **C** and a measured
viewing-ray direction **d**, the *angular* reprojection error is the angle
between  ``(X − C)``  and **d**.
"""

from __future__ import annotations

import numpy as np

from trajectory_tracking.utils.math_utils import angle_between, normalize


def reprojection_error_angular(
    X: np.ndarray,
    C: np.ndarray,
    d: np.ndarray,
) -> float:
    """Angular reprojection error in **degrees**.

    Parameters
    ----------
    X : np.ndarray (3,)
        Estimated 3-D point (base frame).
    C : np.ndarray (3,)
        Camera optical centre (base frame).
    d : np.ndarray (3,)
        Unit viewing ray (base frame).

    Returns
    -------
    float
        Error in degrees.
    """
    v = normalize(X - C)
    return float(np.rad2deg(angle_between(v, d)))


def mean_reprojection_error(
    X: np.ndarray,
    centres: list[np.ndarray],
    directions: list[np.ndarray],
) -> float:
    """Mean angular reprojection error over a set of rays."""
    errors = [reprojection_error_angular(X, c, d) for c, d in zip(centres, directions)]
    return float(np.mean(errors))
