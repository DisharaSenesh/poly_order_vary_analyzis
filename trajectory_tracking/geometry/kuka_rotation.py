"""
KUKA ZYX Euler-angle rotation matrix.

Convention
----------
Robot reports angles *A*, *B*, *C* in **degrees**.

    R_tool_base = Rz(A) @ Ry(B) @ Rx(C)

This matches the KUKA controller convention *and* the reference
implementation in ``reference/triangulation.py``.
"""

from __future__ import annotations

import numpy as np

from trajectory_tracking.utils.math_utils import Rx, Ry, Rz


def kuka_rotation_matrix(A_deg: float, B_deg: float, C_deg: float) -> np.ndarray:
    """Build the 3×3 tool → base rotation from KUKA Euler angles.

    Parameters
    ----------
    A_deg : float
        Rotation about Z (degrees).
    B_deg : float
        Rotation about Y (degrees).
    C_deg : float
        Rotation about X (degrees).

    Returns
    -------
    np.ndarray
        3×3 rotation matrix  ``Rz(A) @ Ry(B) @ Rx(C)``.
    """
    A = np.deg2rad(A_deg)
    B = np.deg2rad(B_deg)
    C = np.deg2rad(C_deg)
    return Rz(A) @ Ry(B) @ Rx(C)
