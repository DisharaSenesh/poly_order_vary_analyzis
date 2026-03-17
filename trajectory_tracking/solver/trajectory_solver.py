"""
Solve the polynomial trajectory from the linear system.

Returns the coefficient vectors and derives position / velocity /
acceleration at every measurement time.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from trajectory_tracking.core.measurement import Measurement
from trajectory_tracking.solver.build_system import build_polynomial_system


def solve_trajectory(
    measurements: List[Measurement],
    order: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """Least-squares fit of a polynomial trajectory.

    Parameters
    ----------
    measurements : list[Measurement]
    order : int
        0 to 6.

    Returns
    -------
    coeffs : np.ndarray  (3*(order+1),)
        Stacked coefficient vectors  [a0 | a1 | a2 …].
    times : np.ndarray  (M,)
        Relative times.
    trajectories : dict
        Keys: ``position`` (M×3), ``velocity`` (M×3), ``acceleration`` (M×3).
    """
    A, b, times = build_polynomial_system(measurements, order)
    theta, *_ = np.linalg.lstsq(A, b, rcond=None)

    n_coeffs = order + 1
    # Unpack coefficients → list of 3-vectors
    a: List[np.ndarray] = [
        theta[3 * k : 3 * k + 3] for k in range(n_coeffs)
    ]

    M = len(times)
    positions = np.zeros((M, 3), dtype=np.float64)
    velocities = np.zeros((M, 3), dtype=np.float64)
    accelerations = np.zeros((M, 3), dtype=np.float64)

    for i, t in enumerate(times):
        # X(t) = Σ a_k · t^k
        pos = np.zeros(3, dtype=np.float64)
        for k in range(n_coeffs):
            pos += a[k] * (t ** k)
        positions[i] = pos

        # V(t) = Σ k · a_k · t^(k-1)
        vel = np.zeros(3, dtype=np.float64)
        for k in range(1, n_coeffs):
            vel += k * a[k] * (t ** (k - 1))
        velocities[i] = vel

        # A(t) = Σ k·(k-1) · a_k · t^(k-2)
        acc = np.zeros(3, dtype=np.float64)
        for k in range(2, n_coeffs):
            acc += k * (k - 1) * a[k] * (t ** (k - 2))
        accelerations[i] = acc

    trajectories = {
        "position": positions,
        "velocity": velocities,
        "acceleration": accelerations,
    }
    return theta, times, trajectories
