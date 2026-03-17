"""
Construct the linear system for polynomial trajectory fitting.

Motion model
------------
    X(t) = a0 + a1·t + a2·t² + …

The constraint for each ray *i* is that the estimated position lies on the
viewing ray::

    d_i × (X(t_i) − C_i) = 0     (three cross-product rows)

Expanding with the polynomial model and stacking all measurements gives the
overdetermined system  ``A·θ = b``  solved via ``numpy.linalg.lstsq``.

Parameters
----------
*θ* is the concatenation of coefficient vectors for each polynomial order::

    N = 0  →  θ = [a0]                       (3 unknowns)
    N = 1  →  θ = [a0 | a1]                  (6 unknowns)
    N = 2  →  θ = [a0 | a1 | a2]             (9 unknowns)
    N = 3  →  θ = [a0 | a1 | a2 | a3]        (12 unknowns)
    N = 4  →  θ = [a0 | a1 | … | a4]         (15 unknowns)
    N = 5  →  θ = [a0 | a1 | … | a5]         (18 unknowns)
    N = 6  →  θ = [a0 | a1 | … | a6]         (21 unknowns)
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from trajectory_tracking.core.measurement import Measurement
from trajectory_tracking.utils.math_utils import skew


def build_polynomial_system(
    measurements: List[Measurement],
    order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build  ``A θ = b``  for a polynomial trajectory of given *order*.

    Parameters
    ----------
    measurements : list[Measurement]
        Each must have ``camera_position``, ``ray_direction``, and
        ``timestamp`` populated.
    order : int
        Polynomial order (0 to 6).  0 = static, 1 = const-velocity, 2 = const-accel.

    Returns
    -------
    A : np.ndarray  (3·M, 3·(order+1))
    b : np.ndarray  (3·M,)
    times : np.ndarray  (M,)
        Relative times used (t − t0).
    """
    M = len(measurements)
    n_coeffs = order + 1           # number of polynomial terms
    n_unknowns = 3 * n_coeffs      # each term is a 3-D vector

    # Use relative time to improve numerical conditioning.
    t0 = measurements[0].timestamp
    raw_times = np.array([m.timestamp - t0 for m in measurements], dtype=np.float64)

    # Normalise to [0, 1] to avoid large powers of t.
    T = raw_times[-1] if len(raw_times) > 1 else 1.0
    if abs(T) < 1e-15:
        T = 1.0  # degenerate: all timestamps identical
    times = raw_times / T

    A = np.zeros((3 * M, n_unknowns), dtype=np.float64)
    b = np.zeros(3 * M, dtype=np.float64)

    for i, m in enumerate(measurements):
        assert m.camera_position is not None and m.ray_direction is not None
        C_i = m.camera_position
        d_i = m.ray_direction
        t_i = times[i]

        Dx = skew(d_i)            # 3×3

        row = 3 * i
        # d_i × X(t_i) = d_i × C_i
        # d_i × (a0 + a1·t + a2·t² + …) = d_i × C_i
        # Dx @ (a0 + a1·t + a2·t² + …) = Dx @ C_i
        for k in range(n_coeffs):
            col = 3 * k
            A[row:row + 3, col:col + 3] = Dx * (t_i ** k)

        b[row:row + 3] = Dx @ C_i

    return A, b, times
