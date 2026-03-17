"""
KUKA robot interface via OpenShowVar (py_openshowvar).

Reads the current Cartesian tool pose ``[X, Y, Z, A, B, C]`` from the
controller.  This module is only active in **online** mode.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np


class KUKAControl:
    """Thin wrapper around ``py_openshowvar`` for reading KUKA pose.

    Parameters
    ----------
    ip : str
        Controller IP address.
    port : int
        OpenShowVar port (typically 7000).
    """

    def __init__(self, ip: str, port: int) -> None:
        self.ip = ip
        self.port = port
        self._client = None

    # ── connection ──────────────────────────────────────────────────
    def connect(self) -> bool:
        try:
            from py_openshowvar import openshowvar  # type: ignore[import-untyped]

            self._client = openshowvar(self.ip, self.port)
            if not self._client.can_connect:
                print(f"[KUKAControl] Cannot connect to {self.ip}:{self.port}")
                self._client = None
                return False
            print(f"[KUKAControl] Connected to {self.ip}:{self.port}")
            return True
        except Exception as exc:
            print(f"[KUKAControl] Connection error: {exc}")
            self._client = None
            return False

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # ── pose reading ────────────────────────────────────────────────
    def read_pose(self) -> Optional[np.ndarray]:
        """Return ``[X, Y, Z, A, B, C]`` or ``None`` on failure."""
        if self._client is None:
            return None
        try:
            x = float(self._client.read("$POS_ACT.X", debug=False).decode())
            y = float(self._client.read("$POS_ACT.Y", debug=False).decode())
            z = float(self._client.read("$POS_ACT.Z", debug=False).decode())
            a = float(self._client.read("$POS_ACT.A", debug=False).decode())
            b = float(self._client.read("$POS_ACT.B", debug=False).decode())
            c = float(self._client.read("$POS_ACT.C", debug=False).decode())
            return np.array([x, y, z, a, b, c], dtype=np.float64)
        except Exception as exc:
            print(f"[KUKAControl] Pose read error: {exc}")
            return None
