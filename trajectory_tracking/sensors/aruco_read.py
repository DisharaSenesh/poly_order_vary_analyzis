"""
AprilTag 36h11 detector — thin wrapper around OpenCV ArUco.

In **online** mode the ``AprilTagTracker`` class captures frames from a
camera in a background thread and exposes the latest detected centre via
a thread-safe API.

In **offline** mode (CSV replay) this module is not used; the pixel
coordinates come directly from the dataset file.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple

import cv2
import numpy as np


# ── detector configuration ──────────────────────────────────────────
TARGET_ID: int = 5

_apriltag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_params.cornerRefinementWinSize = 9
_params.cornerRefinementMaxIterations = 200
_params.cornerRefinementMinAccuracy = 1e-4
_params.adaptiveThreshWinSizeMin = 3
_params.adaptiveThreshWinSizeMax = 35
_params.adaptiveThreshWinSizeStep = 2
_params.errorCorrectionRate = 0.6

_detector = cv2.aruco.ArucoDetector(_apriltag_dict, _params)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
_subpix_criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    200,
    1e-4,
)


# ── single-frame detection ─────────────────────────────────────────
def detect_apriltag_subpixel(
    frame_bgr: np.ndarray,
) -> Tuple[list, Optional[np.ndarray]]:
    """Detect all AprilTag 36h11 markers with sub-pixel corner refinement.

    Tries CLAHE-enhanced grayscale first, then raw; prefers the result
    that contains ``TARGET_ID``.

    Returns
    -------
    corners : list
        List of corner arrays (shape [1, 4, 2]) per marker.
    ids : np.ndarray | None
        Detected marker IDs.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = _clahe.apply(gray)

    best = (gray, [], None, 0, False)
    for g in (gray_eq, gray):
        corners, ids, _ = _detector.detectMarkers(g)
        n = 0 if ids is None else len(ids)
        has_target = ids is not None and TARGET_ID in ids.flatten()
        if (has_target and not best[4]) or (has_target == best[4] and n > best[3]):
            best = (g, corners, ids, n, has_target)

    used_gray, corners, ids, _, _ = best

    if ids is not None and len(corners) > 0:
        refined = []
        for c in corners:
            pts = np.asarray(c, dtype=np.float32).reshape(-1, 1, 2)
            cv2.cornerSubPix(
                used_gray, pts,
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=_subpix_criteria,
            )
            refined.append(pts.reshape(1, 4, 2))
        corners = refined

    return corners, ids


def detect_tag_centre(frame_bgr: np.ndarray, target_id: int = TARGET_ID) -> Optional[Tuple[float, float]]:
    """Return the sub-pixel centre ``(u, v)`` of *target_id*, or ``None``."""
    corners, ids = detect_apriltag_subpixel(frame_bgr)
    if ids is None:
        return None
    ids_flat = ids.flatten().astype(int)
    matches = np.where(ids_flat == target_id)[0]
    if len(matches) == 0:
        return None
    c = corners[matches[0]].reshape(4, 2)
    centre = c.mean(axis=0)
    return float(centre[0]), float(centre[1])


# ── threaded live tracker ───────────────────────────────────────────
class AprilTagTracker:
    """Background-threaded AprilTag tracker.

    Continuously reads from a camera and exposes the latest detected
    tag centre via thread-safe getters.
    """

    def __init__(
        self,
        target_id: int = TARGET_ID,
        camera_index: int = 0,
        show_video: bool = False,
    ) -> None:
        self.target_id = target_id
        self.camera_index = camera_index
        self.show_video = show_video

        self._lock = threading.Lock()
        self._centre: Optional[Tuple[float, float]] = None
        self._detected = False
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── public API ──────────────────────────────────────────────────
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def get_centre(self) -> Optional[Tuple[float, float]]:
        with self._lock:
            return self._centre

    def is_detected(self) -> bool:
        with self._lock:
            return self._detected

    # ── background loop ─────────────────────────────────────────────
    def _loop(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            print(f"[AprilTagTracker] Cannot open camera {self.camera_index}")
            self._running = False
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        frame_count: int = 0
        _prev_detected: bool = False

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Detect in one pass so corners are available for drawing.
            corners, ids = detect_apriltag_subpixel(frame)
            result: Optional[Tuple[float, float]] = None
            target_corners: Optional[np.ndarray] = None

            if ids is not None:
                ids_flat = ids.flatten().astype(int)
                matches = np.where(ids_flat == self.target_id)[0]
                if len(matches) > 0:
                    c = corners[matches[0]].reshape(4, 2)
                    centre_pt = c.mean(axis=0)
                    result = (float(centre_pt[0]), float(centre_pt[1]))
                    target_corners = c

            # FIX: clear _centre when tag is not detected so stale
            # coordinates never leak into the pipeline.
            with self._lock:
                if result is not None:
                    self._centre = result
                    self._detected = True
                else:
                    self._centre = None
                    self._detected = False

            # Debug: print only on status change to avoid console flood.
            detected_now = result is not None
            if detected_now != _prev_detected:
                _prev_detected = detected_now
                if detected_now:
                    print(
                        f"[AprilTagTracker] Frame {frame_count}: DETECTED "
                        f"(u={result[0]:.1f}, v={result[1]:.1f})"  # type: ignore[index]
                    )
                else:
                    print(f"[AprilTagTracker] Frame {frame_count}: tag LOST")

            if self.show_video:
                display = frame.copy()

                if result is not None and target_corners is not None:
                    # Draw the four corners joined as a polygon.
                    pts = target_corners.astype(np.int32)
                    cv2.polylines(
                        display, [pts], isClosed=True, color=(0, 255, 0), thickness=2
                    )
                    # Highlight each individual corner.
                    for pt in pts:
                        cv2.circle(
                            display, (int(pt[0]), int(pt[1])), 4, (255, 0, 0), -1
                        )
                    # Draw centre point.
                    cx_px = int(round(result[0]))
                    cy_px = int(round(result[1]))
                    cv2.circle(display, (cx_px, cy_px), 6, (0, 0, 255), -1)
                    # Pixel coordinate label near the centre.
                    coord_text = f"u={result[0]:.1f}  v={result[1]:.1f}"
                    cv2.putText(
                        display, coord_text,
                        (cx_px + 10, cy_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                        cv2.LINE_AA,
                    )
                    status_text = "AprilTag detected"
                    status_color: tuple = (0, 255, 0)
                else:
                    status_text = "AprilTag not detected"
                    status_color = (0, 0, 255)

                # Detection status banner (top-left).
                cv2.putText(
                    display, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA,
                )
                # Frame counter overlay.
                cv2.putText(
                    display, f"Frame: {frame_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
                )

                # cv2.imshow("AprilTag Tracker", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                    break

        cap.release()
        if self.show_video:
            cv2.destroyAllWindows()
