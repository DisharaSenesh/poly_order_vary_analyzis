import numpy as np
import cv2
import threading
import time


# AprilTag 36h11 detection (target ID=5) with subpixel-accurate corners
# arUco tag
TARGET_ID = 5

apriltag_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 9
params.cornerRefinementMaxIterations = 200
params.cornerRefinementMinAccuracy = 1e-4
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 35
params.adaptiveThreshWinSizeStep = 2
params.errorCorrectionRate = 0.6

detector = cv2.aruco.ArucoDetector(apriltag_dict, params)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-4)


def detect_apriltag_subpixel(frame_bgr):
    """
    Detect all AprilTag 36h11 markers in a BGR frame with subpixel-accurate corners.

    Tries CLAHE-enhanced grayscale first, then raw grayscale; prefers whichever
    result contains TARGET_ID, then whichever finds more markers overall.

    Args:
        frame_bgr: BGR image (numpy array).

    Returns:
        corners: list of refined corner arrays (shape [1, 4, 2] each).
        ids:     numpy array of detected IDs, or None.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_eq = clahe.apply(gray)

    # Try enhanced first, then raw; prefer the one that finds TARGET_ID.
    candidates = [gray_eq, gray]
    best = (None, None, None, -1, False)  # (gray, corners, ids, count, has_target)

    for g in candidates:
        corners, ids, _ = detector.detectMarkers(g)
        n = 0 if ids is None else len(ids)
        has_target = ids is not None and TARGET_ID in ids.flatten()

        if (has_target and not best[4]) or (has_target == best[4] and n > best[3]):
            best = (g, corners, ids, n, has_target)

    used_gray, corners, ids, _, _ = best

    # Explicit subpixel refinement on each detected corner.
    if ids is not None and len(corners) > 0:
        refined = []
        for c in corners:
            pts = np.asarray(c, dtype=np.float32).reshape(-1, 1, 2)
            cv2.cornerSubPix(used_gray, pts, winSize=(5, 5),
                             zeroZone=(-1, -1), criteria=subpix_criteria)
            refined.append(pts.reshape(1, 4, 2))
        corners = refined

    return corners, ids


def center_track(image_file):
    """
    Detect AprilTag 36h11 (ID=TARGET_ID) in a still image and return
    subpixel-accurate corner coordinates and the tag center.

    Args:
        image_file (str): Path to the image file.

    Returns:
        tuple: (c1, c2, c3, c4, center) where each element is an (x, y)
               pixel coordinate, ordered [top-left, top-right, bottom-right,
               bottom-left].  Returns None if TARGET_ID is not detected.

    Raises:
        FileNotFoundError: If the image cannot be read from disk.
    """
    frame = cv2.imread(image_file)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_file}")

    corners, ids = detect_apriltag_subpixel(frame)

    if ids is None:
        return None

    ids_flat = ids.flatten().astype(int)
    matches = np.where(ids_flat == TARGET_ID)[0]
    if len(matches) == 0:
        return None

    c = corners[matches[0]].reshape(4, 2)
    c1, c2, c3, c4 = tuple(c[0]), tuple(c[1]), tuple(c[2]), tuple(c[3])
    center = tuple(c.mean(axis=0))

    return c1, c2, c3, c4, center


class AprilTagTracker:
    """
    Threaded AprilTag tracker that continuously reads from a camera and
    exposes the latest detected position via a thread-safe interface.
    """

    def __init__(self, target_id=TARGET_ID, camera_index=0, show_video=True):
        """
        Initialize and immediately start the tracking thread.

        Args:
            target_id (int):    AprilTag ID to track (default: TARGET_ID).
            camera_index (int): Camera device index (default: 0).
            show_video (bool):  Whether to display a live video window.
        """
        self.target_id = target_id
        self.camera_index = camera_index
        self.show_video = show_video

        # Thread-safe state
        self._lock = threading.Lock()
        self._current_center = None
        self._current_corners = None
        self._is_detected = False

        # Thread control
        self._running = False
        self._thread = None

        self.start()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self):
        """Start the tracking thread (no-op if already running)."""
        if self._running:
            print("Tracker is already running.")
            return

        self._running = True
        self._thread = threading.Thread(target=self._track_loop, daemon=True)
        self._thread.start()
        print(f"AprilTag tracker started for ID: {self.target_id}")

    def stop(self):
        """Signal the tracking thread to stop and wait for it to finish."""
        if not self._running:
            print("Tracker is not running.")
            return

        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        cv2.destroyAllWindows()
        print("Tracker stopped.")

    def get_current_coordinate(self):
        """
        Return the latest center coordinate of the tracked tag.

        Returns:
            tuple | None: (x, y) in pixels, or None if not currently detected.
        """
        with self._lock:
            return self._current_center

    def get_current_corners(self):
        """
        Return the latest corner coordinates of the tracked tag.

        Returns:
            tuple | None: (c1, c2, c3, c4) each as (x, y) in pixels,
                          or None if not currently detected.
        """
        with self._lock:
            return self._current_corners

    def is_detected(self):
        """
        Return whether the target tag is currently visible.

        Returns:
            bool: True if the tag is detected in the most recent frame.
        """
        with self._lock:
            return self._is_detected

    # ------------------------------------------------------------------
    # Internal loop
    # ------------------------------------------------------------------

    def _track_loop(self):
        """Capture frames and update detection state until stop() is called."""
        cap = cv2.VideoCapture(self.camera_index)

        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            self._running = False
            return

        # Request a specific capture resolution.
        # Note: many webcams ignore this; always read back the actual values.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera capture resolution: {actual_w}x{actual_h}")
        

        print("Camera opened. Tracking started.")
        if self.show_video:
            print("Press 'q' in the video window to quit.")

        while self._running:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            # One-time sanity print of the actual frame shape (in case CAP_PROP lies)
            if hasattr(self, "_printed_frame_shape") is False:
                self._printed_frame_shape = True
                h, w = frame.shape[:2]
                print(f"Camera frame shape: {w}x{h}")

            corners, ids = detect_apriltag_subpixel(frame)

            detected = False
            center = None
            corners_tuple = None

            if ids is not None:
                ids_flat = ids.flatten().astype(int)
                for i, marker_id in enumerate(ids_flat):
                    if marker_id == self.target_id:
                        c = corners[i].reshape(4, 2)
                        center = tuple(c.mean(axis=0))
                        corners_tuple = (
                            tuple(c[0]),  # top-left
                            tuple(c[1]),  # top-right
                            tuple(c[2]),  # bottom-right
                            tuple(c[3]),  # bottom-left
                        )
                        detected = True
                        break

            with self._lock:
                self._is_detected = detected
                self._current_center = center
                self._current_corners = corners_tuple

            if self.show_video:
                annotated = frame.copy()

                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(annotated, corners, ids)

                if detected and center is not None:
                    cv2.circle(annotated,
                                (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
                    cv2.putText(
                        annotated,
                        f"ID {self.target_id} Center: ({center[0]:.2f}, {center[1]:.2f})",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow("AprilTag Live Tracking", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._running = False
                    break

        cap.release()
        if self.show_video:
            cv2.destroyAllWindows()


# if __name__ == "__main__":
#     tracker = AprilTagTracker(target_id=TARGET_ID, show_video=True)

#     try:
#         while True:
#             time.sleep(1)
#             if tracker.is_detected():
#                 center = tracker.get_current_coordinate()
#                 if center:
#                     print(f"Current Center: ({center[0]:.4f}, {center[1]:.4f})")
#             else:
#                 print("AprilTag not detected")
#     except KeyboardInterrupt:
#         print("\nStopping tracker...")
#         tracker.stop()
