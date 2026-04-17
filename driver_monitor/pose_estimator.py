# ─────────────────────────────────────────────────────────────────────────────
# pose_estimator.py  –  Optional MediaPipe upper-body pose overlay
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import cv2
from typing import Optional, Dict, Tuple


class PoseEstimator:
    """
    Wraps MediaPipe Pose to extract upper-body landmarks for the driver crop.
    Only upper-body landmarks (shoulders, elbows, wrists, nose, eyes, ears)
    are used since the lower body is hidden behind the steering wheel.

    Requires:  pip install mediapipe
    """

    # Landmark indices we care about (MediaPipe 33-point model)
    UPPER_BODY_IDX = {
        "nose":          0,
        "left_eye":      2,  "right_eye":      5,
        "left_ear":      7,  "right_ear":      8,
        "left_shoulder": 11, "right_shoulder": 12,
        "left_elbow":    13, "right_elbow":    14,
        "left_wrist":    15, "right_wrist":    16,
    }

    def __init__(self, min_detection_conf: float = 0.5,
                 min_tracking_conf: float = 0.5):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError("Run:  pip install mediapipe")

        self._mp_pose = mp.solutions.pose
        self._mp_draw = mp.solutions.drawing_utils
        self._pose    = self._mp_pose.Pose(
            static_image_mode        = False,
            model_complexity         = 0,          # 0=lite, 1=full, 2=heavy
            min_detection_confidence = min_detection_conf,
            min_tracking_confidence  = min_tracking_conf,
        )

    def estimate(self, roi: np.ndarray
                 ) -> Tuple[Optional[Dict[str, Tuple[int, int]]], np.ndarray]:
        """
        Run pose estimation on a BGR ROI (driver crop).

        Returns:
          landmarks – dict  {name: (px, py)}  or None
          vis       – annotated copy of roi
        """
        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        res = self._pose.process(rgb)
        vis = roi.copy()

        if not res.pose_landmarks:
            return None, vis

        h, w = roi.shape[:2]
        points: Dict[str, Tuple[int, int]] = {}

        for name, idx in self.UPPER_BODY_IDX.items():
            lm = res.pose_landmarks.landmark[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            points[name] = (px, py)
            cv2.circle(vis, (px, py), 4, (0, 255, 180), -1)

        # draw skeleton lines
        connections = [
            ("left_shoulder",  "right_shoulder"),
            ("left_shoulder",  "left_elbow"),
            ("left_elbow",     "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow",    "right_wrist"),
            ("left_shoulder",  "nose"),
            ("right_shoulder", "nose"),
        ]
        for a, b in connections:
            if a in points and b in points:
                cv2.line(vis, points[a], points[b], (0, 200, 100), 2)

        return points, vis

    def head_pose_check(self,
                        landmarks: Dict[str, Tuple[int, int]]) -> str:
        """
        Very rough head orientation from shoulder/nose geometry.
        Returns one of: "FORWARD" | "DISTRACTED_LEFT" | "DISTRACTED_RIGHT" | "UNKNOWN"
        """
        if landmarks is None:
            return "UNKNOWN"
        nose   = landmarks.get("nose")
        l_sh   = landmarks.get("left_shoulder")
        r_sh   = landmarks.get("right_shoulder")
        if not (nose and l_sh and r_sh):
            return "UNKNOWN"

        mid_x  = (l_sh[0] + r_sh[0]) / 2
        offset = nose[0] - mid_x
        width  = abs(l_sh[0] - r_sh[0])

        if width == 0:
            return "UNKNOWN"
        ratio = offset / width
        if ratio < -0.25:
            return "DISTRACTED_LEFT"
        elif ratio > 0.25:
            return "DISTRACTED_RIGHT"
        return "FORWARD"
