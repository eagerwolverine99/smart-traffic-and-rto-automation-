# ─────────────────────────────────────────────────────────────────────────────
# drowsiness.py  –  Eye-aspect-ratio (EAR) drowsiness detection
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import cv2
from collections import deque
from typing import Optional, Tuple


# EAR threshold: below this for CONSEC_FRAMES → alert
EAR_THRESHOLD   = 0.25
CONSEC_FRAMES   = 20          # ~0.67 s at 30 FPS


class DrowsinessDetector:
    """
    Uses dlib 68-point landmark detector (or MediaPipe) to compute EAR.
    Falls back to a brightness-based blink proxy if dlib is unavailable.

    Requires (preferred):   pip install dlib
    shape_predictor_68_face_landmarks.dat must be downloaded separately:
      http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    """

    def __init__(self, predictor_path: str = "models/shape_predictor_68_face_landmarks.dat"):
        self._use_dlib  = False
        self._counter   = 0
        self._ear_hist  = deque(maxlen=60)
        self._alert     = False

        try:
            import dlib
            self._detector  = dlib.get_frontal_face_detector()
            self._predictor = dlib.shape_predictor(predictor_path)
            self._use_dlib  = True
            print("[Drowsiness] dlib landmark detector loaded.")
        except Exception:
            print("[Drowsiness] dlib not available – using brightness proxy.")

    # ── public API ────────────────────────────────────────────────────────────

    def analyse(self, face_crop: np.ndarray
                ) -> Tuple[float, bool, np.ndarray]:
        """
        Analyse a single driver face crop.

        Returns:
          ear   – eye aspect ratio (0–1; lower = more closed)
          alert – True if drowsiness detected
          vis   – annotated face crop
        """
        if self._use_dlib:
            ear, vis = self._ear_dlib(face_crop)
        else:
            ear, vis = self._ear_brightness(face_crop)

        if ear < EAR_THRESHOLD:
            self._counter += 1
        else:
            self._counter = 0

        self._ear_hist.append(ear)
        self._alert = (self._counter >= CONSEC_FRAMES)

        color = (0, 0, 255) if self._alert else (0, 255, 0)
        label = f"EAR:{ear:.2f}  {'DROWSY!' if self._alert else 'AWAKE'}"
        cv2.putText(vis, label, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return ear, self._alert, vis

    # ── backends ──────────────────────────────────────────────────────────────

    def _ear_dlib(self, crop: np.ndarray) -> Tuple[float, np.ndarray]:
        import dlib
        gray  = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        rects = self._detector(gray, 0)
        vis   = crop.copy()
        if not rects:
            return 1.0, vis

        shape  = self._predictor(gray, rects[0])
        pts    = np.array([[shape.part(i).x, shape.part(i).y]
                           for i in range(68)])

        left_eye  = pts[36:42]
        right_eye = pts[42:48]
        ear = (self._compute_ear(left_eye) + self._compute_ear(right_eye)) / 2.0

        for pt in np.vstack([left_eye, right_eye]):
            cv2.circle(vis, tuple(pt), 2, (0, 255, 255), -1)

        return float(ear), vis

    @staticmethod
    def _compute_ear(eye: np.ndarray) -> float:
        """Eye Aspect Ratio formula (Soukupova & Cech, 2016)."""
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C + 1e-6)

    @staticmethod
    def _ear_brightness(crop: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Brightness proxy: the upper-eye region is darker when eyes are closed.
        Returns a normalised value in [0,1].
        """
        vis  = crop.copy()
        h, w = crop.shape[:2]
        # eye band: vertical 20–45% of face height, full width
        eye_band = crop[int(h*0.20):int(h*0.45), :]
        gray     = cv2.cvtColor(eye_band, cv2.COLOR_BGR2GRAY)
        # high brightness → eyes open; low → eyes closed (approximation)
        ear = float(gray.mean()) / 255.0
        return ear, vis
