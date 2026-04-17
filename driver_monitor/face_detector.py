# ─────────────────────────────────────────────────────────────────────────────
# face_detector.py  –  Multi-backend face detection
# Backends: "dnn" (default) | "haar" | "retinaface"
# DNN always tries: DNN → frontal Haar → profile Haar → upper-region retry
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import urllib.request
import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Optional

from config import (FACE_BACKEND, DNN_CONF, DNN_PROTO, DNN_MODEL,
                    HAAR_CASCADE, MODEL_DIR)

_PROTO_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/"
    "deploy.prototxt"
)
_MODEL_URL = (
    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/"
    "res10_300x300_ssd_iter_140000.caffemodel"
)


@dataclass
class FaceDetection:
    bbox: tuple        # (x1, y1, x2, y2) in ROI coordinates
    confidence: float
    crop: np.ndarray   # BGR crop of the face


class FaceDetector:

    def __init__(self, backend: str = FACE_BACKEND):
        self.backend   = backend.lower()
        self._haar     = None
        self._haar_profile = None
        self._dnn_net  = None

        if self.backend == "haar":
            self._load_haar()
        elif self.backend == "dnn":
            self._load_dnn()
            self._load_haar()           # Haar always loaded as fallback
        elif self.backend == "retinaface":
            self._load_retinaface()
        else:
            raise ValueError(f"Unknown backend: {backend}")

    # ── loaders ───────────────────────────────────────────────────────────────

    def _load_haar(self):
        frontal_path = (HAAR_CASCADE if os.path.exists(HAAR_CASCADE)
                        else cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        profile_path = cv2.data.haarcascades + "haarcascade_profileface.xml"

        self._haar = cv2.CascadeClassifier(frontal_path)
        if self._haar.empty():
            raise RuntimeError(f"Cannot load frontal Haar: {frontal_path}")

        # profile cascade — silently skip if missing (older OpenCV builds)
        if os.path.exists(profile_path):
            self._haar_profile = cv2.CascadeClassifier(profile_path)
            if self._haar_profile.empty():
                self._haar_profile = None

    def _load_dnn(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        if not os.path.exists(DNN_PROTO):
            print("[FaceDetector] Downloading deploy.prototxt …")
            urllib.request.urlretrieve(_PROTO_URL, DNN_PROTO)
        if not os.path.exists(DNN_MODEL):
            print("[FaceDetector] Downloading caffemodel (~10 MB) …")
            urllib.request.urlretrieve(_MODEL_URL, DNN_MODEL)
        self._dnn_net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)

    def _load_retinaface(self):
        try:
            from insightface.app import FaceAnalysis
            self._retina = FaceAnalysis(allowed_modules=["detection"])
            self._retina.prepare(ctx_id=-1, det_size=(640, 640))
        except ImportError:
            raise ImportError("pip install insightface onnxruntime")

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, roi: np.ndarray,
               min_face_ratio: float = 0.005) -> List[FaceDetection]:
        """
        Detect all faces in `roi` (BGR).
        min_face_ratio: drop faces whose area < this fraction of roi — default
        lowered to 0.005 so large-frame images don't incorrectly discard hits.
        """
        if roi is None or roi.size == 0:
            return []

        if self.backend == "haar":
            faces = self._run_haar(roi)
        elif self.backend == "dnn":
            faces = self._run_dnn_with_fallbacks(roi)
        else:
            faces = self._run_retinaface(roi)

        # filter tiny detections
        roi_area = roi.shape[0] * roi.shape[1]
        min_area = roi_area * min_face_ratio
        faces    = [f for f in faces
                    if (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]) >= min_area]
        faces.sort(key=lambda f: f.confidence, reverse=True)
        return faces

    # ── DNN pipeline with layered fallbacks ───────────────────────────────────

    def _run_dnn_with_fallbacks(self, roi: np.ndarray) -> List[FaceDetection]:
        # Pass 1: DNN on full ROI
        faces = self._dnn(roi, conf_override=DNN_CONF)
        if faces:
            return faces

        # Pass 2: DNN on upper-60% (face usually in top portion of vehicle crop)
        h = roi.shape[0]
        upper = roi[:int(h * 0.65), :]
        faces = self._dnn(upper, conf_override=DNN_CONF)
        if faces:
            return self._rebase(faces, 0, 0)   # coords already relative to upper crop

        # Pass 3: DNN at very low threshold on full ROI
        faces = self._dnn(roi, conf_override=0.20)
        if faces:
            return faces

        # Pass 4: frontal Haar on full ROI
        faces = self._run_haar(roi, profile=False)
        if faces:
            return faces

        # Pass 5: profile Haar (handles side/3-quarter angle faces)
        faces = self._run_haar(roi, profile=True)
        return faces

    def _dnn(self, roi: np.ndarray,
             conf_override: float = DNN_CONF) -> List[FaceDetection]:
        """Run single DNN inference pass on roi."""
        h, w = roi.shape[:2]
        if h == 0 or w == 0:
            return []

        # correct blob — let blobFromImage handle resize internally (no pre-resize)
        blob = cv2.dnn.blobFromImage(
            roi, scalefactor=1.0, size=(300, 300),
            mean=(104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self._dnn_net.setInput(blob)
        dets    = self._dnn_net.forward()
        results = []

        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf < conf_override:
                continue
            x1 = max(0, int(dets[0, 0, i, 3] * w))
            y1 = max(0, int(dets[0, 0, i, 4] * h))
            x2 = min(w, int(dets[0, 0, i, 5] * w))
            y2 = min(h, int(dets[0, 0, i, 6] * h))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                crop=roi[y1:y2, x1:x2].copy(),
            ))
        return results

    def _run_haar(self, roi: np.ndarray,
                  profile: bool = False) -> List[FaceDetection]:
        cascade = self._haar_profile if (profile and self._haar_profile) else self._haar
        if cascade is None:
            return []

        gray  = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # equalizeHist helps in uneven lighting before Haar
        gray  = cv2.equalizeHist(gray)

        rects = cascade.detectMultiScale(
            gray,
            scaleFactor  = 1.05,    # finer scale steps → catches more faces
            minNeighbors = 3,       # lowered from 5 → better recall
            minSize      = (20, 20),
            flags        = cv2.CASCADE_SCALE_IMAGE,
        )
        results = []
        for (x, y, w, h) in (rects if len(rects) else []):
            x1, y1, x2, y2 = x, y, x+w, y+h
            results.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=0.70,
                crop=roi[y1:y2, x1:x2].copy(),
            ))

        # also try horizontally flipped for profile cascade (catches both sides)
        if profile and self._haar_profile and len(results) == 0:
            flipped = cv2.flip(gray, 1)
            rects2  = self._haar_profile.detectMultiScale(
                flipped, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20)
            )
            fw = roi.shape[1]
            for (x, y, w, h) in (rects2 if len(rects2) else []):
                # mirror x back to original coordinates
                x1 = fw - (x + w)
                x2 = fw - x
                y1, y2 = y, y + h
                x1, x2 = max(0, x1), min(fw, x2)
                results.append(FaceDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=0.65,
                    crop=roi[y1:y2, x1:x2].copy(),
                ))
        return results

    def _run_retinaface(self, roi: np.ndarray) -> List[FaceDetection]:
        results = []
        for face in self._retina.get(roi):
            x1, y1, x2, y2 = map(int, face.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(roi.shape[1], x2), min(roi.shape[0], y2)
            results.append(FaceDetection(
                bbox=(x1, y1, x2, y2),
                confidence=float(face.det_score),
                crop=roi[y1:y2, x1:x2].copy(),
            ))
        return results

    @staticmethod
    def _rebase(faces: List[FaceDetection],
                ox: int, oy: int) -> List[FaceDetection]:
        """Translate face bboxes by offset (ox, oy)."""
        out = []
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            out.append(FaceDetection(
                bbox=(x1+ox, y1+oy, x2+ox, y2+oy),
                confidence=f.confidence,
                crop=f.crop,
            ))
        return out

    # ── draw helper ───────────────────────────────────────────────────────────

    @staticmethod
    def draw(roi: np.ndarray, faces: List[FaceDetection],
             offset: tuple = (0, 0)) -> np.ndarray:
        vis = roi.copy()
        ox, oy = offset
        for f in faces:
            x1, y1, x2, y2 = f.bbox
            cv2.rectangle(vis, (x1+ox, y1+oy), (x2+ox, y2+oy), (0, 100, 255), 2)
            cv2.putText(vis, f"face {f.confidence:.2f}",
                        (x1+ox, y1+oy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)
        return vis
