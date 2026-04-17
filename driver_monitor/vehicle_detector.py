# ─────────────────────────────────────────────────────────────────────────────
# vehicle_detector.py  –  YOLOv8-based vehicle detection
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List

from config import (YOLO_MODEL, YOLO_CONF, YOLO_IOU,
                    VEHICLE_CLASSES, DRIVER_SIDE, DRIVER_ZONE_X)


@dataclass
class VehicleDetection:
    bbox: tuple          # (x1, y1, x2, y2) in pixel coords
    class_id: int
    label: str
    confidence: float
    crop: np.ndarray = field(repr=False, default=None)   # BGR crop of vehicle
    driver_zone: tuple  = field(default=None)            # (x1,y1,x2,y2) inside crop


class VehicleDetector:
    """Wraps Ultralytics YOLOv8 for vehicle-only detection."""

    def __init__(self,
                 model_path: str = YOLO_MODEL,
                 conf: float     = YOLO_CONF,
                 iou: float      = YOLO_IOU):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Run:  pip install ultralytics")

        self.model    = YOLO(model_path)
        self.conf     = conf
        self.iou      = iou

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[VehicleDetection]:
        """
        Run inference on a BGR frame.
        Returns a list of VehicleDetection (sorted largest → smallest area).
        """
        results = self.model.predict(
            source   = frame,
            conf     = self.conf,
            iou      = self.iou,
            classes  = list(VEHICLE_CLASSES.keys()),
            verbose  = False,
        )[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf_score      = float(box.conf[0])

            # clamp to frame bounds
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            crop         = frame[y1:y2, x1:x2].copy()
            driver_zone  = self._driver_zone(crop)

            detections.append(VehicleDetection(
                bbox        = (x1, y1, x2, y2),
                class_id    = cls_id,
                label       = VEHICLE_CLASSES[cls_id],
                confidence  = conf_score,
                crop        = crop,
                driver_zone = driver_zone,
            ))

        # largest vehicle first (most likely to be the subject)
        detections.sort(key=lambda d: (d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]),
                        reverse=True)
        return detections

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _driver_zone(crop: np.ndarray) -> tuple:
        """
        Return the bounding box of the driver-side window region inside `crop`.

        Heuristic:
          - Top half of the crop (windscreen / window level).
          - Left half  → left-hand drive  (DRIVER_SIDE = "left").
          - Right half → right-hand drive (DRIVER_SIDE = "right").
        """
        h, w = crop.shape[:2]
        zone_w = int(w * DRIVER_ZONE_X)

        if DRIVER_SIDE == "left":
            return (0, 0, zone_w, h)              # x1, y1, x2, y2
        else:
            return (w - zone_w, 0, w, h)

    def draw(self, frame: np.ndarray,
             detections: List[VehicleDetection]) -> np.ndarray:
        """Overlay detection bounding boxes on `frame` (in-place copy)."""
        vis = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 0), 2)
            label = f"{d.label} {d.confidence:.2f}"
            cv2.putText(vis, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        return vis
