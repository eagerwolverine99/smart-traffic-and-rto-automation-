# ─────────────────────────────────────────────────────────────────────────────
# driver_extractor.py  –  Combines vehicle + face detection into driver crops
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import time
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List, Optional

from config import DRIVER_SIDE, OUTPUT_DIR, SAVE_CROPS, CABIN_MODE
from vehicle_detector import VehicleDetector, VehicleDetection
from face_detector import FaceDetector, FaceDetection


@dataclass
class DriverResult:
    vehicle: VehicleDetection
    face: Optional[FaceDetection]          # None if no face found
    driver_crop: np.ndarray = field(repr=False)   # best available crop
    confidence: float       = 0.0
    # absolute bbox on original frame
    abs_face_bbox: Optional[tuple] = None


class DriverExtractor:
    """
    High-level pipeline:
      frame → vehicle detection → driver zone → face detection → DriverResult
    """

    def __init__(self,
                 vehicle_detector: Optional[VehicleDetector] = None,
                 face_detector: Optional[FaceDetector]       = None):
        self.vdet = vehicle_detector or VehicleDetector()
        self.fdet = face_detector    or FaceDetector()
        self._frame_idx = 0

    # ── main pipeline ─────────────────────────────────────────────────────────

    def process(self, frame: np.ndarray) -> List[DriverResult]:
        """
        Run the full pipeline on one BGR frame.
        Returns one DriverResult per detected vehicle (may be empty list).

        In CABIN_MODE the vehicle detection step is skipped — face detection
        runs directly on the full frame (use when webcam / dashcam is already
        inside the vehicle pointing at the driver).
        """
        results = []

        vehicles = self.vdet.detect(frame) if not CABIN_MODE else []

        if vehicles:
            # External camera — car found → use vehicle pipeline
            for vehicle in vehicles:
                result = self._extract_driver(frame, vehicle)
                if result is not None:
                    results.append(result)
                    if SAVE_CROPS:
                        self._save_crop(result)
        else:
            # No car detected (cabin cam OR YOLO missed) → face detect on full frame
            result = self._extract_driver_cabin(frame)
            if result is not None:
                results.append(result)
                if SAVE_CROPS:
                    self._save_crop(result)

        self._frame_idx += 1
        return results

    # ── cabin-mode: no vehicle detection needed ───────────────────────────────

    def _extract_driver_cabin(self, frame: np.ndarray) -> Optional[DriverResult]:
        """
        Direct face detection on the full frame.
        Used when the camera is already inside the vehicle (webcam / dashcam).
        Creates a synthetic VehicleDetection covering the entire frame so the
        rest of the pipeline (visualise, save, etc.) stays identical.
        """
        h, w = frame.shape[:2]

        # Synthetic vehicle covering full frame
        synthetic_vehicle = VehicleDetection(
            bbox        = (0, 0, w, h),
            class_id    = 2,
            label       = "cabin",
            confidence  = 1.0,
            crop        = frame.copy(),
            driver_zone = (0, 0, w, h),
        )

        faces = self.fdet.detect(frame)
        best_face = self._pick_best_face(faces, frame.shape)

        if best_face is not None:
            fx1, fy1, fx2, fy2 = best_face.bbox
            abs_bbox    = (fx1, fy1, fx2, fy2)
            driver_crop = best_face.crop
            confidence  = best_face.confidence
        else:
            abs_bbox    = None
            driver_crop = frame.copy()
            confidence  = 0.0           # no face found

        return DriverResult(
            vehicle       = synthetic_vehicle,
            face          = best_face,
            driver_crop   = driver_crop,
            confidence    = confidence,
            abs_face_bbox = abs_bbox,
        )

    # ── per-vehicle driver extraction ─────────────────────────────────────────

    def _extract_driver(self,
                        frame: np.ndarray,
                        vehicle: VehicleDetection) -> Optional[DriverResult]:
        crop  = vehicle.crop
        if crop is None or crop.size == 0:
            return None

        dz        = vehicle.driver_zone            # (x1,y1,x2,y2) in crop coords
        zone_crop = crop[dz[1]:dz[3], dz[0]:dz[2]]
        if zone_crop.size == 0:
            return None

        faces = self.fdet.detect(zone_crop)

        # ── fallback: if no face in driver zone, search full vehicle crop ─────
        # Handles cases where the camera angle puts the driver on the
        # "wrong" side of the heuristic split (e.g. passenger-side camera).
        if not faces:
            faces_full = self.fdet.detect(crop)
            if faces_full:
                # Re-anchor face coords relative to driver-zone origin (0,0)
                # by subtracting the driver zone offset so translation below works.
                # We treat the full crop as if dz offset = (0,0).
                dz = (0, 0, crop.shape[1], crop.shape[0])
                zone_crop = crop
                faces = faces_full

        # ── pick the "most driver-like" face ─────────────────────────────────
        best_face = self._pick_best_face(faces, zone_crop.shape)

        if best_face is not None:
            # translate face bbox back to absolute frame coordinates
            vx1, vy1 = vehicle.bbox[0], vehicle.bbox[1]
            dzx1, dzy1 = dz[0], dz[1]
            fx1, fy1, fx2, fy2 = best_face.bbox
            abs_bbox = (
                vx1 + dzx1 + fx1,
                vy1 + dzy1 + fy1,
                vx1 + dzx1 + fx2,
                vy1 + dzy1 + fy2,
            )
            driver_crop = best_face.crop
            confidence  = best_face.confidence
        else:
            # Fallback: return the full driver zone crop with low confidence
            abs_bbox    = None
            driver_crop = zone_crop.copy()
            confidence  = vehicle.confidence * 0.4

        return DriverResult(
            vehicle      = vehicle,
            face         = best_face,
            driver_crop  = driver_crop,
            confidence   = confidence,
            abs_face_bbox= abs_bbox,
        )

    @staticmethod
    def _pick_best_face(faces: List[FaceDetection],
                        zone_shape: tuple) -> Optional[FaceDetection]:
        """
        From all detected faces inside the driver zone, pick the most likely
        driver face:
          1. Prefer the face closest to the steering-wheel side (left/right).
          2. Among ties, prefer highest confidence.
        """
        if not faces:
            return None
        if len(faces) == 1:
            return faces[0]

        zone_w = zone_shape[1]
        # score = confidence + proximity bonus toward driving side
        def _score(f: FaceDetection) -> float:
            cx   = (f.bbox[0] + f.bbox[2]) / 2.0
            prox = 1.0 - (cx / zone_w) if DRIVER_SIDE == "left" else (cx / zone_w)
            return f.confidence * 0.7 + prox * 0.3

        return max(faces, key=_score)

    # ── persistence ───────────────────────────────────────────────────────────

    def _save_crop(self, result: DriverResult):
        ts   = int(time.time() * 1000)
        name = f"driver_{self._frame_idx:06d}_{ts}.jpg"
        path = os.path.join(OUTPUT_DIR, name)
        cv2.imwrite(path, result.driver_crop)

    # ── visualisation ─────────────────────────────────────────────────────────

    def visualize(self, frame: np.ndarray,
                  results: List[DriverResult]) -> np.ndarray:
        """
        Draw all detections on a copy of `frame`.
        Green box  = vehicle
        Orange box = driver zone
        Blue box   = face
        """
        vis = frame.copy()

        for r in results:
            vx1, vy1, vx2, vy2 = r.vehicle.bbox

            # vehicle box — skip in cabin mode (it wraps the whole frame, looks wrong)
            if not CABIN_MODE:
                cv2.rectangle(vis, (vx1, vy1), (vx2, vy2), (0, 220, 0), 2)
                cv2.putText(vis, f"{r.vehicle.label} {r.vehicle.confidence:.2f}",
                            (vx1, max(12, vy1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

                # driver zone divider (only meaningful for external camera)
                dz = r.vehicle.driver_zone
                dz_abs = (vx1 + dz[0], vy1 + dz[1], vx1 + dz[2], vy1 + dz[3])
                cv2.rectangle(vis,
                              (dz_abs[0], dz_abs[1]),
                              (dz_abs[2], dz_abs[3]),
                              (0, 165, 255), 1)

            # face box — always shown
            if r.abs_face_bbox is not None:
                fx1, fy1, fx2, fy2 = r.abs_face_bbox
                status = "DRIVER DETECTED" if CABIN_MODE else "driver"
                color  = (0, 200, 255)
                # draw rounded-feel thick box
                cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), color, 3)
                label  = f"{status}  {r.confidence:.2f}"
                # background pill for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis, (fx1, fy1 - th - 12), (fx1 + tw + 6, fy1), color, -1)
                cv2.putText(vis, label, (fx1 + 3, fy1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            else:
                # No face found — show red warning
                cv2.putText(vis, "NO DRIVER DETECTED", (20, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        if not results:
            cv2.putText(vis, "SCANNING...", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        return vis
