"""
Real-Time ANPR System — v6 (PaddleOCR)
=======================================
Uses PaddleOCR instead of EasyOCR. PaddleOCR is specifically better at:
 - Character confusions (K/X, M/N/H, O/0, I/1) — much more accurate
 - Indian plate fonts
 - Low-resolution text
 - Angle/rotation handling (has built-in angle classifier)

Installation (one-time):
    install_paddleocr.bat         # Windows
    OR
    pip install paddlepaddle paddleocr   # manual

Then run:
    python main_v6.py --plate-model models\\license_plate_detector.pt --source "video.mp4"
"""

import cv2
import numpy as np
import time
import re
import argparse
import os
import csv
import traceback
from collections import defaultdict

from ultralytics import YOLO

# PaddleOCR replaces EasyOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("[WARNING] PaddleOCR not installed. Run install_paddleocr.bat first.")
    print("         Falling back to EasyOCR.")
    import easyocr

from strong_ocr import (
    PlateVoter, infer_plate_pattern,
    LETTER_TO_DIGIT, DIGIT_TO_LETTER,
    LETTER_CONFUSIONS, INDIAN_STATE_CODES, correct_state_code,
    deskew_plate, stretch_contrast, sharpen,
    remove_plate_border, morphology_cleanup,
    is_plausible_plate,
)
from iou_tracker import IOUTracker

# Plate info decoding + Excel logging
from plate_info import decode_plate

# Excel logging with plate info decoding
try:
    from excel_logger import ExcelLogger
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("[WARNING] excel_logger not available. Run: pip install openpyxl pillow")


class Config:
    VEHICLE_MODEL = "yolov8n.pt"
    VEHICLE_CLASSES = [2, 3, 5, 7]
    VEHICLE_CONF_THRESHOLD = 0.4
    OCR_CONF_THRESHOLD = 0.3

    YOLO_IMG_SIZE = 480

    OCR_EVERY_N_FRAMES = 3
    MAX_OCR_PER_FRAME = 3

    MAX_AGE = 20
    N_INIT = 2

    MIN_READINGS_TO_CONFIRM = 5
    CONFIRM_CONFIDENCE = 0.60

    MIN_VEHICLE_AREA = 6000

    ENFORCE_INDIAN_PLATE = True

    DISPLAY_MAX_WIDTH = 1280
    DISPLAY_MAX_HEIGHT = 720

    WINDOW_NAME = "ANPR v6 (PaddleOCR)"
    OUTPUT_DIR = "output"
    LOG_FILE = "detected_plates.csv"

    VERBOSE_OCR = False


# =====================================================================
# PaddleOCR wrapper — normalizes the interface to match what our code expects
# =====================================================================
class PaddleOCRWrapper:
    """Wraps PaddleOCR (3.x API) to match EasyOCR-style interface."""

    def __init__(self, use_gpu=False):
        print("[INIT] Loading PaddleOCR (first run downloads ~30MB of models)...")
        # PaddleOCR 3.x uses different parameter names than 2.x.
        # Drops use_gpu / show_log entirely; renames use_angle_cls,
        # det_db_thresh, det_db_box_thresh.
        self.ocr = PaddleOCR(
            use_textline_orientation=True,   # was use_angle_cls
            lang='en',
            text_det_thresh=0.3,              # was det_db_thresh
            text_det_box_thresh=0.5,          # was det_db_box_thresh
        )

    def readtext(self, image, **kwargs):
        """Return EasyOCR-style results: [(bbox, text, confidence), ...]

        PaddleOCR 3.x returns a different structure than 2.x, so try both.
        """
        try:
            # 3.x API: use .predict(); 2.x API: use .ocr()
            if hasattr(self.ocr, 'predict'):
                result = self.ocr.predict(image)
            else:
                result = self.ocr.ocr(image)

            if not result:
                return []

            formatted = []

            # 3.x format: result is a list of dicts with keys like
            # 'rec_texts', 'rec_scores', 'rec_polys' / 'dt_polys'
            first = result[0] if isinstance(result, list) else result

            if isinstance(first, dict):
                texts = first.get('rec_texts', [])
                scores = first.get('rec_scores', [])
                polys = first.get('rec_polys', first.get('dt_polys', []))
                for i, text in enumerate(texts):
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    bbox = polys[i] if i < len(polys) else None
                    formatted.append((bbox, text, conf))
                return formatted

            # 2.x format: list of [bbox, (text, confidence)]
            lines = result[0] if result else []
            if lines:
                for line in lines:
                    try:
                        bbox, (text, conf) = line
                        formatted.append((bbox, text, float(conf)))
                    except (ValueError, TypeError):
                        continue
            return formatted

        except Exception as e:
            # Don't spam errors on every frame; just return empty
            return []


# =====================================================================
# OCR preprocessing variants (same approach as strong_ocr, tuned for Paddle)
# =====================================================================
def generate_ocr_variants(plate_img):
    if plate_img is None or plate_img.size == 0:
        return []

    plate_img = deskew_plate(plate_img)

    h, w = plate_img.shape[:2]
    target_h = 140
    if h < target_h:
        scale = target_h / h
        plate_img = cv2.resize(plate_img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) \
        if len(plate_img.shape) == 3 else plate_img.copy()

    stretched = stretch_contrast(gray)
    sharp = sharpen(plate_img if len(plate_img.shape) == 3
                    else cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR))

    _, otsu = cv2.threshold(stretched, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_clean = morphology_cleanup(otsu)

    # PaddleOCR prefers color/3-channel images. Convert grayscale back to 3ch.
    variants = [
        ("sharp_color", sharp),
        ("original", plate_img),
        ("stretched_3ch", cv2.cvtColor(stretched, cv2.COLOR_GRAY2BGR)),
        ("otsu_3ch", cv2.cvtColor(otsu_clean, cv2.COLOR_GRAY2BGR)),
    ]
    return variants


ALLOWLIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'


def run_paddle_ocr(reader, plate_img, conf_threshold=0.3):
    """OCR the plate with multiple variants, return list of (text, confidences)."""
    variants = generate_ocr_variants(plate_img)
    all_readings = []

    for name, img in variants:
        try:
            results = reader.readtext(img)
            combined = ""
            combined_confs = []
            for (_, text, conf) in results:
                if conf < conf_threshold:
                    continue
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                combined += cleaned
                combined_confs.extend([conf] * len(cleaned))
            if is_plausible_plate(combined):
                all_readings.append((combined, combined_confs))
        except Exception:
            continue

    return all_readings


# =====================================================================
# Plate region finder (same as v5, for the contour fallback)
# =====================================================================
def find_plate_candidates(vehicle_crop, top_k=3):
    if vehicle_crop is None or vehicle_crop.size == 0:
        return []

    h, w = vehicle_crop.shape[:2]
    scale = 1.0
    if max(h, w) > 500:
        scale = 500 / max(h, w)
        small = cv2.resize(vehicle_crop, None, fx=scale, fy=scale)
    else:
        small = vehicle_crop

    sh, sw = small.shape[:2]
    lower_start = int(sh * 0.4)
    lower_crop = small[lower_start:, :]

    gray = cv2.cvtColor(lower_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(edged, cv2.RETR_TREE,
                                   cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

    candidates = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        if cw < 50 or ch < 15:
            continue
        aspect = cw / float(ch)
        if not (2.2 <= aspect <= 5.5):
            continue
        area = cw * ch
        if area < (sw * sh * 0.003):
            continue
        if area > (sw * sh * 0.15):
            continue
        abs_y = y + lower_start
        candidates.append((area, x, abs_y, x + cw, abs_y + ch))

    candidates.sort(key=lambda c: -c[0])
    candidates = candidates[:top_k]

    results = []
    for _, x1, y1, x2, y2 in candidates:
        if scale != 1.0:
            inv = 1 / scale
            x1, y1, x2, y2 = (int(x1 * inv), int(y1 * inv),
                              int(x2 * inv), int(y2 * inv))
        m = 3
        x1 = max(0, x1 - m)
        y1 = max(0, y1 - m)
        x2 = min(w, x2 + m)
        y2 = min(h, y2 + m)
        results.append((x1, y1, x2, y2))
    return results


# =====================================================================
# ANPR SYSTEM
# =====================================================================
class ANPR:
    def __init__(self, video_source=0, vehicle_model_path=None,
                 plate_model_path=None):
        self.video_source = video_source
        self.cfg = Config()

        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.log_path = os.path.join(self.cfg.OUTPUT_DIR, self.cfg.LOG_FILE)

        print("[INIT] Loading YOLO...")
        self.vehicle_model = YOLO(vehicle_model_path or self.cfg.VEHICLE_MODEL)

        if plate_model_path and os.path.exists(plate_model_path):
            print(f"[INIT] Loading plate model: {plate_model_path}")
            self.plate_model = YOLO(plate_model_path)
            self.use_plate_detector = True
        else:
            self.plate_model = None
            self.use_plate_detector = False
            print("[INIT] Using contour fallback")

        # Load PaddleOCR (preferred) or fall back to EasyOCR
        if PADDLE_AVAILABLE:
            self.reader = PaddleOCRWrapper(use_gpu=False)
            self.ocr_name = "PaddleOCR"
        else:
            print("[INIT] Loading EasyOCR (fallback)...")
            self.reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            self.ocr_name = "EasyOCR"

        print(f"[INIT] Using OCR: {self.ocr_name}")
        print("[INIT] Loading IOU tracker...")
        self.tracker = IOUTracker(
            max_age=self.cfg.MAX_AGE,
            n_init=self.cfg.N_INIT,
            iou_threshold=0.25,
        )

        self.voters = defaultdict(PlateVoter)
        self.confirmed_plates = {}
        self.logged_plates = set()
        self.confirmed_track_ids = set()
        self.plate_bboxes_cache = {}
        # NEW: remember each track's vehicle class and best plate image
        self.track_class_ids = {}    # track_id -> YOLO class_id
        self.track_best_plate_img = {}  # track_id -> best plate crop so far
        # NEW: cache decoded plate info per track for live overlay
        self.track_plate_info = {}   # track_id -> dict from decode_plate()

        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(["timestamp", "track_id", "plate",
                                    "confidence", "readings"])

        # NEW: Excel logger
        if EXCEL_AVAILABLE:
            try:
                self.excel = ExcelLogger(
                    output_dir=self.cfg.OUTPUT_DIR,
                    filename='anpr_log.xlsx'
                )
                print(f"[INIT] Excel log: {self.excel.filepath}")
            except Exception as e:
                print(f"[WARNING] Excel logger init failed: {e}")
                self.excel = None
        else:
            self.excel = None

    def detect_vehicles(self, frame):
        try:
            results = self.vehicle_model(
                frame, verbose=False,
                conf=self.cfg.VEHICLE_CONF_THRESHOLD,
                imgsz=self.cfg.YOLO_IMG_SIZE,
            )
        except Exception as e:
            print(f"[ERROR detect_vehicles] {e}")
            return []

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in self.cfg.VEHICLE_CLASSES:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append([x1, y1, x2, y2, conf, cls_id])
        return detections

    def _log_confirmed(self, track_id, plate, conf, num_readings):
        self.confirmed_plates[track_id] = plate
        self.confirmed_track_ids.add(track_id)

        # Decode plate info ONCE and cache for live overlay
        class_id = self.track_class_ids.get(track_id)
        info = decode_plate(plate, class_id)
        self.track_plate_info[track_id] = info

        if plate not in self.logged_plates:
            self.logged_plates.add(plate)
            ts = time.strftime('%Y-%m-%d %H:%M:%S')

            # Full info banner in terminal — user sees this INSTANTLY
            print("\n" + "=" * 60)
            print(f"[PLATE DETECTED] Track #{track_id}")
            print("=" * 60)
            print(f"  Plate Number : {plate}")
            print(f"  State        : {info['state']}")
            print(f"  RTO Code     : {info['rto_code']}")
            print(f"  City/Region  : {info['city']}")
            print(f"  Category     : {info['category']}")
            print(f"  Series       : {info['series']}")
            print(f"  Vehicle Type : {info['vehicle_type']}")
            print(f"  Confidence   : {conf:.1%}")
            print(f"  OCR Reads    : {num_readings}")
            print(f"  Time         : {ts}")
            print("=" * 60 + "\n")

            with open(self.log_path, 'a', newline='') as f:
                csv.writer(f).writerow([ts, track_id, plate,
                                        f"{conf:.3f}", num_readings])

            # Log to Excel with full plate info
            if self.excel is not None:
                try:
                    plate_img = self.track_best_plate_img.get(track_id)
                    self.excel.log_plate(
                        plate_text=plate,
                        track_id=track_id,
                        confidence=conf,
                        num_readings=num_readings,
                        plate_img=plate_img,
                        vehicle_class_id=class_id,
                    )
                except Exception as e:
                    print(f"[WARNING] Excel log failed: {e}")

    def process_ocr(self, frame, track_bbox, track_id):
        if track_id in self.confirmed_track_ids:
            return self.plate_bboxes_cache.get(track_id)

        try:
            x1, y1, x2, y2 = map(int, track_bbox)
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None

            area = (x2 - x1) * (y2 - y1)
            if area < self.cfg.MIN_VEHICLE_AREA:
                return None

            vehicle_crop = frame[y1:y2, x1:x2]
            best_plate_img = None
            best_plate_bbox_abs = None
            best_readings = []

            if self.use_plate_detector:
                results = self.plate_model(vehicle_crop, verbose=False,
                                            conf=0.4, imgsz=320)
                best_conf = 0
                for r in results:
                    if r.boxes is None:
                        continue
                    for box in r.boxes:
                        c = float(box.conf[0])
                        if c > best_conf:
                            best_conf = c
                            px1, py1, px2, py2 = map(int, box.xyxy[0])
                            best_plate_img = vehicle_crop[py1:py2, px1:px2]
                            best_plate_bbox_abs = (x1 + px1, y1 + py1,
                                                    x1 + px2, y1 + py2)

                # CRITICAL: reject plate crops too small to OCR reliably.
                # Below ~80px wide or ~25px tall, EVERY OCR model produces junk.
                # Better to wait until the vehicle gets closer than poison the
                # voter with bad early readings.
                if best_plate_img is not None and best_plate_img.size > 0:
                    ph, pw = best_plate_img.shape[:2]
                    MIN_PLATE_WIDTH = 80
                    MIN_PLATE_HEIGHT = 25
                    if pw < MIN_PLATE_WIDTH or ph < MIN_PLATE_HEIGHT:
                        # Plate too small — skip OCR, keep the bbox for display
                        if self.cfg.VERBOSE_OCR:
                            print(f"  [SKIP track#{track_id}] plate too small: "
                                  f"{pw}x{ph}px (need {MIN_PLATE_WIDTH}x"
                                  f"{MIN_PLATE_HEIGHT}+)")
                        return best_plate_bbox_abs

                    best_readings = run_paddle_ocr(
                        self.reader, best_plate_img,
                        self.cfg.OCR_CONF_THRESHOLD)
            else:
                candidates = find_plate_candidates(vehicle_crop, top_k=3)
                best_score = 0
                for (px1, py1, px2, py2) in candidates:
                    plate_img = vehicle_crop[py1:py2, px1:px2]
                    if plate_img is None or plate_img.size == 0:
                        continue
                    readings = run_paddle_ocr(
                        self.reader, plate_img,
                        self.cfg.OCR_CONF_THRESHOLD)
                    score = 0
                    for text, confs in readings:
                        if 6 <= len(text) <= 11:
                            avg = sum(confs) / len(confs) if confs else 0
                            has_l = any(c.isalpha() for c in text)
                            has_d = any(c.isdigit() for c in text)
                            if has_l and has_d:
                                s = avg * len(text)
                                if s > score:
                                    score = s
                    if score > best_score:
                        best_score = score
                        best_plate_img = plate_img
                        best_plate_bbox_abs = (x1 + px1, y1 + py1,
                                                x1 + px2, y1 + py2)
                        best_readings = readings

            if best_plate_bbox_abs is not None:
                self.plate_bboxes_cache[track_id] = best_plate_bbox_abs

            # Remember the biggest plate crop we've seen for this track
            # (for Excel snapshot)
            if best_plate_img is not None and best_plate_img.size > 0:
                prev = self.track_best_plate_img.get(track_id)
                if prev is None or best_plate_img.size > prev.size:
                    # Copy to detach from frame buffer
                    self.track_best_plate_img[track_id] = best_plate_img.copy()

            if self.cfg.VERBOSE_OCR and best_readings:
                # Show plate size alongside readings so user can diagnose
                if best_plate_img is not None:
                    ph, pw = best_plate_img.shape[:2]
                    print(f"  [OCR track#{track_id}] plate={pw}x{ph}px "
                          f"readings={[r[0] for r in best_readings]}")

            # Compute plate-size weight: bigger crop = more reliable reading.
            # This is the key improvement — far-away plate readings get small
            # weight in the voter, close-up plate readings dominate.
            if best_plate_img is not None and best_plate_img.size > 0:
                ph, pw = best_plate_img.shape[:2]
                plate_area = pw * ph
                # Normalize so a 200x50 plate (10000px) = weight 1.0
                # A 80x25 plate (2000px) = weight 0.2
                size_weight = min(1.0, plate_area / 10000.0)
            else:
                size_weight = 0.5

            voter = self.voters[track_id]
            for text, confs in best_readings:
                if len(text) < 6 or len(text) > 11:
                    continue
                if not (any(c.isalpha() for c in text)
                        and any(c.isdigit() for c in text)):
                    continue
                if self.cfg.ENFORCE_INDIAN_PLATE and len(text) not in (9, 10):
                    continue
                # Scale confidence by plate size weight
                weighted_confs = [c * size_weight for c in confs]
                voter.add_reading(text, weighted_confs)

            if voter.expected_pattern is None:
                if self.cfg.ENFORCE_INDIAN_PLATE:
                    voter.expected_pattern = ['L', 'L', 'D', 'D',
                                              'L', 'L', 'D', 'D', 'D', 'D']
                elif len(voter.readings) >= 3:
                    voter.expected_pattern = infer_plate_pattern(
                        [r[0] for r in voter.readings]
                    )

            best, conf = voter.get_best_plate(
                min_readings=self.cfg.MIN_READINGS_TO_CONFIRM
            )
            if best and conf >= self.cfg.CONFIRM_CONFIDENCE:
                self._log_confirmed(track_id, best, conf,
                                     len(voter.readings))

            return best_plate_bbox_abs

        except Exception as e:
            print(f"[ERROR process_ocr track#{track_id}] {e}")
            if self.cfg.VERBOSE_OCR:
                traceback.print_exc()
            return None

    def draw_overlay(self, frame, track, plate_bbox):
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        tid = track.track_id

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        plate_text = self.confirmed_plates.get(tid, "")
        n_reads = len(self.voters[tid].readings) if tid in self.voters else 0

        if plate_text:
            # Full info overlay — multiple lines above the vehicle box
            info = self.track_plate_info.get(tid, {})
            lines = [
                f"ID:{tid}  |  {plate_text}",
                f"{info.get('state', 'Unknown')}",
                f"{info.get('city', 'Unknown')}",
                f"{info.get('vehicle_type', 'Vehicle')} | {info.get('category', 'Private')}",
            ]
            self._draw_multiline_label(frame, x1, y1, lines,
                                        bg_color=(0, 170, 0))
            box_color = (0, 200, 0)
        elif n_reads > 0:
            label = f"ID:{tid} | reading ({n_reads})"
            self._draw_single_label(frame, x1, y1, label,
                                     bg_color=(0, 165, 255))
            box_color = (0, 165, 255)
        else:
            label = f"ID:{tid}"
            self._draw_single_label(frame, x1, y1, label,
                                     bg_color=(0, 255, 0))
            box_color = (0, 255, 0)

        if plate_bbox is not None:
            px1, py1, px2, py2 = map(int, plate_bbox)
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)

    def _draw_single_label(self, frame, x, y, text, bg_color=(0, 255, 0)):
        """Draw a single-line label above position (x, y)."""
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y), bg_color, -1)
        cv2.putText(frame, text, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_multiline_label(self, frame, x, y, lines, bg_color=(0, 170, 0)):
        """Draw a multi-line info box above (x, y)."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1
        line_height = 22
        pad = 6

        # Measure max text width
        max_w = 0
        for line in lines:
            (tw, _), _ = cv2.getTextSize(line, font, font_scale, thickness)
            max_w = max(max_w, tw)

        box_w = max_w + pad * 2
        box_h = line_height * len(lines) + pad * 2

        top = y - box_h
        if top < 0:
            # If not enough room above the car, draw inside/below
            top = y + 5

        # Solid background
        cv2.rectangle(frame, (x, top), (x + box_w, top + box_h),
                      bg_color, -1)
        # Subtle dark border for readability
        cv2.rectangle(frame, (x, top), (x + box_w, top + box_h),
                      (0, 80, 0), 1)

        for i, line in enumerate(lines):
            ty = top + pad + line_height * (i + 1) - 6
            # First line bold (plate number) - draw slightly brighter
            color = (255, 255, 255) if i == 0 else (230, 255, 230)
            cv2.putText(frame, line, (x + pad, ty),
                        font, font_scale, color, thickness, cv2.LINE_AA)

    def draw_hud(self, frame, fps):
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Plates: {len(self.logged_plates)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 0), 2)
        cv2.putText(frame, f"OCR: {self.ocr_name}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 2)

    def run(self):
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open: {self.video_source}")
            return

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale_w = self.cfg.DISPLAY_MAX_WIDTH / src_w if src_w > 0 else 1.0
        scale_h = self.cfg.DISPLAY_MAX_HEIGHT / src_h if src_h > 0 else 1.0
        self.display_scale = min(1.0, scale_w, scale_h)
        disp_w = int(src_w * self.display_scale)
        disp_h = int(src_h * self.display_scale)
        print(f"[INFO] Source: {src_w}x{src_h} -> Display: {disp_w}x{disp_h}")

        cv2.namedWindow(self.cfg.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.cfg.WINDOW_NAME, disp_w, disp_h)

        print("[RUN] Press 'q' to quit, 's' to snapshot.")

        frame_count = 0
        fps_timer = time.time()
        fps = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                detections = self.detect_vehicles(frame)
                ds_dets = [
                    ([x1, y1, x2 - x1, y2 - y1], conf, str(cls_id))
                    for x1, y1, x2, y2, conf, cls_id in detections
                ]
                tracks = self.tracker.update_tracks(ds_dets)

                # Remember each track's class_id for Excel logging
                for t in tracks:
                    if hasattr(t, 'class_id') and t.class_id is not None:
                        try:
                            self.track_class_ids[t.track_id] = int(t.class_id)
                        except (ValueError, TypeError):
                            pass

                should_ocr = (frame_count % self.cfg.OCR_EVERY_N_FRAMES == 0)
                ocr_candidates = []
                if should_ocr:
                    for t in tracks:
                        if not t.is_confirmed(self.cfg.N_INIT):
                            continue
                        if t.track_id in self.confirmed_track_ids:
                            continue
                        bbox = t.to_ltrb()
                        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                        ocr_candidates.append((area, t))
                    ocr_candidates.sort(key=lambda x: -x[0])
                    ocr_candidates = ocr_candidates[:self.cfg.MAX_OCR_PER_FRAME]

                ocr_track_ids = {t.track_id for _, t in ocr_candidates}

                for track in tracks:
                    if not track.is_confirmed(self.cfg.N_INIT):
                        continue
                    tid = track.track_id
                    plate_bbox = None
                    if tid in ocr_track_ids:
                        plate_bbox = self.process_ocr(frame, track.to_ltrb(), tid)
                    else:
                        plate_bbox = self.plate_bboxes_cache.get(tid)
                    self.draw_overlay(frame, track, plate_bbox)

                self.draw_hud(frame, fps)

                if frame_count % 10 == 0:
                    elapsed = time.time() - fps_timer
                    fps = 10 / elapsed if elapsed > 0 else 0
                    fps_timer = time.time()

                if self.display_scale < 1.0:
                    display_frame = cv2.resize(
                        frame, None,
                        fx=self.display_scale, fy=self.display_scale,
                        interpolation=cv2.INTER_AREA)
                else:
                    display_frame = frame

                cv2.imshow(self.cfg.WINDOW_NAME, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    snap = os.path.join(self.cfg.OUTPUT_DIR,
                                        f"snapshot_{int(time.time())}.jpg")
                    cv2.imwrite(snap, frame)
                    print(f"[SAVE] {snap}")

        except Exception as e:
            print(f"[FATAL] {e}")
            traceback.print_exc()
        finally:
            cap.release()
            cv2.destroyAllWindows()

            print("\n" + "=" * 60)
            print("SESSION SUMMARY")
            print("=" * 60)
            print(f"OCR engine: {self.ocr_name}")
            print(f"Unique plates: {len(self.logged_plates)}")
            for p in sorted(self.logged_plates):
                print(f"  * {p}")
            print(f"\nCSV log: {self.log_path}")
            if self.excel is not None:
                self.excel.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0")
    parser.add_argument("--vehicle-model", type=str, default=None)
    parser.add_argument("--plate-model", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    system = ANPR(
        video_source=source,
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
    )
    if args.verbose:
        system.cfg.VERBOSE_OCR = True
    system.run()


if __name__ == "__main__":
    main()
