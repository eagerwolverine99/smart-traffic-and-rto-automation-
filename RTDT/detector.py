# =============================================================================
#  detector.py — Load RT-DETR (or YOLO fallback) and run vehicle detection
# =============================================================================

import os
import zipfile
import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import List
import config


# ---------------------------------------------------------------------------
#  Model loader — validates file integrity, auto-fallbacks
# ---------------------------------------------------------------------------

def _valid(path: str) -> bool:
    """True if path exists, is >1 MB, and is a valid zip (PyTorch format)."""
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < 1_000_000:
        return False
    try:
        with zipfile.ZipFile(path, "r"):
            return True
    except zipfile.BadZipFile:
        return False


def _wipe(name: str):
    """Delete every known cached copy of a model file."""
    locations = [
        name,
        os.path.join(os.path.dirname(__file__), name),
        os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Ultralytics", name),
        os.path.join(os.path.expanduser("~"), ".config", "Ultralytics", name),
    ]
    for p in locations:
        if os.path.isfile(p):
            try:
                os.remove(p)
                print(f"  [!] Removed corrupted file: {p}")
            except OSError:
                pass


def _resolve_device() -> str:
    """
    Return the best available device string for Ultralytics.
    Prints a full GPU diagnosis so you know exactly why GPU is/isn't used.
    """
    if config.DEVICE != "auto":
        print(f"[RTDT] Device forced to: {config.DEVICE}")
        return config.DEVICE

    print("[RTDT] Checking GPU availability ...")
    try:
        import torch
        print(f"[RTDT]   PyTorch version : {torch.__version__}")
        print(f"[RTDT]   CUDA built-in   : {torch.version.cuda}")
        print(f"[RTDT]   CUDA available  : {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                vram = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"[RTDT]   GPU [{i}]         : {name}  ({vram:.1f} GB VRAM)")
            print(f"[RTDT] ✓ Using GPU 0")
            return "0"
        else:
            # explain WHY cuda is unavailable
            if torch.version.cuda is None:
                print("[RTDT] ✗ PyTorch installed WITHOUT CUDA support.")
                print("[RTDT]   Fix: pip install torch torchvision --index-url "
                      "https://download.pytorch.org/whl/cu121")
            else:
                print("[RTDT] ✗ CUDA drivers not found or GPU not detected.")
                print("[RTDT]   Check: nvidia-smi  (run in terminal)")
    except ImportError:
        print("[RTDT] ✗ PyTorch not installed.  pip install torch")

    print("[RTDT] Running on CPU.")
    return "cpu"


def load_model():
    """
    Try to load RT-DETR first, then YOLO fallback.
    Validates zip integrity, wipes corrupted files, auto-selects device.
    """
    from ultralytics import YOLO

    device = _resolve_device()

    for name in [config.MODEL_NAME, config.FALLBACK]:
        for loc in [
            name,
            os.path.join(os.path.dirname(__file__), name),
            os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "Ultralytics", name),
        ]:
            if os.path.isfile(loc) and not _valid(loc):
                _wipe(name)
                break
        try:
            print(f"[RTDT] Loading {name} on device={device} ...")
            model = YOLO(name)
            model.to(device)
            print(f"[RTDT] ✓ Ready: {name}  device={device}")
            return model, name, device
        except Exception as e:
            print(f"[RTDT] ✗ {name} failed ({e.__class__.__name__}). Trying next ...")
            _wipe(name)

    raise RuntimeError("[RTDT] All models failed. Check internet & ultralytics install.")


# ---------------------------------------------------------------------------
#  Detection result
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    bbox:       tuple               # (x1, y1, x2, y2)
    label:      str
    confidence: float
    crop:       np.ndarray = field(repr=False, default=None)


# ---------------------------------------------------------------------------
#  Detector class
# ---------------------------------------------------------------------------

class Detector:

    def __init__(self):
        self.model, self.model_name, self.device = load_model()
        self.use_half = config.USE_HALF and self.device != "cpu"
        self.imgsz    = config.IMGSZ_GPU if self.device != "cpu" else config.IMGSZ_CPU
        if self.use_half:
            print("[RTDT] FP16 (half precision) enabled — 2× faster on GPU")
        print(f"[RTDT] Inference size: {self.imgsz}px")

    # ------------------------------------------------------------------ run

    def detect(self, frame: np.ndarray) -> List[Detection]:
        h, w = frame.shape[:2]
        results = self.model.predict(
            source  = frame,
            conf    = config.CONF,
            iou     = config.IOU,
            imgsz   = self.imgsz,
            classes = list(config.VEHICLES.keys()),
            device  = self.device,
            half    = self.use_half,      # FP16 on GPU, FP32 on CPU
            verbose = False,
        )[0]

        out = []
        for box in results.boxes:
            cid = int(box.cls[0])
            if cid not in config.VEHICLES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            out.append(Detection(
                bbox       = (x1, y1, x2, y2),
                label      = config.VEHICLES[cid],
                confidence = round(float(box.conf[0]), 3),
                crop       = frame[y1:y2, x1:x2].copy(),
            ))

        return sorted(out, key=lambda d: d.confidence, reverse=True)

    # ----------------------------------------------------------------- draw

    def draw(self, frame: np.ndarray,
             detections: List[Detection],
             tracks: dict = None) -> np.ndarray:
        """
        Draw boxes on frame.
        If `tracks` dict {bbox_tuple: track_id} is supplied,
        track IDs are shown in the label.
        """
        vis = frame.copy()

        for d in detections:
            color = config.COLORS.get(d.label, config.DEFAULT_COLOR)
            x1, y1, x2, y2 = d.bbox

            # -- box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            # -- corner ticks
            t = max(12, (x2 - x1) // 10)
            for px, py, sx, sy in [
                (x1, y1,  1,  1), (x2, y1, -1,  1),
                (x1, y2,  1, -1), (x2, y2, -1, -1),
            ]:
                cv2.line(vis, (px, py), (px + sx*t, py), color, 3)
                cv2.line(vis, (px, py), (px, py + sy*t), color, 3)

            # -- label
            tid   = tracks.get(d.bbox) if tracks else None
            label = f"{d.label}  {d.confidence:.0%}"
            if tid:
                label = f"#{tid} {label}"

            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            ty = max(0, y1 - th - 8)
            cv2.rectangle(vis, (x1, ty), (x1 + tw + 8, y1), color, -1)
            cv2.putText(vis, label, (x1 + 4, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

        return vis
