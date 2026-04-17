# ─────────────────────────────────────────────────────────────────────────────
# preprocessing.py  –  Frame conditioning before model inference
# ─────────────────────────────────────────────────────────────────────────────

import cv2
import numpy as np
from config import LOW_LIGHT_THRESH, FRAME_WIDTH, FRAME_HEIGHT


def resize_frame(frame: np.ndarray,
                 width: int = FRAME_WIDTH,
                 height: int = FRAME_HEIGHT) -> np.ndarray:
    """Resize to a fixed resolution while preserving aspect ratio (letterbox)."""
    h, w = frame.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    x_off  = (width  - new_w) // 2
    y_off  = (height - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization on the L channel."""
    lab   = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    enhanced = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def denoise(frame: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Light Gaussian blur to suppress sensor / compression noise."""
    return cv2.GaussianBlur(frame, (ksize, ksize), 0)


def is_low_light(frame: np.ndarray, threshold: int = LOW_LIGHT_THRESH) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Full preprocessing pipeline applied to every incoming frame:
      1. Resize / letterbox
      2. Denoise
      3. CLAHE if low-light
    Returns the conditioned frame (BGR).
    """
    frame = resize_frame(frame)
    frame = denoise(frame)
    if is_low_light(frame):
        frame = apply_clahe(frame)
    return frame


def preprocess_roi(roi: np.ndarray, target: tuple = (300, 300)) -> np.ndarray:
    """Resize a region-of-interest for DNN face detection input blob."""
    return cv2.resize(roi, target, interpolation=cv2.INTER_LINEAR)


def normalize(frame: np.ndarray) -> np.ndarray:
    """Return float32 frame normalized to [0, 1]."""
    return frame.astype(np.float32) / 255.0
