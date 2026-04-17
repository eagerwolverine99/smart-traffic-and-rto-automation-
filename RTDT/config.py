# =============================================================================
#  config.py — RTDT (Real-Time Detection Transformer) Settings
# =============================================================================

# --- Model -------------------------------------------------------------------
#  GPU : "rtdetr-l.pt" → ~60 FPS  |  "rtdetr-x.pt" → ~45 FPS (most accurate)
#  CPU : auto-falls back to FALLBACK model
MODEL_NAME  = "rtdetr-l.pt"
FALLBACK    = "yolov8n.pt"

CONF        = 0.40
IOU         = 0.50

# COCO class IDs to detect
VEHICLES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck",
    1: "Bicycle",
}

# --- GPU / Device ------------------------------------------------------------
DEVICE      = "auto"       # "auto" | "cpu" | "cuda" | "0"
USE_HALF    = True         # FP16 on GPU (2× faster) — auto-disabled on CPU

# --- Performance (auto-selected by device) -----------------------------------
IMGSZ_GPU       = 640      # full resolution on GPU
IMGSZ_CPU       = 320      # half resolution on CPU — ~3× faster

SKIP_FRAMES_GPU = 1        # infer every frame on GPU
SKIP_FRAMES_CPU = 5        # infer every 5th frame on CPU → smooth video playback

# --- Display -----------------------------------------------------------------
WINDOW_NAME = "RTDT - Vehicle Detection"
FRAME_W     = 1280
FRAME_H     = 720
FPS_CAP     = 60
SHOW_FPS    = True

# BGR colours per class
COLORS = {
    "Car":        (0,   220,   0),
    "Motorcycle": (0,   165, 255),
    "Bus":        (255,  80,   0),
    "Truck":      (0,   100, 255),
    "Bicycle":    (180,   0, 255),
}
DEFAULT_COLOR = (200, 200, 200)
