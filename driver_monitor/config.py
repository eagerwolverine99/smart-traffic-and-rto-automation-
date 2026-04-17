# ─────────────────────────────────────────────────────────────────────────────
# config.py  –  Central configuration for the Driver Monitoring Pipeline
# ─────────────────────────────────────────────────────────────────────────────

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
HAAR_CASCADE    = os.path.join(MODEL_DIR, "haarcascade_frontalface_default.xml")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR,  exist_ok=True)

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"          # nano = fastest; swap to yolov8s/m for accuracy
YOLO_CONF       = 0.45                  # minimum detection confidence
YOLO_IOU        = 0.45                  # NMS IoU threshold
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ── Face detection backend  (choose: "haar" | "dnn" | "retinaface") ───────────
FACE_BACKEND    = "dnn"                 # recommended default
DNN_PROTO       = os.path.join(MODEL_DIR, "deploy.prototxt")
DNN_MODEL       = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")
DNN_CONF        = 0.35   # lowered: angled/side-profile faces score 0.35-0.50

# ── Driver-side heuristic ─────────────────────────────────────────────────────
# In most countries the driver sits on the LEFT half of the vehicle crop.
# Set to "right" if targeting right-hand-drive regions (UK, India, AU …).
DRIVER_SIDE     = "left"               # "left" | "right"
DRIVER_ZONE_X   = 0.55                 # fraction of vehicle crop width used as driver zone

# ── Preprocessing ─────────────────────────────────────────────────────────────
FRAME_WIDTH     = 1280
FRAME_HEIGHT    = 720
PREPROCESS_SIZE = (300, 300)           # resize before DNN inference
LOW_LIGHT_THRESH = 60                  # mean brightness below this → apply CLAHE

# ── Camera mode ───────────────────────────────────────────────────────────────
# CABIN_MODE = True  → camera is already inside the vehicle (dashcam / webcam
#   pointing at driver).  Skips YOLO vehicle detection entirely and runs face
#   detection directly on the full frame.
# CABIN_MODE = False → external traffic camera; expects a vehicle in frame first.
CABIN_MODE      = True

# ── Real-time capture ─────────────────────────────────────────────────────────
WEBCAM_INDEX    = 0
TARGET_FPS      = 30

# ── Output ────────────────────────────────────────────────────────────────────
SAVE_CROPS      = True                 # persist driver crops to OUTPUT_DIR
DISPLAY_WINDOW  = True
SHOW_FPS        = True
