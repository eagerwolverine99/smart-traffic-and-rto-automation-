# ─────────────────────────────────────────────────────────────────────────────
# colab_demo.py  –  Google Colab / Jupyter-friendly demo
# Run this cell-by-cell in Colab.
# ─────────────────────────────────────────────────────────────────────────────

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 – Install dependencies                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# !pip install ultralytics opencv-python-headless numpy Pillow

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 – Imports & inline display helper                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import sys, os
sys.path.insert(0, "/content/driver_monitor")   # adjust if needed

import cv2
import numpy as np
from IPython.display import display, Image as IPImage
import io
from PIL import Image as PILImage


def show(frame_bgr: np.ndarray, title: str = ""):
    """Display a BGR frame inline in Colab."""
    rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img  = PILImage.fromarray(rgb)
    buf  = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    display(IPImage(data=buf.getvalue()))
    if title:
        print(title)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 – Upload a test image from your machine                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# from google.colab import files
# uploaded = files.upload()
# img_path = list(uploaded.keys())[0]

# ──  OR use a sample image from the web  ──────────────────────────────────
# import urllib.request
# urllib.request.urlretrieve("<your_image_url>", "test_car.jpg")
# img_path = "test_car.jpg"

img_path = "test_car.jpg"     # ← set this


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 – Run the pipeline                                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# patch config for headless Colab environment
import config
config.DISPLAY_WINDOW = False
config.SAVE_CROPS     = False

from preprocessing    import preprocess_frame
from vehicle_detector import VehicleDetector
from face_detector    import FaceDetector
from driver_extractor import DriverExtractor

vdet      = VehicleDetector()
fdet      = FaceDetector(backend="dnn")
extractor = DriverExtractor(vdet, fdet)

frame     = cv2.imread(img_path)
assert frame is not None, f"Could not read {img_path}"

frame     = preprocess_frame(frame)
results   = extractor.process(frame)
vis       = extractor.visualize(frame, results)

show(vis, f"{len(results)} driver(s) detected")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 – Show individual driver crops                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

for i, r in enumerate(results):
    print(f"Driver {i}: vehicle={r.vehicle.label}  conf={r.confidence:.2f}")
    show(r.driver_crop, f"Driver crop #{i}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 – Process a video file frame-by-frame                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# video_path = "traffic.mp4"    # ← set this
# cap        = cv2.VideoCapture(video_path)
# frame_no   = 0
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame_no += 1
#     if frame_no % 10 != 0:          # process every 10th frame
#         continue
#     frame   = preprocess_frame(frame)
#     results = extractor.process(frame)
#     vis     = extractor.visualize(frame, results)
#     show(vis, f"Frame {frame_no}  |  {len(results)} driver(s)")
#
# cap.release()
