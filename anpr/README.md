# ANPR — Real-Time Automatic Number Plate Recognition

A real-time Indian license plate recognition system built with **YOLOv8**, **PaddleOCR**, and **OpenCV**. Detects vehicles, locates license plates, runs OCR with majority voting across multiple frames, and decodes plate metadata (state, RTO code, city, vehicle type).

Logs results to CSV and Excel with plate snapshots.

---

## Features

- **Vehicle detection** with YOLOv8 (cars, motorcycles, buses, trucks)
- **Plate detection** via dedicated YOLO model, with contour-based fallback
- **Multi-frame OCR voting** — averages readings across frames to fight character confusions (K/X, M/N, O/0, etc.)
- **Plate-size-weighted confidence** — close-up reads count more than far-away ones
- **Indian plate decoding** — state, RTO code, city, vehicle category
- **IOU-based tracker** — lightweight, no DeepSORT dependency
- **Multi-backend video opener** — auto-falls-back through FFMPEG / MSMF / DSHOW on Windows
- **Excel logging** with plate crop snapshots embedded in cells
- **CSV log** appended across runs
- Live HUD with FPS, plate count, OCR engine in use

---

## Project structure

```
anpr/
├── main_v6.py              # Main entry point
├── strong_ocr.py           # Plate voter, character confusion correction, preprocessing
├── iou_tracker.py          # Lightweight IOU-based tracker (DeepSORT replacement)
├── plate_info.py           # Indian plate decoder (state/RTO/category)
├── excel_logger.py         # Excel output with embedded plate snapshots
├── yolov8n.pt              # Vehicle detection model (auto-downloaded)
├── models/
│   └── license_plate_detector.pt   # Plate detection model (you provide)
├── vehicle.mp4             # Sample input video
├── output/                 # Generated logs and snapshots
│   ├── detected_plates.csv
│   ├── anpr_log.xlsx
│   └── plate_snapshots/
├── anpr.bat                # Windows convenience runner (optional)
├── requirements.txt
└── README.md
```

---

## Requirements

- **Python 3.10 or 3.11** (PaddleOCR has issues with 3.12+)
- **Windows / Linux / macOS**
- **8 GB RAM** minimum, 16 GB recommended
- **GPU optional** (everything runs fine on CPU)
- A working **webcam** or video file

---

## Installation

### Step 1 — Open a terminal in the project folder

Open your terminal (PowerShell, cmd, bash, etc.) and `cd` into the folder containing `main_v6.py`. All commands below assume your current directory is the project root.

```
cd path/to/anpr
```

### Step 2 — (Recommended) Create a virtual environment

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Linux / macOS / Git Bash:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies

**Option A: Install everything in one line**

```
python -m pip install --upgrade pip
python -m pip install opencv-python numpy ultralytics openpyxl pillow paddlepaddle paddleocr
```

**Option B: From requirements.txt**

```
pip install -r requirements.txt
```

**Option C: Use anpr.bat (Windows only)**

```powershell
.\anpr.bat install
```

### Step 4 — Get the YOLO models

The vehicle model (`yolov8n.pt`) downloads automatically on first run.

The plate detector (`license_plate_detector.pt`) is **not included** — you need to provide it. Place it at:

```
models/license_plate_detector.pt
```

(relative to the project root — i.e. the same folder as `main_v6.py`)

If you don't have one, the script falls back to a contour-based detector (less accurate but still functional). Sources for plate detector models:

- Train your own with YOLOv8 on a plate dataset
- Find community-trained models on GitHub (search "yolov8 license plate")
- Use Roboflow Universe datasets

### Step 5 — Verify install

```
python -c "import cv2, ultralytics, paddleocr; print('All imports OK')"
```

---

## Running

### Quick start

```
python main_v6.py --source vehicle.mp4 --plate-model models/license_plate_detector.pt
```

> On Windows you can use either forward or back slashes — `models/license_plate_detector.pt` and `models\license_plate_detector.pt` both work.

### Run on webcam

```
python main_v6.py --source 0 --plate-model models/license_plate_detector.pt
```

### Run without plate detector (contour fallback)

```
python main_v6.py --source vehicle.mp4
```

### Run with verbose OCR logging

```
python main_v6.py --source vehicle.mp4 --plate-model models/license_plate_detector.pt --verbose
```

### Using the Windows runner (optional)

```powershell
.\anpr.bat run                       # default: vehicle.mp4
.\anpr.bat run-cam                   # webcam
.\anpr.bat run-verbose               # verbose mode
.\anpr.bat run other_video.mp4       # custom source
```

### Keyboard controls (during run)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save snapshot of current frame to `output/` |

---

## Command-line arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Video file path or webcam index (`0`, `1`, ...) |
| `--vehicle-model` | `yolov8n.pt` | Path to vehicle detection YOLO model |
| `--plate-model` | None | Path to plate detection YOLO model (recommended) |
| `--verbose` | off | Print per-frame OCR diagnostics |

---

## Output files

All outputs land in `output/`:

- **`detected_plates.csv`** — appended across runs. Columns: timestamp, track_id, plate, confidence, num_readings
- **`anpr_log.xlsx`** — Excel log with plate snapshots embedded in cells, plus decoded state/RTO/city/category
- **`plate_snapshots/`** — individual JPEG crops of each detected plate
- **`snapshot_<timestamp>.jpg`** — full-frame screenshots when you press `s`

---

## Configuration

Tunables live in the `Config` class at the top of `main_v6.py`:

| Setting | Default | What it controls |
|---------|---------|------------------|
| `VEHICLE_CONF_THRESHOLD` | 0.4 | Minimum YOLO confidence for vehicle detection |
| `OCR_CONF_THRESHOLD` | 0.3 | Minimum OCR confidence to accept a character |
| `OCR_EVERY_N_FRAMES` | 3 | Run OCR every N frames (lower = more accurate, slower) |
| `MAX_OCR_PER_FRAME` | 3 | Cap on number of vehicles to OCR per frame |
| `MIN_READINGS_TO_CONFIRM` | 5 | Need this many readings before confirming a plate |
| `CONFIRM_CONFIDENCE` | 0.60 | Minimum voted confidence to confirm a plate |
| `MIN_VEHICLE_AREA` | 6000 | Skip vehicles smaller than this (pixels²) |
| `ENFORCE_INDIAN_PLATE` | True | Require LLDDLLDDDD format |
| `YOLO_IMG_SIZE` | 480 | Inference resolution (smaller = faster) |

---

## Troubleshooting

### Script exits immediately with no error

Almost always a video codec issue. The patched `main_v6.py` tries multiple OpenCV backends (FFMPEG, MSMF, DSHOW). If all fail, re-encode your video:

```
ffmpeg -i input.mp4 -c:v libx264 -pix_fmt yuv420p -c:a copy fixed.mp4
```

### `[INIT] Using contour fallback`

Means `--plate-model` wasn't found. Confirm the file exists:

```powershell
# Windows
dir models\license_plate_detector.pt
```

```bash
# Linux / macOS
ls models/license_plate_detector.pt
```

### PaddleOCR install fails

PaddleOCR has Python version constraints. If `pip install paddlepaddle paddleocr` fails:

1. Check your Python version (`python --version`) — use 3.10 or 3.11
2. Fall back to EasyOCR: `pip install easyocr`
   The script auto-detects which one is available.

### Plate snapshot too small to OCR

The script skips plates smaller than 80×25 px. Move the camera closer or zoom in for better recognition.

### Plate read keeps showing wrong characters

Edit `strong_ocr.py` and check `LETTER_CONFUSIONS`. Common Indian-plate confusions:
- `K` ↔ `X`
- `M` ↔ `N` ↔ `H`
- `O` ↔ `0`
- `I` ↔ `1`
- `B` ↔ `8`

The voter corrects most of these automatically given enough readings.

### "Out of memory" errors

Lower `YOLO_IMG_SIZE` from 480 to 320 in the `Config` class, and increase `OCR_EVERY_N_FRAMES`.

---

## How it works

1. **Capture** — read frame from video/webcam
2. **Detect vehicles** — YOLOv8n classifies cars/motorcycles/buses/trucks
3. **Track** — IOU tracker assigns persistent IDs across frames
4. **Detect plates** — YOLO plate detector finds plate within each vehicle crop (or contour fallback)
5. **Preprocess** — deskew, contrast stretch, sharpen, Otsu threshold (4 variants)
6. **OCR** — PaddleOCR reads each variant; confident reads added to per-track voter
7. **Vote** — once 5+ readings collected and weighted confidence ≥ 0.60, plate is confirmed
8. **Decode** — extract state/RTO/city from plate prefix
9. **Log** — write to CSV + Excel with snapshot

---

## Performance tips

- **Use a GPU** — install `paddlepaddle-gpu` instead of `paddlepaddle` for ~5× OCR speedup
- **Lower YOLO image size** — `YOLO_IMG_SIZE = 320` for ~2× speedup, small accuracy hit
- **Increase frame skip** — `OCR_EVERY_N_FRAMES = 5` if your stream is fast
- **Reduce candidates** — `MAX_OCR_PER_FRAME = 1` for very crowded scenes if you only care about closest vehicle

---

## Acknowledgments

- **Ultralytics YOLOv8** — vehicle and plate detection
- **PaddleOCR** — text recognition
- **OpenCV** — video I/O and image processing

---

## License

Add your license here (MIT / Apache-2.0 / etc.)
