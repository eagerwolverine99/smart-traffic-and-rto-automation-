# Smart Traffic & RTO Automation

An AI-powered platform for traffic monitoring and automated RTO (Regional Transport Office) operations. It combines real-time vehicle detection, license plate recognition, driver behavior monitoring, and a live web dashboard backed by a MySQL database.

Built around computer vision (YOLOv8 + PaddleOCR), a vanilla-JS dashboard, and a relational schema for storing detections, congestion stats, alerts, and reports.

---

## Components

The repository is organized into three loosely-coupled modules that share a common database schema.

### 1. `anpr/` — Automatic Number Plate Recognition

Real-time Indian license plate recognition pipeline.

- Detects vehicles (cars, motorcycles, buses, trucks) with **YOLOv8**
- Locates plates with a dedicated YOLO plate detector (with contour-based fallback)
- Reads plates with **PaddleOCR**, voting across multiple frames to fight character confusions (K↔X, M↔N, O↔0, etc.)
- Decodes plate metadata: state, RTO code, city, vehicle category
- Logs results to CSV and Excel with embedded plate snapshots
- Lightweight IOU tracker (no DeepSORT dependency)

See [`anpr/README.md`](anpr/README.md) for detailed setup and usage.

### 2. `driver_monitor/` — Driver Behavior Monitoring

AI-based driver monitoring system for detecting unsafe driving.

- Real-time detection of driver state from in-cabin camera feeds
- Classifies driver behavior (attentive, drowsy, distracted, etc.)
- Logs incidents for downstream RTO action

See [`driver_monitor/README.md`](driver_monitor/README.md) (or the module's own docs) for setup details.

### 3. Web Dashboard — `index.html`, `script.js`, `styles.css`

Live monitoring dashboard for traffic operations centers.

- **Real-time traffic flow chart** (last 60 seconds)
- **Vehicle distribution donut** by category
- **Live camera feeds** (4-tile grid)
- **Zone congestion** indicators with severity levels
- **System metrics** — model name, FPS, accuracy, frames processed
- **Recent detections table** (filterable + searchable)
- **Hourly bar chart** for the day
- **Weekly density heatmap** by hour and day

Built with vanilla HTML/CSS/JS + Chart.js. No build step required.

### 4. `database.sql` — MySQL schema

Shared schema for all components. Defines:

- `cameras`, `zones`, `vehicle_types` — reference data
- `detections` — every vehicle the AI sees
- `traffic_stats`, `daily_reports` — aggregated counts
- `congestion_log` — zone congestion snapshots
- `users` — RTO officer / admin accounts with role-based access (`admin`, `officer`, `viewer`)
- `alerts` — system notifications (high congestion, camera offline, etc.)
- `system_log` — model performance tracking

Plus three convenience views: `v_live_zone_counts`, `v_today_hourly`, `v_zone_congestion`.

---

## Architecture

```
       ┌──────────────────┐         ┌──────────────────┐
       │  Camera streams  │────────▶│  ANPR pipeline   │
       │  (RTSP / file /  │         │  (YOLOv8 +       │
       │   webcam)        │         │   PaddleOCR)     │
       └──────────────────┘         └────────┬─────────┘
                                             │
       ┌──────────────────┐                  │  detections,
       │  In-cabin cams   │                  │  plates,
       │   (driver feed)  │──────┐           │  vehicle types
       └──────────────────┘      │           │
                                 ▼           ▼
                        ┌────────────────────────────┐
                        │   driver_monitor pipeline  │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────┐
                        │     MySQL (database.sql)   │
                        │  cameras / zones /         │
                        │  detections / alerts /     │
                        │  traffic_stats / reports   │
                        └────────────┬───────────────┘
                                     │
                                     ▼
                        ┌────────────────────────────┐
                        │   Web dashboard            │
                        │   (index.html + Chart.js)  │
                        └────────────────────────────┘
```

---

## Quick start

### Prerequisites

- **Python 3.10 or 3.11** (PaddleOCR has issues with 3.12+)
- **MySQL 8.0+** (or MariaDB 10.5+)
- **Modern web browser** (Chrome, Firefox, Edge)
- **Webcam** or video file for testing
- **Optional GPU** for faster inference

### 1. Clone the repo

```
git clone https://github.com/eagerwolverine99/smart-traffic-and-rto-automation-.git
cd smart-traffic-and-rto-automation-
```

### 2. Set up the database

```
mysql -u root -p < database.sql
```

This creates the `rto_automation` database with sample zones, cameras, and an admin user. Replace the placeholder password hash in production.

### 3. Set up the ANPR pipeline

```
cd anpr
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
python main_v6.py --source vehicle.mp4 --plate-model models/license_plate_detector.pt
```

See [`anpr/README.md`](anpr/README.md) for full instructions, including how to provide the plate detector model.

### 4. Set up the driver monitor

```
cd ../driver_monitor
# follow the README inside that folder
```

### 5. Open the dashboard

Open `index.html` in your browser, or serve it through any static HTTP server:

```
# Python's built-in server (run from the repo root)
python -m http.server 8000
```

Then visit http://localhost:8000.

> **Note:** The dashboard currently reads from in-page JavaScript demo data. To wire it to the live MySQL database, you'll need a small backend API (Flask / FastAPI / Express) that exposes `/api/detections`, `/api/zones`, etc. — that's not included in this repo yet.

---

## Repository structure

```
smart-traffic-and-rto-automation-/
├── anpr/                        # Number plate recognition module
│   ├── main_v6.py
│   ├── strong_ocr.py
│   ├── iou_tracker.py
│   ├── plate_info.py
│   ├── excel_logger.py
│   ├── requirements.txt
│   ├── anpr.bat                 # Windows convenience runner
│   └── README.md
│
├── driver_monitor/              # Driver behavior monitoring
│   └── ...
│
├── index.html                   # Web dashboard
├── script.js                    # Dashboard logic + Chart.js
├── styles.css                   # Dashboard styling
│
├── database.sql                 # MySQL schema + seed data
├── .gitignore
├── LICENSE                      # MIT
└── README.md                    # This file
```

---

## Tech stack

| Layer | Technology |
|---|---|
| Vehicle / plate detection | YOLOv8 (Ultralytics) |
| OCR | PaddleOCR (with EasyOCR fallback) |
| Computer vision | OpenCV |
| Tracker | Custom IOU tracker (lightweight) |
| Driver monitoring | YOLOv8 + computer vision heuristics |
| Database | MySQL 8 |
| Frontend | HTML5 + vanilla JS + Chart.js |
| Logging | CSV + Excel (openpyxl) |

---

## Roadmap

- [ ] REST/WebSocket API to bridge MySQL ↔ dashboard
- [ ] Authentication for the dashboard (matching the `users` table)
- [ ] Dockerization for reproducible deployment
- [ ] RTSP stream support out of the box
- [ ] Auto-aggregation jobs to fill `traffic_stats` and `daily_reports`
- [ ] Mobile-responsive dashboard layout

---

## Contributing

Contributions, bug reports, and feature requests are welcome. Please open an issue or submit a pull request.

---

## License

[MIT](LICENSE) © 2026 eagerwolverine99
