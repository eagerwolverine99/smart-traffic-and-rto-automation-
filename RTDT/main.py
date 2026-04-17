# =============================================================================
#  main.py — RTDT Entry Point
#
#  Usage:
#    python main.py                         # webcam
#    python main.py --source video.mp4      # video file
#    python main.py --source car.jpg        # single image
# =============================================================================

import argparse
import time
import threading
import queue
import os

import cv2
import numpy as np

import config
from detector import Detector
from tracker  import Tracker


# =============================================================================
#  Threaded camera reader  (keeps inference from blocking on frame capture)
# =============================================================================

_STREAM_END = object()   # unique sentinel — only placed in queue on real EOF


class Camera(threading.Thread):
    def __init__(self, source):
        super().__init__(daemon=True)
        self.cap        = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
        self._q         = queue.Queue(maxsize=4)   # bigger buffer
        self._stop      = threading.Event()
        self._fail_count = 0

    @property
    def opened(self):
        return self.cap.isOpened()

    def run(self):
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self._fail_count += 1
                if self._fail_count >= 5:
                    # 5 consecutive failures → real end of stream
                    self._q.put(_STREAM_END)
                    break
                time.sleep(0.05)   # brief wait then retry
                continue
            self._fail_count = 0
            if self._q.full():
                try:
                    self._q.get_nowait()   # drop oldest to stay fresh
                except queue.Empty:
                    pass
            self._q.put(frame)

    def read(self):
        """
        Returns:
          np.ndarray  → valid frame
          None        → no frame right now (timeout), caller should retry
          _STREAM_END → camera truly stopped
        """
        try:
            return self._q.get(timeout=0.1)
        except queue.Empty:
            return None          # just a hiccup, not end-of-stream

    def release(self):
        self._stop.set()
        self.cap.release()


# =============================================================================
#  FPS counter
# =============================================================================

class FPS:
    def __init__(self, window=30):
        self._t = []
        self._w = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._t.append(now)
        if len(self._t) > self._w:
            self._t.pop(0)
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0])


# =============================================================================
#  HUD overlay
# =============================================================================

def draw_hud(frame: np.ndarray, fps: float, count: int, model_name: str):
    h, w = frame.shape[:2]

    # top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # FPS
    if config.SHOW_FPS and fps > 0:
        cv2.putText(frame, f"FPS  {fps:.1f}",
                    (14, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # vehicle count
    vc = f"Vehicles: {count}"
    (tw, _), _ = cv2.getTextSize(vc, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, vc, (w - tw - 14, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2)

    # model + device badge bottom-right
    try:
        import torch
        dev = "GPU" if torch.cuda.is_available() else "CPU"
    except Exception:
        dev = "CPU"
    badge = f"{model_name}  |  {dev}"
    (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    badge_color = (0, 200, 100) if dev == "GPU" else (110, 110, 110)
    cv2.putText(frame, badge, (w - bw - 8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, badge_color, 1)


# =============================================================================
#  Single-image mode
# =============================================================================

def run_image(path: str, detector: Detector):
    frame = cv2.imread(path)
    if frame is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    detections = detector.detect(frame)
    vis        = detector.draw(frame, detections)
    draw_hud(vis, fps=0, count=len(detections), model_name=detector.model_name)

    print(f"\n[RTDT] {len(detections)} vehicle(s) found:")
    for d in detections:
        print(f"  {d.label:<12} {d.confidence:.0%}   bbox={d.bbox}")

    cv2.imshow(config.WINDOW_NAME, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# =============================================================================
#  Real-time stream mode
# =============================================================================

def run_stream(source, detector: Detector, tracker: Tracker):
    cam = Camera(source)
    if not cam.opened:
        raise RuntimeError(f"Cannot open source: {source}")
    cam.start()

    fps_counter = FPS()
    interval    = 1.0 / config.FPS_CAP

    # auto-select skip frames based on device
    skip = config.SKIP_FRAMES_GPU if detector.device != "cpu" else config.SKIP_FRAMES_CPU
    print(f"[RTDT] Frame skip: every {skip} frame(s)  |  imgsz: {detector.imgsz}px")

    # frame-skip cache
    last_detections = []
    last_track_map  = {}
    frame_idx       = 0

    # ── session statistics ────────────────────────────────────────────────────
    session_start   = time.perf_counter()
    total_frames    = 0
    peak_count      = 0                        # max vehicles seen in one frame
    class_totals    = {}                       # label → max tracks seen at once

    print(f"\n[RTDT] Stream started — press  Q  or  ESC  to quit.\n")

    try:
        while True:
            frame = cam.read()

            if frame is None:
                cv2.waitKey(1)
                continue

            if frame is _STREAM_END:
                print("[RTDT] Stream ended.")
                break

            t0 = time.perf_counter()

            if frame_idx % skip == 0:
                last_detections = detector.detect(frame)
                last_track_map  = tracker.update(last_detections)

                # update stats
                n = len(last_detections)
                if n > peak_count:
                    peak_count = n
                for d in last_detections:
                    class_totals[d.label] = class_totals.get(d.label, 0) + 1

            frame_idx  += 1
            total_frames += 1

            fps = fps_counter.tick()
            vis = detector.draw(frame, last_detections, tracks=last_track_map)
            draw_hud(vis, fps=fps, count=len(last_detections),
                     model_name=detector.model_name)

            cv2.imshow(config.WINDOW_NAME, vis)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

            elapsed = time.perf_counter() - t0
            if interval - elapsed > 0:
                time.sleep(interval - elapsed)

    finally:
        cam.release()
        cv2.destroyAllWindows()

        # ── session summary ───────────────────────────────────────────────────
        duration = time.perf_counter() - session_start
        avg_fps  = total_frames / duration if duration > 0 else 0

        print("\n" + "═" * 45)
        print("  RTDT  —  SESSION SUMMARY")
        print("═" * 45)
        print(f"  Duration      : {duration:.1f} sec")
        print(f"  Total frames  : {total_frames}")
        print(f"  Avg FPS       : {avg_fps:.1f}")
        print(f"  Peak vehicles : {peak_count}  (in a single frame)")
        print("─" * 45)
        if class_totals:
            print("  Vehicles detected by type:")
            for label, count in sorted(class_totals.items(),
                                       key=lambda x: x[1], reverse=True):
                bar = "█" * min(count // 10 + 1, 20)
                print(f"    {label:<14} {count:>5}x   {bar}")
        else:
            print("  No vehicles detected.")
        print("═" * 45 + "\n")


# =============================================================================
#  Entry point
# =============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RTDT – Real-Time Vehicle Detection")
    ap.add_argument("--source", default="0",
                    help="Webcam index (0, 1 …) OR path to video/image file")
    args = ap.parse_args()

    # resolve source type
    raw    = args.source
    source = int(raw) if raw.isdigit() else raw
    is_img = isinstance(source, str) and os.path.isfile(source) and \
             source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))

    # load models
    detector = Detector()
    tracker  = Tracker()

    if is_img:
        run_image(source, detector)
    else:
        run_stream(source, detector, tracker)
