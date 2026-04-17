# ─────────────────────────────────────────────────────────────────────────────
# pipeline.py  –  Real-time processing loop (threaded, FPS-capped)
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import time
import threading
import queue
import numpy as np
import cv2

from config import (WEBCAM_INDEX, TARGET_FPS, DISPLAY_WINDOW, SHOW_FPS,
                    FRAME_WIDTH, FRAME_HEIGHT)
from preprocessing  import preprocess_frame
from driver_extractor import DriverExtractor


class FrameReader(threading.Thread):
    """
    Background thread that reads frames from the capture device / file so the
    main thread is never blocked on I/O.
    """

    def __init__(self, source, maxsize: int = 4):
        super().__init__(daemon=True)
        self.cap     = cv2.VideoCapture(source)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        self._q      = queue.Queue(maxsize=maxsize)
        self._stop   = threading.Event()

    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                self._q.put(None)           # signal end-of-stream
                break
            if self._q.full():
                try:
                    self._q.get_nowait()    # drop stale frame
                except queue.Empty:
                    pass
            self._q.put(frame)

    def read(self, timeout: float = 1.0):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._stop.set()
        self.cap.release()

    @property
    def is_opened(self):
        return self.cap.isOpened()


# ─────────────────────────────────────────────────────────────────────────────

class DriverMonitoringPipeline:
    """
    Entry point for both real-time and file-based processing.

    Usage:
        pipeline = DriverMonitoringPipeline()
        pipeline.run(source=0)          # webcam
        pipeline.run(source="clip.mp4") # video file
        pipeline.run(source="photo.jpg", single_image=True)
    """

    def __init__(self):
        self.extractor = DriverExtractor()
        self._interval = 1.0 / TARGET_FPS

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, source=WEBCAM_INDEX, single_image: bool = False):
        if single_image:
            self._run_image(source)
        else:
            self._run_video(source)

    def process_frame(self, frame: np.ndarray):
        """Expose single-frame processing for integration into other modules."""
        frame = preprocess_frame(frame)
        return self.extractor.process(frame), frame

    # ── internal loops ────────────────────────────────────────────────────────

    def _run_image(self, path: str):
        frame = cv2.imread(path)
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {path}")

        results, processed = self.process_frame(frame)
        vis = self.extractor.visualize(processed, results)

        print(f"[Pipeline] {len(results)} driver(s) found in image.")
        for i, r in enumerate(results):
            print(f"  [{i}] vehicle={r.vehicle.label}  "
                  f"conf={r.confidence:.2f}  "
                  f"face_bbox={r.abs_face_bbox}")

        if DISPLAY_WINDOW:
            cv2.imshow("Driver Monitor – Image", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _run_video(self, source):
        reader = FrameReader(source)
        if not reader.is_opened:
            raise RuntimeError(f"Cannot open source: {source}")
        reader.start()

        fps_counter = _FPSCounter()

        try:
            while True:
                t0    = time.perf_counter()
                frame = reader.read()

                if frame is None:
                    print("[Pipeline] Stream ended.")
                    break

                results, vis_frame = self.process_frame(frame)
                vis = self.extractor.visualize(vis_frame, results)

                if SHOW_FPS:
                    fps = fps_counter.tick()
                    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                if DISPLAY_WINDOW:
                    cv2.imshow("Driver Monitor", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:   # q or ESC
                        break

                # FPS cap: sleep if we're running too fast
                elapsed = time.perf_counter() - t0
                sleep_t = self._interval - elapsed
                if sleep_t > 0:
                    time.sleep(sleep_t)

        finally:
            reader.stop()
            cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────

class _FPSCounter:
    def __init__(self, window: int = 30):
        self._times: list = []
        self._window      = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Driver Monitoring Pipeline")
    parser.add_argument("--source",  default=str(WEBCAM_INDEX),
                        help="webcam index (0,1…) OR path to video/image file")
    parser.add_argument("--image",   action="store_true",
                        help="treat --source as a static image")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    DriverMonitoringPipeline().run(source=source, single_image=args.image)
