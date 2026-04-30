"""
Simple IOU-based Tracker
========================
Drop-in replacement for DeepSORT for ANPR use cases.
Why: DeepSORT's re-identification network is overkill for tracking cars
between adjacent frames (they don't teleport). IOU matching is:
 - 5-10x faster
 - Zero model loading
 - Zero dependencies (pure numpy)
 - Good enough when frame rate > 5 FPS (vehicles move < 1 car length/frame)

Interface mirrors DeepSORT's `update_tracks()` so the main loop stays simple.
"""

import numpy as np
from collections import deque


def compute_iou(box_a, box_b):
    """IOU between two [x1,y1,x2,y2] boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h
    area_a = max(0, box_a[2] - box_a[0]) * max(0, box_a[3] - box_a[1])
    area_b = max(0, box_b[2] - box_b[0]) * max(0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


class Track:
    """Represents one tracked object."""
    def __init__(self, track_id, bbox, class_id, confidence):
        self.track_id = track_id
        self.bbox = bbox            # [x1, y1, x2, y2]
        self.class_id = class_id
        self.confidence = confidence
        self.hits = 1               # consecutive matched frames
        self.time_since_update = 0  # frames since last match
        # Simple velocity estimation for motion prediction
        self.history = deque(maxlen=3)
        self.history.append(bbox)

    def predict(self):
        """Predict next bbox position using simple linear velocity."""
        if len(self.history) < 2:
            return self.bbox
        prev = self.history[-2]
        curr = self.history[-1]
        dx1 = curr[0] - prev[0]
        dy1 = curr[1] - prev[1]
        dx2 = curr[2] - prev[2]
        dy2 = curr[3] - prev[3]
        return [curr[0] + dx1, curr[1] + dy1,
                curr[2] + dx2, curr[3] + dy2]

    def update(self, bbox, confidence):
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox)

    def mark_missed(self):
        self.time_since_update += 1

    def is_confirmed(self, n_init):
        return self.hits >= n_init

    def to_ltrb(self):
        return self.bbox


class IOUTracker:
    """
    Minimal tracker. Mimics DeepSORT's interface.

    update_tracks(detections, frame=None) where
        detections = [([x, y, w, h], conf, class_str), ...]
    returns a list of Track objects.
    """

    def __init__(self, max_age=15, n_init=2, iou_threshold=0.3):
        self.max_age = max_age
        self.n_init = n_init
        self.iou_threshold = iou_threshold
        self.tracks = []
        self._next_id = 1

    def update_tracks(self, detections, frame=None):
        # Convert detections to [x1,y1,x2,y2] format
        det_boxes = []
        for (ltwh, conf, cls) in detections:
            x, y, w, h = ltwh
            det_boxes.append(([x, y, x + w, y + h], conf, cls))

        # Predict next position for each track (motion compensation)
        predicted_boxes = [t.predict() for t in self.tracks]

        # Build IOU matrix: tracks x detections
        n_tracks = len(self.tracks)
        n_dets = len(det_boxes)
        matched_tracks = set()
        matched_dets = set()

        if n_tracks > 0 and n_dets > 0:
            iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
            for i, pbox in enumerate(predicted_boxes):
                for j, (dbox, _, _) in enumerate(det_boxes):
                    iou_matrix[i, j] = compute_iou(pbox, dbox)

            # Greedy match: take highest-IOU pairs first
            while True:
                if iou_matrix.size == 0:
                    break
                idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                best_iou = iou_matrix[idx]
                if best_iou < self.iou_threshold:
                    break
                ti, di = idx
                if ti in matched_tracks or di in matched_dets:
                    iou_matrix[ti, di] = 0
                    continue
                # Match
                dbox, dconf, _ = det_boxes[di]
                self.tracks[ti].update(dbox, dconf)
                matched_tracks.add(ti)
                matched_dets.add(di)
                # Zero out this track and detection row/col
                iou_matrix[ti, :] = 0
                iou_matrix[:, di] = 0

        # Unmatched tracks → mark missed
        for i, t in enumerate(self.tracks):
            if i not in matched_tracks:
                t.mark_missed()

        # Unmatched detections → spawn new tracks
        for j, (dbox, dconf, dcls) in enumerate(det_boxes):
            if j not in matched_dets:
                self.tracks.append(
                    Track(self._next_id, dbox, dcls, dconf)
                )
                self._next_id += 1

        # Remove stale tracks
        self.tracks = [t for t in self.tracks
                       if t.time_since_update <= self.max_age]

        return self.tracks
