# =============================================================================
#  tracker.py — Lightweight IoU tracker (no extra dependencies)
#  Assigns a stable Track ID to each vehicle across frames.
# =============================================================================

from __future__ import annotations
from typing import Dict, List, Tuple
from detector import Detection


class Tracker:
    """
    Greedy IoU tracker.
    Call update(detections) every frame → returns {bbox: track_id} mapping.
    """

    MAX_AGE   = 8     # frames a track survives with no match
    MIN_IOU   = 0.30  # minimum IoU to link detection → track

    def __init__(self):
        # track_id → [bbox, age, hits]
        self._tracks: Dict[int, list] = {}
        self._next = 1

    def update(self, detections: List[Detection]) -> Dict[tuple, int]:
        """
        Returns a dict   { bbox_tuple : track_id }
        for all current detections.
        """
        det_boxes = [d.bbox for d in detections]

        if not self._tracks:
            return self._register_all(det_boxes)

        t_ids   = list(self._tracks.keys())
        t_boxes = [self._tracks[i][0] for i in t_ids]

        matched_t = set()
        matched_d = set()
        result: Dict[tuple, int] = {}

        # build IoU matrix and greedily match
        iou_pairs = []
        for ti, tb in enumerate(t_boxes):
            for di, db in enumerate(det_boxes):
                iou = self._iou(tb, db)
                if iou >= self.MIN_IOU:
                    iou_pairs.append((iou, ti, di))
        iou_pairs.sort(reverse=True)

        for iou, ti, di in iou_pairs:
            if ti in matched_t or di in matched_d:
                continue
            tid = t_ids[ti]
            self._tracks[tid][0]  = det_boxes[di]   # update bbox
            self._tracks[tid][1]  = 0                # reset age
            self._tracks[tid][2] += 1                # increment hits
            result[det_boxes[di]] = tid
            matched_t.add(ti)
            matched_d.add(di)

        # age unmatched tracks, remove dead ones
        for ti, tid in enumerate(t_ids):
            if ti not in matched_t:
                self._tracks[tid][1] += 1
        self._tracks = {
            tid: v for tid, v in self._tracks.items()
            if v[1] <= self.MAX_AGE
        }

        # register new detections
        for di, db in enumerate(det_boxes):
            if di not in matched_d:
                tid = self._next
                self._tracks[tid] = [db, 0, 1]
                self._next += 1
                result[db] = tid

        return result

    def _register_all(self, boxes: List[tuple]) -> Dict[tuple, int]:
        result = {}
        for b in boxes:
            self._tracks[self._next] = [b, 0, 1]
            result[b] = self._next
            self._next += 1
        return result

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            return 0.0
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / (union + 1e-6)
