# ─────────────────────────────────────────────────────────────────────────────
# face_recognizer.py  –  Lightweight face recognition using face_recognition lib
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import pickle
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple

ENCODINGS_FILE = os.path.join(os.path.dirname(__file__), "models", "face_encodings.pkl")


class FaceRecognizer:
    """
    1-vs-N face recognition.

    Workflow:
      1. enroll(name, face_crops)  — add a driver to the database
      2. identify(face_crop)       — return (name, distance) or ("unknown", 1.0)

    Requires:  pip install face_recognition
    (which requires cmake + dlib)
    """

    TOLERANCE = 0.55          # L2 distance threshold; lower = stricter

    def __init__(self):
        try:
            import face_recognition
            self._fr = face_recognition
        except ImportError:
            raise ImportError(
                "Run:  pip install face_recognition\n"
                "(requires cmake and dlib – see https://github.com/ageitgey/face_recognition)"
            )

        self._db: Dict[str, List[np.ndarray]] = {}   # name → list of encodings
        self._load()

    # ── enrolment ─────────────────────────────────────────────────────────────

    def enroll(self, name: str, crops: List[np.ndarray]):
        """
        Add a driver by name.  Provide multiple face crops for robustness.
        """
        encodings = []
        for crop in crops:
            rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            encs = self._fr.face_encodings(rgb)
            encodings.extend(encs)

        if not encodings:
            print(f"[FaceRecognizer] No faces found in enrolment crops for '{name}'.")
            return

        self._db.setdefault(name, []).extend(encodings)
        self._save()
        print(f"[FaceRecognizer] Enrolled '{name}' with {len(encodings)} encoding(s).")

    # ── identification ────────────────────────────────────────────────────────

    def identify(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Returns (name, distance).  name = "unknown" if no match.
        """
        if not self._db:
            return "unknown", 1.0

        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        encs = self._fr.face_encodings(rgb)
        if not encs:
            return "unknown", 1.0

        query = encs[0]
        best_name, best_dist = "unknown", self.TOLERANCE

        for name, db_encs in self._db.items():
            dists = self._fr.face_distance(db_encs, query)
            d     = float(dists.min())
            if d < best_dist:
                best_dist = d
                best_name = name

        return best_name, best_dist

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self):
        os.makedirs(os.path.dirname(ENCODINGS_FILE), exist_ok=True)
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(self._db, f)

    def _load(self):
        if os.path.exists(ENCODINGS_FILE):
            with open(ENCODINGS_FILE, "rb") as f:
                self._db = pickle.load(f)
            print(f"[FaceRecognizer] Loaded {len(self._db)} enrolled driver(s).")
