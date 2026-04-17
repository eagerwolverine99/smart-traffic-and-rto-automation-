# ─────────────────────────────────────────────────────────────────────────────
# database.py  –  SQLite persistence for driver events
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import sqlite3
import datetime
import numpy as np
import cv2
from typing import Optional

DB_PATH   = os.path.join(os.path.dirname(__file__), "driver_events.db")
CROP_DIR  = os.path.join(os.path.dirname(__file__), "output", "db_crops")
os.makedirs(CROP_DIR, exist_ok=True)


class DriverDatabase:
    """
    Stores each driver detection event with:
      - timestamp
      - vehicle type & confidence
      - driver name (if recognised)
      - path to saved face crop
      - drowsiness flag
    """

    DDL = """
    CREATE TABLE IF NOT EXISTS driver_events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT    NOT NULL,
        vehicle     TEXT,
        veh_conf    REAL,
        driver_name TEXT,
        face_conf   REAL,
        crop_path   TEXT,
        drowsy      INTEGER DEFAULT 0
    );
    """

    def __init__(self, db_path: str = DB_PATH):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(self.DDL)
        self._conn.commit()

    # ── write ─────────────────────────────────────────────────────────────────

    def log_event(self,
                  vehicle: str,
                  veh_conf: float,
                  driver_name: str  = "unknown",
                  face_conf: float  = 0.0,
                  crop: Optional[np.ndarray] = None,
                  drowsy: bool = False) -> int:
        """Insert one row and return the new row id."""
        ts        = datetime.datetime.now().isoformat()
        crop_path = None

        if crop is not None and crop.size > 0:
            fname      = f"event_{ts.replace(':','-')}.jpg"
            crop_path  = os.path.join(CROP_DIR, fname)
            cv2.imwrite(crop_path, crop)

        cur = self._conn.execute(
            """INSERT INTO driver_events
               (timestamp, vehicle, veh_conf, driver_name, face_conf, crop_path, drowsy)
               VALUES (?,?,?,?,?,?,?)""",
            (ts, vehicle, veh_conf, driver_name, face_conf, crop_path, int(drowsy))
        )
        self._conn.commit()
        return cur.lastrowid

    # ── read ──────────────────────────────────────────────────────────────────

    def recent_events(self, limit: int = 20):
        """Return the most recent `limit` events as a list of dicts."""
        cur  = self._conn.execute(
            "SELECT * FROM driver_events ORDER BY id DESC LIMIT ?", (limit,)
        )
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self):
        self._conn.close()
