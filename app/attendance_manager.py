import json
import os
import threading
import time
from typing import Dict, Optional


class AttendanceManager:
    def __init__(self, storage_path: str, flush_interval_seconds: float = 5.0) -> None:
        self.storage_path = storage_path
        self.flush_interval_seconds = flush_interval_seconds
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Dict[str, object]]] = {}
        self._last_flush = 0.0
        self._load()

    def mark_present(
        self,
        person_id: str,
        name: str,
        role: str,
        camera_id: str,
        timestamp: Optional[float] = None,
    ) -> None:
        if timestamp is None:
            timestamp = time.time()
        date_key = time.strftime("%Y-%m-%d", time.localtime(timestamp))
        with self._lock:
            day = self._data.setdefault(date_key, {})
            entry = day.get(person_id)
            if entry is None:
                entry = {
                    "person_id": person_id,
                    "name": name,
                    "role": role,
                    "first_seen": timestamp,
                    "last_seen": timestamp,
                    "camera_id": camera_id,
                    "seen_count": 1,
                }
                day[person_id] = entry
            else:
                entry["last_seen"] = timestamp
                entry["seen_count"] = int(entry.get("seen_count", 0)) + 1
                entry["camera_id"] = camera_id
            self._flush_if_needed_locked()

    def get_attendance(self, date_key: str) -> Dict[str, Dict[str, object]]:
        with self._lock:
            return dict(self._data.get(date_key, {}))

    def _flush_if_needed_locked(self) -> None:
        now = time.time()
        if now - self._last_flush < self.flush_interval_seconds:
            return
        self._persist_locked()
        self._last_flush = now

    def _persist_locked(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        tmp_path = f"{self.storage_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self._data, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self.storage_path)

    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                self._data = payload
        except Exception:
            return
