import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from zoneinfo import ZoneInfo


@dataclass
class ScheduleEntry:
    start_ts: float
    end_ts: float
    teacher_id: Optional[str]


class ScheduleManager:
    def __init__(self, path: str, tz_name: str) -> None:
        self.path = path
        self.tz = ZoneInfo(tz_name)
        self._lock = threading.Lock()
        self._mtime: Optional[float] = None
        self._entries: Dict[str, List[ScheduleEntry]] = {}

    def is_teacher_present(self, room_id: str, ts: float) -> bool:
        entry = self._find_entry(room_id, ts)
        return entry is not None

    def current_teacher(self, room_id: str, ts: float) -> Optional[str]:
        entry = self._find_entry(room_id, ts)
        return entry.teacher_id if entry else None

    def _find_entry(self, room_id: str, ts: float) -> Optional[ScheduleEntry]:
        self._load_if_needed()
        with self._lock:
            entries = self._entries.get(room_id, [])
            for entry in entries:
                if entry.start_ts <= ts <= entry.end_ts:
                    return entry
        return None

    def _load_if_needed(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            mtime = os.path.getmtime(self.path)
        except OSError:
            return
        with self._lock:
            if self._mtime is not None and self._mtime >= mtime:
                return
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        loaded: Dict[str, List[ScheduleEntry]] = {}
        for room_id, entries in payload.items():
            if not isinstance(room_id, str) or not isinstance(entries, list):
                continue
            parsed: List[ScheduleEntry] = []
            for item in entries:
                if not isinstance(item, dict):
                    continue
                start = _parse_local(item.get("start"), self.tz)
                end = _parse_local(item.get("end"), self.tz)
                if start is None or end is None:
                    continue
                teacher_id = item.get("teacher_id")
                parsed.append(
                    ScheduleEntry(
                        start_ts=start,
                        end_ts=end,
                        teacher_id=teacher_id if isinstance(teacher_id, str) else None,
                    )
                )
            loaded[room_id] = parsed
        with self._lock:
            self._entries = loaded
            try:
                self._mtime = os.path.getmtime(self.path)
            except OSError:
                self._mtime = None


def _parse_local(raw: object, tz: ZoneInfo) -> Optional[float]:
    if not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt)
            dt = dt.replace(tzinfo=tz)
            return dt.timestamp()
        except ValueError:
            continue
    return None
