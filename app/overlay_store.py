import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class OverlayBuffer:
    events: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=5000))
    last_flush: float = 0.0


class OverlayStore:
    def __init__(
        self,
        root_path: str,
        retention_seconds: float = 60.0,
        flush_interval_seconds: float = 1.0,
        person_conf_threshold: float = 0.7,
        object_conf_threshold: float = 0.5,
        disk_retention_seconds: float = 86400.0,
        cleanup_interval_seconds: float = 60.0,
    ) -> None:
        self.root_path = root_path
        self.retention_seconds = retention_seconds
        self.flush_interval_seconds = flush_interval_seconds
        self.disk_retention_seconds = max(0.0, disk_retention_seconds)
        self.cleanup_interval_seconds = max(5.0, cleanup_interval_seconds)
        self._buffers: Dict[str, Dict[str, OverlayBuffer]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._person_conf_threshold = max(0.0, min(1.0, person_conf_threshold))
        self._object_conf_threshold = max(0.0, min(1.0, object_conf_threshold))
        self._last_cleanup = 0.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._flush_all()

    def add_event(self, room_id: str, camera_id: str, event: Dict[str, object]) -> None:
        now = time.time()
        if not self._should_store_event(event):
            return
        with self._lock:
            room = self._buffers.setdefault(room_id, {})
            buf = room.setdefault(camera_id, OverlayBuffer())
            buf.events.append(event)
            self._prune_locked(buf, now)
        logger.debug(
            "overlay_store.add_event room_id=%s camera_id=%s event_type=%s",
            room_id,
            camera_id,
            event.get("event_type"),
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._flush_due()
            self._cleanup_due()
            time.sleep(0.2)

    def _flush_due(self) -> None:
        now = time.time()
        to_flush = []
        with self._lock:
            for room_id, cameras in self._buffers.items():
                for camera_id, buf in cameras.items():
                    self._prune_locked(buf, now)
                    if now - buf.last_flush >= self.flush_interval_seconds:
                        buf.last_flush = now
                        to_flush.append((room_id, camera_id, list(buf.events)))
        for room_id, camera_id, events in to_flush:
            self._write_events(room_id, camera_id, events)

    def _flush_all(self) -> None:
        with self._lock:
            items = [
                (room_id, camera_id, list(buf.events))
                for room_id, cameras in self._buffers.items()
                for camera_id, buf in cameras.items()
            ]
        for room_id, camera_id, events in items:
            self._write_events(room_id, camera_id, events)

    def _write_events(self, room_id: str, camera_id: str, events: list) -> None:
        dir_path = os.path.join(self.root_path, room_id, camera_id)
        os.makedirs(dir_path, exist_ok=True)
        grouped: Dict[int, list] = {}
        for event in events:
            ts = int(_get_float(event, "timestamp"))
            grouped.setdefault(ts, []).append(event)
        for ts, bucket in grouped.items():
            file_path = os.path.join(dir_path, f"{ts}.jsonl")
            tmp_path = f"{file_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                for event in bucket:
                    handle.write(json.dumps(event, separators=(",", ":")))
                    handle.write("\n")
            os.replace(tmp_path, file_path)
            self._append_index(dir_path, ts, bucket)
            logger.info(
                "overlay_store.flush room_id=%s camera_id=%s ts=%s events=%d path=%s",
                room_id,
                camera_id,
                ts,
                len(bucket),
                file_path,
            )

    def _append_index(self, dir_path: str, ts: int, events: list) -> None:
        date_key = time.strftime("%Y-%m-%d", time.localtime(ts))
        index_path = os.path.join(dir_path, f"index-{date_key}.jsonl")
        try:
            with open(index_path, "a", encoding="utf-8") as handle:
                for event in events:
                    payload = {
                        "timestamp": ts,
                        "event_type": event.get("event_type"),
                        "global_person_id": event.get("global_person_id"),
                        "confidence": event.get("confidence"),
                        "file": f"{ts}.jsonl",
                    }
                    handle.write(json.dumps(payload, separators=(",", ":")))
                    handle.write("\n")
        except Exception as exc:
            logger.warning("overlay_store.index_failed path=%s error=%s", index_path, exc)

    def _cleanup_due(self) -> None:
        if self.disk_retention_seconds <= 0:
            return
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval_seconds:
            return
        self._last_cleanup = now
        self._cleanup_disk(now)

    def _cleanup_disk(self, now: float) -> None:
        cutoff = now - self.disk_retention_seconds
        if not os.path.exists(self.root_path):
            return
        for room_id in os.listdir(self.root_path):
            room_path = os.path.join(self.root_path, room_id)
            if not os.path.isdir(room_path):
                continue
            for camera_id in os.listdir(room_path):
                cam_path = os.path.join(room_path, camera_id)
                if not os.path.isdir(cam_path):
                    continue
                for name in os.listdir(cam_path):
                    if not name.endswith(".jsonl"):
                        continue
                    full_path = os.path.join(cam_path, name)
                    if name.startswith("index-"):
                        date_part = name[len("index-") : -len(".jsonl")]
                        try:
                            date_ts = time.mktime(time.strptime(date_part, "%Y-%m-%d"))
                        except Exception:
                            continue
                        if date_ts < cutoff:
                            try:
                                os.remove(full_path)
                            except OSError:
                                pass
                        continue
                    try:
                        ts = int(name[:-len(".jsonl")])
                    except ValueError:
                        continue
                    if ts < cutoff:
                        try:
                            os.remove(full_path)
                        except OSError:
                            pass

    def _prune_locked(self, buf: OverlayBuffer, now: float) -> None:
        cutoff = now - self.retention_seconds
        while buf.events and _get_float(buf.events[0], "timestamp") < cutoff:
            buf.events.popleft()

    def _should_store_event(self, event: Dict[str, object]) -> bool:
        event_type = event.get("event_type")
        confidence = _get_float(event, "confidence")
        if isinstance(event_type, str) and event_type.startswith("person_"):
            return confidence >= self._person_conf_threshold
        if event_type in ("object_detected", "object_associated"):
            return confidence >= self._object_conf_threshold
        return True


def _get_float(event: Dict[str, object], key: str) -> float:
    value = event.get(key, 0.0)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
