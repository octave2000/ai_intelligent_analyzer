import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import cv2

logger = logging.getLogger(__name__)


@dataclass
class OverlayBuffer:
    events: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=5000))
    last_flush: float = 0.0


@dataclass
class SnapshotBuffer:
    frame: "cv2.typing.MatLike"
    annotations: list
    last_update: float


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
        snapshot_enabled: bool = False,
        snapshot_path: str = "data/overlay_snapshots",
        snapshot_all: bool = False,
        snapshot_min_interval_seconds: float = 1.0,
    ) -> None:
        self.root_path = root_path
        self.retention_seconds = retention_seconds
        self.flush_interval_seconds = flush_interval_seconds
        self.disk_retention_seconds = max(0.0, disk_retention_seconds)
        self.cleanup_interval_seconds = max(5.0, cleanup_interval_seconds)
        self.snapshot_enabled = snapshot_enabled
        self.snapshot_path = snapshot_path
        self.snapshot_all = snapshot_all
        self.snapshot_min_interval_seconds = max(0.1, snapshot_min_interval_seconds)
        self._last_snapshot_all: Dict[str, Dict[str, float]] = {}
        self._buffers: Dict[str, Dict[str, OverlayBuffer]] = {}
        self._snapshot_buffers: Dict[str, Dict[str, Dict[int, SnapshotBuffer]]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._person_conf_threshold = max(0.0, min(1.0, person_conf_threshold))
        self._object_conf_threshold = max(0.0, min(1.0, object_conf_threshold))
        self._last_cleanup = 0.0
        self._last_event_ts: Dict[str, Dict[str, int]] = {}

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

    def add_event(
        self,
        room_id: str,
        camera_id: str,
        event: Dict[str, object],
        frame: Optional["cv2.typing.MatLike"] = None,
    ) -> None:
        now = time.time()
        if not self._should_store_event(event):
            return
        with self._lock:
            room = self._buffers.setdefault(room_id, {})
            buf = room.setdefault(camera_id, OverlayBuffer())
            buf.events.append(event)
            self._prune_locked(buf, now)
        if self.snapshot_enabled and frame is not None:
            self._buffer_snapshot(room_id, camera_id, event, frame)
        logger.debug(
            "overlay_store.add_event room_id=%s camera_id=%s event_type=%s",
            room_id,
            camera_id,
            event.get("event_type"),
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._flush_due()
            self._flush_snapshots_due()
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
                        pending = list(buf.events)
                        if pending:
                            to_flush.append((room_id, camera_id, pending))
                            buf.events.clear()
        for room_id, camera_id, events in to_flush:
            self._write_events(room_id, camera_id, events)

    def _flush_all(self) -> None:
        with self._lock:
            items = [
                (room_id, camera_id, list(buf.events))
                for room_id, cameras in self._buffers.items()
                for camera_id, buf in cameras.items()
            ]
            snapshot_items = self._collect_snapshot_items_locked(force=True)
        for room_id, camera_id, events in items:
            self._write_events(room_id, camera_id, events)
        for room_id, camera_id, ts, buf in snapshot_items:
            self._write_snapshot_batch(room_id, camera_id, ts, buf)

    def _write_events(self, room_id: str, camera_id: str, events: list) -> None:
        dir_path = os.path.join(self.root_path, room_id, camera_id)
        os.makedirs(dir_path, exist_ok=True)
        grouped: Dict[int, list] = {}
        for event in events:
            ts = int(_get_float(event, "timestamp"))
            grouped.setdefault(ts, []).append(event)
        if not grouped:
            return
        for ts, bucket in grouped.items():
            last_ts = self._last_event_ts.setdefault(room_id, {}).get(camera_id)
            if last_ts is not None and ts > last_ts + 1:
                missing = (ts - last_ts) - 1
                logger.info(
                    "overlay_store.missing_seconds room_id=%s camera_id=%s last_ts=%s next_ts=%s missing=%s reason=no_events",
                    room_id,
                    camera_id,
                    last_ts,
                    ts,
                    missing,
                )
            self._last_event_ts.setdefault(room_id, {})[camera_id] = ts
            file_path = os.path.join(dir_path, f"{ts}.json")
            tmp_path = f"{file_path}.tmp"
            existing: list = []
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as handle:
                        loaded = json.load(handle)
                    if isinstance(loaded, list):
                        existing = loaded
                except Exception:
                    existing = []
            payload = existing + bucket
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
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

    def _buffer_snapshot(
        self,
        room_id: str,
        camera_id: str,
        event: Dict[str, object],
        frame: "cv2.typing.MatLike",
    ) -> None:
        ts_raw = _get_float(event, "timestamp")
        ts = int(ts_raw)
        try:
            image = frame.copy()
        except Exception:
            return
        label = event.get("event_type", "event")
        gid = event.get("global_person_id")
        if gid is not None:
            label = f"{label}:gid={gid}"
        label = f"{label}@{ts_raw:.2f}"
        annotation = {
            "bbox": event.get("bbox"),
            "label": label,
            "confidence": event.get("confidence"),
        }
        with self._lock:
            room = self._snapshot_buffers.setdefault(room_id, {})
            camera = room.setdefault(camera_id, {})
            buf = camera.get(ts)
            if buf is None:
                camera[ts] = SnapshotBuffer(
                    frame=image,
                    annotations=[annotation],
                    last_update=time.time(),
                )
            else:
                buf.annotations.append(annotation)
                buf.last_update = time.time()

    def _flush_snapshots_due(self) -> None:
        if not self.snapshot_enabled:
            return
        now = time.time()
        with self._lock:
            items = self._collect_snapshot_items_locked(now=now)
        for room_id, camera_id, ts, buf in items:
            self._write_snapshot_batch(room_id, camera_id, ts, buf)

    def _collect_snapshot_items_locked(
        self,
        now: Optional[float] = None,
        force: bool = False,
    ) -> list:
        items = []
        if now is None:
            now = time.time()
        cutoff_ts = int(now) - 1
        for room_id, cameras in list(self._snapshot_buffers.items()):
            for camera_id, buckets in list(cameras.items()):
                for ts, buf in list(buckets.items()):
                    if force or ts <= cutoff_ts:
                        items.append((room_id, camera_id, ts, buf))
                        buckets.pop(ts, None)
                if not buckets:
                    cameras.pop(camera_id, None)
            if not cameras:
                self._snapshot_buffers.pop(room_id, None)
        return items

    def _write_snapshot_batch(
        self,
        room_id: str,
        camera_id: str,
        ts: int,
        buf: SnapshotBuffer,
    ) -> None:
        try:
            image = buf.frame.copy()
        except Exception:
            return
        try:
            cv2.putText(
                image,
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)),
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 255),
                2,
            )
        except Exception:
            pass
        for ann in buf.annotations:
            bbox = ann.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 255), 2)
                label = ann.get("label")
                if label:
                    cv2.putText(
                        image,
                        str(label),
                        (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 220, 255),
                        1,
                    )
        dir_path = os.path.join(self.snapshot_path, room_id, camera_id)
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{ts}.jpg"
        path = os.path.join(dir_path, filename)
        try:
            cv2.imwrite(path, image)
        except Exception:
            return

    def add_snapshot_all(
        self,
        room_id: str,
        camera_id: str,
        annotations: list,
        frame: "cv2.typing.MatLike",
        timestamp: Optional[float] = None,
    ) -> None:
        if not self.snapshot_all:
            return
        now = time.time() if timestamp is None else timestamp
        with self._lock:
            room = self._last_snapshot_all.setdefault(room_id, {})
            last_ts = room.get(camera_id, 0.0)
            if now - last_ts < self.snapshot_min_interval_seconds:
                return
            room[camera_id] = now
        try:
            image = frame.copy()
        except Exception:
            return
        for ann in annotations:
            bbox = ann.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1, y1, x2, y2 = (int(v) for v in bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = ann.get("label")
                if label:
                    cv2.putText(
                        image,
                        str(label),
                        (x1, max(12, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )
        dir_path = os.path.join(self.snapshot_path, room_id, camera_id, "all")
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{int(now * 1000)}.jpg"
        path = os.path.join(dir_path, filename)
        try:
            cv2.imwrite(path, image)
        except Exception:
            return

    def _append_index(self, dir_path: str, ts: int, events: list) -> None:
        date_key = time.strftime("%Y-%m-%d", time.localtime(ts))
        index_path = os.path.join(dir_path, f"index-{date_key}.json")
        try:
            existing = []
            if os.path.exists(index_path):
                with open(index_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                    if isinstance(payload, list):
                        existing = payload
            for event in events:
                existing.append(
                    {
                        "timestamp": ts,
                        "event_type": event.get("event_type"),
                        "global_person_id": event.get("global_person_id"),
                        "confidence": event.get("confidence"),
                        "file": f"{ts}.json",
                    }
                )
            with open(index_path, "w", encoding="utf-8") as handle:
                json.dump(existing, handle, separators=(",", ":"))
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
                    if not name.endswith(".json"):
                        continue
                    full_path = os.path.join(cam_path, name)
                    if name.startswith("index-"):
                        date_part = name[len("index-") : -len(".json")]
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
                        ts = int(name[:-len(".json")])
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
