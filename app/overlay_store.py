import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


@dataclass
class OverlayBuffer:
    events: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=5000))
    last_flush: float = 0.0


@dataclass
class SnapshotBuffer:
    frame: "cv2.typing.MatLike"
    last_update: float
    event_timestamp: Optional[float] = None
    frame_source_timestamp: Optional[float] = None
    timestamp_offset_seconds: Optional[float] = None
    emitted_at: Optional[float] = None
    frame_age_seconds: Optional[float] = None
    frame_transport_delay_seconds: Optional[float] = None
    event_type: Optional[str] = None
    event_count: int = 0
    events: List[Dict[str, object]] = field(default_factory=list)


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
        event_timestamp = _get_optional_float(event, "timestamp")
        frame_source_timestamp = _get_optional_float(event, "frame_source_timestamp")
        timestamp_offset_seconds = _get_optional_float(event, "timestamp_offset_seconds")
        emitted_at = _get_optional_float(event, "emitted_at")
        frame_age_seconds = _get_optional_float(event, "frame_age_seconds")
        frame_transport_delay_seconds = _get_optional_float(
            event, "frame_transport_delay_seconds"
        )
        event_type = event.get("event_type")
        if not isinstance(event_type, str):
            event_type = None
        try:
            image = frame.copy()
        except Exception:
            return
        with self._lock:
            room = self._snapshot_buffers.setdefault(room_id, {})
            camera = room.setdefault(camera_id, {})
            buf = camera.get(ts)
            if buf is None:
                camera[ts] = SnapshotBuffer(
                    frame=image,
                    last_update=time.time(),
                    event_timestamp=event_timestamp,
                    frame_source_timestamp=frame_source_timestamp,
                    timestamp_offset_seconds=timestamp_offset_seconds,
                    emitted_at=emitted_at,
                    frame_age_seconds=frame_age_seconds,
                    frame_transport_delay_seconds=frame_transport_delay_seconds,
                    event_type=event_type,
                    event_count=1,
                    events=[event.copy()],
                )
            else:
                buf.frame = image
                buf.last_update = time.time()
                buf.event_count += 1
                if len(buf.events) < 200:
                    buf.events.append(event.copy())
                if event_timestamp is not None:
                    buf.event_timestamp = event_timestamp
                if frame_source_timestamp is not None:
                    buf.frame_source_timestamp = frame_source_timestamp
                if timestamp_offset_seconds is not None:
                    buf.timestamp_offset_seconds = timestamp_offset_seconds
                if emitted_at is not None:
                    buf.emitted_at = emitted_at
                if frame_age_seconds is not None:
                    buf.frame_age_seconds = frame_age_seconds
                if frame_transport_delay_seconds is not None:
                    buf.frame_transport_delay_seconds = frame_transport_delay_seconds
                if event_type is not None:
                    buf.event_type = event_type

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
        self._draw_snapshot_events(image, buf.events)
        dir_path = os.path.join(self.snapshot_path, room_id, camera_id)
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{ts}.jpg"
        path = os.path.join(dir_path, filename)
        try:
            cv2.imwrite(path, image)
        except Exception:
            return
        write_ts = time.time()
        event_timestamp = buf.event_timestamp if buf.event_timestamp is not None else float(ts)
        emitted_at = buf.emitted_at
        sidecar = {
            "snapshot_type": "event_snapshot",
            "room_id": room_id,
            "camera_id": camera_id,
            "image_file": filename,
            "event_timestamp": event_timestamp,
            "frame_source_timestamp": buf.frame_source_timestamp,
            "timestamp_offset_seconds": buf.timestamp_offset_seconds,
            "emitted_at": emitted_at,
            "write_time": write_ts,
            "frame_age_seconds": buf.frame_age_seconds,
            "frame_transport_delay_seconds": buf.frame_transport_delay_seconds,
            "event_type": buf.event_type,
            "event_count": buf.event_count,
            "drawn_event_count": len(buf.events),
        }
        if emitted_at is not None:
            sidecar["event_pipeline_lag_seconds"] = max(0.0, emitted_at - event_timestamp)
        sidecar["snapshot_lag_seconds"] = max(0.0, write_ts - event_timestamp)
        sidecar_path = os.path.join(dir_path, f"{ts}.json")
        self._write_sidecar(sidecar_path, sidecar)

    def add_snapshot_all(
        self,
        room_id: str,
        camera_id: str,
        frame: "cv2.typing.MatLike",
        timestamp: Optional[float] = None,
    ) -> None:
        if not self.snapshot_all:
            return
        event_ts = time.time() if timestamp is None else timestamp
        with self._lock:
            room = self._last_snapshot_all.setdefault(room_id, {})
            last_ts = room.get(camera_id, 0.0)
            if event_ts - last_ts < self.snapshot_min_interval_seconds:
                return
            room[camera_id] = event_ts
        try:
            image = frame.copy()
        except Exception:
            return
        dir_path = os.path.join(self.snapshot_path, room_id, camera_id, "all")
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{int(event_ts * 1000)}.jpg"
        path = os.path.join(dir_path, filename)
        try:
            cv2.imwrite(path, image)
        except Exception:
            return
        write_ts = time.time()
        sidecar = {
            "snapshot_type": "all_snapshot",
            "room_id": room_id,
            "camera_id": camera_id,
            "image_file": filename,
            "event_timestamp": event_ts,
            "emitted_at": write_ts,
            "write_time": write_ts,
            "snapshot_lag_seconds": max(0.0, write_ts - event_ts),
        }
        sidecar_path = os.path.join(
            dir_path, f"{int(event_ts * 1000)}.json"
        )
        self._write_sidecar(sidecar_path, sidecar)

    def _write_sidecar(self, path: str, payload: Dict[str, object]) -> None:
        tmp_path = f"{path}.tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, separators=(",", ":"))
            os.replace(tmp_path, path)
        except Exception as exc:
            logger.warning("overlay_store.snapshot_sidecar_failed path=%s error=%s", path, exc)

    def _draw_snapshot_events(
        self,
        image: "cv2.typing.MatLike",
        events: List[Dict[str, object]],
    ) -> None:
        if image is None or not events:
            return
        height, width = image.shape[:2]
        seen = set()
        event_lines = []
        for event in events:
            event_type = _get_str(event, "event_type")
            if event_type is None:
                continue
            label = self._snapshot_event_label(event, event_type)
            bbox = _normalize_bbox(event.get("bbox"), width, height)
            if bbox is not None:
                key = (bbox, label)
                if key in seen:
                    continue
                seen.add(key)
                color = _event_color(event_type)
                x1, y1, x2, y2 = bbox
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                self._draw_label(image, label, x1, max(0, y1 - 6), color)
            else:
                key = ("line", label)
                if key in seen:
                    continue
                seen.add(key)
                event_lines.append(label)

        if event_lines:
            max_lines = min(8, len(event_lines))
            for i in range(max_lines):
                text = event_lines[i]
                y = 22 + (i * 18)
                self._draw_label(image, text, 8, y, (255, 255, 255))
        summary = f"events: {len(events)}"
        self._draw_label(image, summary, 8, height - 12, (80, 80, 80))

    @staticmethod
    def _draw_label(
        image: "cv2.typing.MatLike",
        text: str,
        x: int,
        y: int,
        color: Tuple[int, int, int],
    ) -> None:
        if not text:
            return
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x = max(0, x)
        y = max(th + baseline, y)
        bg_tl = (x, max(0, y - th - baseline - 2))
        bg_br = (min(image.shape[1] - 1, x + tw + 4), min(image.shape[0] - 1, y + 2))
        cv2.rectangle(image, bg_tl, bg_br, (0, 0, 0), -1)
        cv2.putText(image, text, (x + 2, y - 2), font, scale, color, thickness, cv2.LINE_AA)

    @staticmethod
    def _snapshot_event_label(event: Dict[str, object], event_type: str) -> str:
        parts = [event_type]
        person_id = _get_str(event, "person_id")
        if person_id:
            parts.append(person_id)
        object_type = _get_str(event, "object_type")
        if object_type:
            parts.append(object_type)
        orientation = _get_str(event, "orientation")
        if orientation:
            parts.append(orientation)
        status = _get_str(event, "status")
        if status:
            parts.append(status)
        confidence = _get_optional_float(event, "confidence")
        if confidence is not None:
            parts.append(f"{confidence:.2f}")
        return " | ".join(parts[:4])

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


def _get_optional_float(event: Dict[str, object], key: str) -> Optional[float]:
    value = event.get(key)
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _get_str(event: Dict[str, object], key: str) -> Optional[str]:
    value = event.get(key)
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _normalize_bbox(
    raw: object,
    width: int,
    height: int,
) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        x1 = int(float(raw[0]))
        y1 = int(float(raw[1]))
        x2 = int(float(raw[2]))
        y2 = int(float(raw[3]))
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width - 1, x2))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _event_color(event_type: str) -> Tuple[int, int, int]:
    if event_type.startswith("person_"):
        return (0, 220, 0)
    if event_type.startswith("object_"):
        return (0, 180, 255)
    if event_type == "role_assigned":
        return (255, 215, 0)
    if event_type == "proximity_event":
        return (0, 255, 255)
    if event_type.startswith("group_"):
        return (255, 120, 0)
    if event_type == "head_orientation_changed":
        return (255, 0, 255)
    if event_type == "body_movement":
        return (255, 64, 0)
    return (240, 240, 240)
