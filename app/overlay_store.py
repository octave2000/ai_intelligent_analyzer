import json
import logging
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

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
        snapshot_raw_enabled: bool = False,
        snapshot_raw_path: str = "data/overlay_snapshots_raw",
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
        self.snapshot_raw_enabled = snapshot_raw_enabled
        self.snapshot_raw_path = snapshot_raw_path
        self.snapshot_all = snapshot_all
        self.snapshot_min_interval_seconds = max(0.1, snapshot_min_interval_seconds)
        self._last_snapshot_all: Dict[str, Dict[str, float]] = {}
        self._buffers: Dict[str, Dict[str, OverlayBuffer]] = {}
        self._snapshot_buffers: Dict[str, Dict[str, Dict[int, SnapshotBuffer]]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._snapshot_thread: Optional[threading.Thread] = None
        self._snapshot_write_queue: "queue.Queue[Optional[Tuple[str, str, int, SnapshotBuffer]]]" = queue.Queue(
            maxsize=256
        )
        self._person_conf_threshold = max(0.0, min(1.0, person_conf_threshold))
        self._object_conf_threshold = max(0.0, min(1.0, object_conf_threshold))
        self._last_cleanup = 0.0
        self._last_event_ts: Dict[str, Dict[str, int]] = {}
        self._last_event_ts_known: Set[Tuple[str, str]] = set()
        self._max_gap_fill_seconds = 600
        self._max_idle_gap_fill_seconds = 30

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if self.snapshot_enabled and (self._snapshot_thread is None or not self._snapshot_thread.is_alive()):
            self._snapshot_thread = threading.Thread(
                target=self._run_snapshot_writer,
                daemon=True,
            )
            self._snapshot_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._flush_all()
        if self.snapshot_enabled and self._snapshot_thread and self._snapshot_thread.is_alive():
            self._snapshot_write_queue.join()
            self._snapshot_write_queue.put(None)
            self._snapshot_thread.join(timeout=2.0)

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
        fill_until_ts = int(now) - 1
        to_flush: List[Tuple[str, str, list, int]] = []
        with self._lock:
            for room_id, cameras in self._buffers.items():
                for camera_id, buf in cameras.items():
                    self._prune_locked(buf, now)
                    if now - buf.last_flush >= self.flush_interval_seconds:
                        buf.last_flush = now
                        pending = list(buf.events)
                        if pending:
                            buf.events.clear()
                        to_flush.append((room_id, camera_id, pending, fill_until_ts))
        for room_id, camera_id, events, fill_until in to_flush:
            if events:
                self._write_events(room_id, camera_id, events)
            self._write_idle_gap(room_id, camera_id, fill_until)

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
            self._enqueue_snapshot_batch(room_id, camera_id, ts, buf, blocking=True)

    def _write_events(self, room_id: str, camera_id: str, events: list) -> None:
        dir_path = os.path.join(self.root_path, room_id, camera_id)
        os.makedirs(dir_path, exist_ok=True)
        last_ts = self._ensure_last_event_ts(room_id, camera_id, dir_path)
        grouped: Dict[int, list] = {}
        for event in events:
            ts = int(_get_float(event, "timestamp"))
            grouped.setdefault(ts, []).append(event)
        if not grouped:
            return
        for ts in sorted(grouped.keys()):
            bucket = grouped.get(ts, [])
            if not bucket:
                continue
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
                fill_count = min(missing, self._max_gap_fill_seconds)
                if fill_count < missing:
                    logger.warning(
                        "overlay_store.missing_seconds_truncated room_id=%s camera_id=%s missing=%s fill_count=%s max_fill=%s",
                        room_id,
                        camera_id,
                        missing,
                        fill_count,
                        self._max_gap_fill_seconds,
                    )
                for miss_ts in range(last_ts + 1, last_ts + 1 + fill_count):
                    filler = self._gap_filler_event(
                        room_id=room_id,
                        camera_id=camera_id,
                        ts=miss_ts,
                        last_ts=miss_ts - 1,
                        next_ts=ts,
                    )
                    self._write_bucket(dir_path, room_id, camera_id, miss_ts, [filler], is_backfill=True)
            self._write_bucket(dir_path, room_id, camera_id, ts, bucket, is_backfill=False)
            last_ts = ts

    def _write_idle_gap(self, room_id: str, camera_id: str, fill_until_ts: int) -> None:
        if fill_until_ts < 0:
            return
        dir_path = os.path.join(self.root_path, room_id, camera_id)
        os.makedirs(dir_path, exist_ok=True)
        last_ts = self._ensure_last_event_ts(room_id, camera_id, dir_path)
        if last_ts is None or fill_until_ts <= last_ts:
            return
        missing = fill_until_ts - last_ts
        if missing > self._max_idle_gap_fill_seconds:
            logger.info(
                "overlay_store.idle_gap_skip room_id=%s camera_id=%s missing=%s max_idle_fill=%s",
                room_id,
                camera_id,
                missing,
                self._max_idle_gap_fill_seconds,
            )
            return
        fill_count = missing
        logger.info(
            "overlay_store.idle_gap_fill room_id=%s camera_id=%s last_ts=%s fill_until=%s fill_count=%s",
            room_id,
            camera_id,
            last_ts,
            fill_until_ts,
            fill_count,
        )
        for miss_ts in range(last_ts + 1, last_ts + 1 + fill_count):
            filler = self._gap_filler_event(
                room_id=room_id,
                camera_id=camera_id,
                ts=miss_ts,
                last_ts=miss_ts - 1,
                next_ts=None,
                reason="missing_second_idle_fill",
                trigger="flush_due",
            )
            self._write_bucket(dir_path, room_id, camera_id, miss_ts, [filler], is_backfill=True)

    def _ensure_last_event_ts(
        self,
        room_id: str,
        camera_id: str,
        dir_path: str,
    ) -> Optional[int]:
        key = (room_id, camera_id)
        room = self._last_event_ts.setdefault(room_id, {})
        if key in self._last_event_ts_known:
            return room.get(camera_id)
        self._last_event_ts_known.add(key)
        last_ts = self._probe_latest_timestamp(dir_path)
        if last_ts is not None:
            room[camera_id] = last_ts
            logger.info(
                "overlay_store.resume_last_ts room_id=%s camera_id=%s ts=%s",
                room_id,
                camera_id,
                last_ts,
            )
        return last_ts

    @staticmethod
    def _probe_latest_timestamp(dir_path: str) -> Optional[int]:
        if not os.path.isdir(dir_path):
            return None
        latest: Optional[int] = None
        try:
            with os.scandir(dir_path) as entries:
                for entry in entries:
                    if not entry.is_file() or not entry.name.endswith(".json"):
                        continue
                    name = entry.name[:-len(".json")]
                    if not name.isdigit():
                        continue
                    ts = int(name)
                    if latest is None or ts > latest:
                        latest = ts
        except OSError:
            return None
        return latest

    def _write_bucket(
        self,
        dir_path: str,
        room_id: str,
        camera_id: str,
        ts: int,
        bucket: list,
        is_backfill: bool = False,
    ) -> None:
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
            "overlay_store.flush room_id=%s camera_id=%s ts=%s events=%d path=%s backfill=%s",
            room_id,
            camera_id,
            ts,
            len(bucket),
            file_path,
            is_backfill,
        )

    @staticmethod
    def _gap_filler_event(
        room_id: str,
        camera_id: str,
        ts: int,
        last_ts: int,
        next_ts: Optional[int],
        reason: str = "missing_second_backfill",
        trigger: str = "next_event",
    ) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "timestamp": float(ts),
            "room_id": room_id,
            "camera_id": camera_id,
            "global_person_id": None,
            "person_id": None,
            "event_type": "frame_skipped",
            "confidence": 0.0,
            "skip_reason": reason,
            "backfill_trigger": trigger,
            "backfill": True,
            "backfill_last_timestamp": last_ts,
        }
        if next_ts is not None:
            payload["backfill_next_timestamp"] = next_ts
        return payload

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
        now = time.time()
        event_copy = event.copy()
        with self._lock:
            room = self._snapshot_buffers.setdefault(room_id, {})
            camera = room.setdefault(camera_id, {})
            buf = camera.get(ts)
            if buf is None:
                try:
                    image = frame.copy()
                except Exception:
                    return
                camera[ts] = SnapshotBuffer(
                    frame=image,
                    last_update=now,
                    event_timestamp=event_timestamp,
                    frame_source_timestamp=frame_source_timestamp,
                    timestamp_offset_seconds=timestamp_offset_seconds,
                    emitted_at=emitted_at,
                    frame_age_seconds=frame_age_seconds,
                    frame_transport_delay_seconds=frame_transport_delay_seconds,
                    event_type=event_type,
                    event_count=1,
                    events=[event_copy],
                )
            else:
                buf.last_update = now
                buf.event_count += 1
                if len(buf.events) < 200:
                    buf.events.append(event_copy)
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
            self._enqueue_snapshot_batch(room_id, camera_id, ts, buf)

    def _enqueue_snapshot_batch(
        self,
        room_id: str,
        camera_id: str,
        ts: int,
        buf: SnapshotBuffer,
        blocking: bool = False,
    ) -> None:
        if self._snapshot_thread is None or not self._snapshot_thread.is_alive():
            self._write_snapshot_batch(room_id, camera_id, ts, buf)
            return
        item = (room_id, camera_id, ts, buf)
        try:
            if blocking:
                self._snapshot_write_queue.put(item, timeout=1.0)
            else:
                self._snapshot_write_queue.put_nowait(item)
        except queue.Full:
            logger.warning(
                "overlay_store.snapshot_drop room_id=%s camera_id=%s ts=%s reason=snapshot_queue_full",
                room_id,
                camera_id,
                ts,
            )

    def _run_snapshot_writer(self) -> None:
        while True:
            item = self._snapshot_write_queue.get()
            try:
                if item is None:
                    return
                room_id, camera_id, ts, buf = item
                self._write_snapshot_batch(room_id, camera_id, ts, buf)
            finally:
                self._snapshot_write_queue.task_done()

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
            raw_image = buf.frame.copy()
        except Exception:
            return
        image = raw_image.copy()
        self._draw_snapshot_events(image, buf.events)
        dir_path = os.path.join(self.snapshot_path, room_id, camera_id)
        os.makedirs(dir_path, exist_ok=True)
        filename = f"{ts}.jpg"
        path = os.path.join(dir_path, filename)
        try:
            cv2.imwrite(path, image)
        except Exception:
            return
        if self.snapshot_raw_enabled:
            raw_dir_path = os.path.join(self.snapshot_raw_path, room_id, camera_id)
            os.makedirs(raw_dir_path, exist_ok=True)
            raw_path = os.path.join(raw_dir_path, filename)
            try:
                cv2.imwrite(raw_path, raw_image)
            except Exception:
                return

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
        if self.snapshot_raw_enabled:
            raw_dir_path = os.path.join(self.snapshot_raw_path, room_id, camera_id, "all")
            os.makedirs(raw_dir_path, exist_ok=True)
            raw_path = os.path.join(raw_dir_path, filename)
            try:
                cv2.imwrite(raw_path, image)
            except Exception:
                return

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
                entry: Dict[str, object] = {
                    "timestamp": ts,
                    "event_type": event.get("event_type"),
                    "global_person_id": event.get("global_person_id"),
                    "confidence": event.get("confidence"),
                    "file": f"{ts}.json",
                }
                role = event.get("role")
                if isinstance(role, str) and role.strip():
                    entry["role"] = role
                role_reason = event.get("role_reason")
                if isinstance(role_reason, str) and role_reason.strip():
                    entry["role_reason"] = role_reason
                person_role = event.get("person_role")
                if isinstance(person_role, str) and person_role.strip():
                    entry["person_role"] = person_role
                person_id = event.get("person_id")
                if isinstance(person_id, str) and person_id.strip():
                    entry["person_id"] = person_id
                existing.append(entry)
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
