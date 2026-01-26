import json
import os
import threading
import time
from typing import Dict, Optional, Tuple

import cv2

from app.stream_ingestor import StreamIngestor


class StreamManager:
    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        target_fps: int,
        backoff_min_seconds: float,
        backoff_max_seconds: float,
        stale_threshold_seconds: float,
        read_timeout_seconds: float,
        use_ffmpeg: str,
        storage_path: str,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = target_fps
        self.backoff_min_seconds = backoff_min_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.use_ffmpeg = use_ffmpeg
        self.stale_threshold_seconds = stale_threshold_seconds
        self.storage_path = storage_path
        self._rooms: Dict[str, Dict[str, StreamIngestor]] = {}
        self._lock = threading.Lock()
        self._load_storage()

    def start(self) -> None:
        with self._lock:
            for cameras in self._rooms.values():
                for ingestor in cameras.values():
                    ingestor.start()

    def stop(self) -> None:
        with self._lock:
            for cameras in self._rooms.values():
                for ingestor in cameras.values():
                    ingestor.stop()

    def register_room(self, room_id: str) -> bool:
        with self._lock:
            if room_id in self._rooms:
                return False
            self._rooms[room_id] = {}
            self._persist_storage_locked()
            return True

    def add_camera(self, room_id: str, camera_id: str, url: str) -> bool:
        with self._lock:
            if room_id not in self._rooms:
                return False
            cameras = self._rooms[room_id]
            if camera_id in cameras:
                return False
            ingestor = StreamIngestor(
                f"{room_id}:{camera_id}",
                url,
                self.frame_width,
                self.frame_height,
                self.target_fps,
                self.backoff_min_seconds,
                self.backoff_max_seconds,
                self.read_timeout_seconds,
                self.use_ffmpeg,
            )
            cameras[camera_id] = ingestor
            ingestor.start()
            self._persist_storage_locked()
            return True

    def remove_camera(self, room_id: str, camera_id: str) -> bool:
        with self._lock:
            cameras = self._rooms.get(room_id)
            if cameras is None:
                return False
            ingestor = cameras.pop(camera_id, None)
            if ingestor is None:
                return False
            ingestor.stop()
            self._persist_storage_locked()
            return True

    def remove_room(self, room_id: str) -> bool:
        with self._lock:
            cameras = self._rooms.pop(room_id, None)
            if cameras is None:
                return False
            for ingestor in cameras.values():
                ingestor.stop()
            self._persist_storage_locked()
            return True

    def list_rooms(self) -> Dict[str, object]:
        with self._lock:
            return {
                room_id: list(cameras.keys())
                for room_id, cameras in self._rooms.items()
            }

    def get_snapshot(
        self, room_id: str, camera_id: str
    ) -> Tuple[Optional["cv2.typing.MatLike"], Optional[float]]:
        with self._lock:
            cameras = self._rooms.get(room_id)
            if cameras is None:
                return None, None
            ingestor = cameras.get(camera_id)
            if ingestor is None:
                return None, None
        return ingestor.snapshot()

    def _camera_status(self, metrics, timestamp: Optional[float]) -> Dict[str, object]:
        now = time.time()
        frame_age = None
        if timestamp is not None:
            frame_age = max(0.0, now - timestamp)

        health = "down"
        if metrics.is_running and frame_age is not None:
            if frame_age <= self.stale_threshold_seconds:
                health = "healthy"
            else:
                health = "degraded"
        elif metrics.is_running:
            health = "degraded"

        return {
            "is_running": metrics.is_running,
            "last_frame_timestamp": timestamp,
            "frame_age_seconds": frame_age,
            "frames_received_total": metrics.frames_received_total,
            "restart_count": metrics.restart_count,
            "last_error_message": metrics.last_error_message,
            "health": health,
        }

    def room_health(self, room_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            cameras = self._rooms.get(room_id)
            if cameras is None:
                return None
            camera_items = list(cameras.items())

        statuses: Dict[str, object] = {}
        overall = "healthy"
        for camera_id, ingestor in camera_items:
            _frame, ts = ingestor.snapshot()
            status = self._camera_status(ingestor.metrics, ts)
            statuses[camera_id] = status
            if status["health"] == "down":
                overall = "down"
            elif status["health"] == "degraded" and overall != "down":
                overall = "degraded"

        return {"overall_status": overall, "cameras": statuses}

    def health(self) -> Dict[str, object]:
        with self._lock:
            room_items = list(self._rooms.items())

        rooms: Dict[str, object] = {}
        overall = "healthy"
        for room_id, cameras in room_items:
            room_status = self.room_health(room_id)
            rooms[room_id] = room_status
            if room_status is None:
                continue
            if room_status["overall_status"] == "down":
                overall = "down"
            elif room_status["overall_status"] == "degraded" and overall != "down":
                overall = "degraded"

        return {"overall_status": overall, "rooms": rooms}

    def _persist_storage_locked(self) -> None:
        payload: Dict[str, Dict[str, str]] = {}
        for room_id, cameras in self._rooms.items():
            payload[room_id] = {camera_id: cam.url for camera_id, cam in cameras.items()}

        os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
        tmp_path = f"{self.storage_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, self.storage_path)

    def _load_storage(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return
        for room_id, cameras in payload.items():
            if not isinstance(room_id, str) or not isinstance(cameras, dict):
                continue
            self._rooms[room_id] = {}
            for camera_id, url in cameras.items():
                if not isinstance(camera_id, str) or not isinstance(url, str):
                    continue
                self._rooms[room_id][camera_id] = StreamIngestor(
                    f"{room_id}:{camera_id}",
                    url,
                    self.frame_width,
                    self.frame_height,
                    self.target_fps,
                    self.backoff_min_seconds,
                    self.backoff_max_seconds,
                    self.read_timeout_seconds,
                    self.use_ffmpeg,
                )
