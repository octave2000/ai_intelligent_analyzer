# import json
# import os
# import threading
# import time
# from dataclasses import dataclass
# from typing import Dict, Optional, Tuple

# import cv2

# from app.stream_ingestor import StreamIngestor


# @dataclass
# class CameraEntry:
#     url: str
#     role: str
#     ingestor: StreamIngestor


# class StreamManager:
#     def __init__(
#         self,
#         frame_width: int,
#         frame_height: int,
#         target_fps: int,
#         backoff_min_seconds: float,
#         backoff_max_seconds: float,
#         stale_threshold_seconds: float,
#         read_timeout_seconds: float,
#         use_ffmpeg: str,
#         rtsp_transport: str,
#         storage_path: str,
#     ) -> None:
#         self.frame_width = frame_width
#         self.frame_height = frame_height
#         self.target_fps = target_fps
#         self.backoff_min_seconds = backoff_min_seconds
#         self.backoff_max_seconds = backoff_max_seconds
#         self.read_timeout_seconds = read_timeout_seconds
#         self.use_ffmpeg = use_ffmpeg
#         self.rtsp_transport = rtsp_transport
#         self.stale_threshold_seconds = stale_threshold_seconds
#         self.storage_path = storage_path
#         self._rooms: Dict[str, Dict[str, CameraEntry]] = {}
#         self._lock = threading.Lock()
#         self._load_storage()

#     def start(self) -> None:
#         with self._lock:
#             for cameras in self._rooms.values():
#                 for entry in cameras.values():
#                     entry.ingestor.start()

#     def stop(self) -> None:
#         with self._lock:
#             for cameras in self._rooms.values():
#                 for entry in cameras.values():
#                     entry.ingestor.stop()

#     def register_room(self, room_id: str) -> bool:
#         with self._lock:
#             if room_id in self._rooms:
#                 return False
#             self._rooms[room_id] = {}
#             self._persist_storage_locked()
#             return True

#     def add_camera(self, room_id: str, camera_id: str, url: str, role: str) -> bool:
#         with self._lock:
#             if room_id not in self._rooms:
#                 return False
#             cameras = self._rooms[room_id]
#             if camera_id in cameras:
#                 return False
#             ingestor = StreamIngestor(
#                 f"{room_id}:{camera_id}",
#                 url,
#                 self.frame_width,
#                 self.frame_height,
#                 self.target_fps,
#                 self.backoff_min_seconds,
#                 self.backoff_max_seconds,
#                 self.read_timeout_seconds,
#                 self.use_ffmpeg,
#                 self.rtsp_transport,
#             )
#             cameras[camera_id] = CameraEntry(url=url, role=role, ingestor=ingestor)
#             ingestor.start()
#             self._persist_storage_locked()
#             return True

#     def remove_camera(self, room_id: str, camera_id: str) -> bool:
#         with self._lock:
#             cameras = self._rooms.get(room_id)
#             if cameras is None:
#                 return False
#             entry = cameras.pop(camera_id, None)
#             if entry is None:
#                 return False
#             entry.ingestor.stop()
#             self._persist_storage_locked()
#             return True

#     def remove_room(self, room_id: str) -> bool:
#         with self._lock:
#             cameras = self._rooms.pop(room_id, None)
#             if cameras is None:
#                 return False
#             for entry in cameras.values():
#                 entry.ingestor.stop()
#             self._persist_storage_locked()
#             return True

#     def list_rooms(self) -> Dict[str, object]:
#         with self._lock:
#             return {
#                 room_id: {
#                     camera_id: entry.role for camera_id, entry in cameras.items()
#                 }
#                 for room_id, cameras in self._rooms.items()
#             }

#     def list_camera_entries(self) -> Dict[str, Dict[str, CameraEntry]]:
#         with self._lock:
#             return {
#                 room_id: dict(cameras) for room_id, cameras in self._rooms.items()
#             }

#     def get_snapshot(
#         self, room_id: str, camera_id: str
#     ) -> Tuple[Optional["cv2.typing.MatLike"], Optional[float]]:
#         with self._lock:
#             cameras = self._rooms.get(room_id)
#             if cameras is None:
#                 return None, None
#             entry = cameras.get(camera_id)
#             if entry is None:
#                 return None, None
#         return entry.ingestor.snapshot()

#     def _camera_status(self, metrics, timestamp: Optional[float]) -> Dict[str, object]:
#         now = time.time()
#         frame_age = None
#         if timestamp is not None:
#             frame_age = max(0.0, now - timestamp)

#         health = "down"
#         if metrics.is_running and frame_age is not None:
#             if frame_age <= self.stale_threshold_seconds:
#                 health = "healthy"
#             else:
#                 health = "degraded"
#         elif metrics.is_running:
#             health = "degraded"

#         return {
#             "is_running": metrics.is_running,
#             "last_frame_timestamp": timestamp,
#             "frame_age_seconds": frame_age,
#             "frames_received_total": metrics.frames_received_total,
#             "restart_count": metrics.restart_count,
#             "last_error_message": metrics.last_error_message,
#             "health": health,
#         }

#     def room_health(self, room_id: str) -> Optional[Dict[str, object]]:
#         with self._lock:
#             cameras = self._rooms.get(room_id)
#             if cameras is None:
#                 return None
#             camera_items = list(cameras.items())

#         statuses: Dict[str, object] = {}
#         overall = "healthy"
#         for camera_id, entry in camera_items:
#             _frame, ts = entry.ingestor.snapshot()
#             status = self._camera_status(entry.ingestor.metrics, ts)
#             statuses[camera_id] = status
#             if status["health"] == "down":
#                 overall = "down"
#             elif status["health"] == "degraded" and overall != "down":
#                 overall = "degraded"

#         return {"overall_status": overall, "cameras": statuses}

#     def health(self) -> Dict[str, object]:
#         with self._lock:
#             room_items = list(self._rooms.items())

#         rooms: Dict[str, object] = {}
#         overall = "healthy"
#         for room_id, cameras in room_items:
#             room_status = self.room_health(room_id)
#             rooms[room_id] = room_status
#             if room_status is None:
#                 continue
#             if room_status["overall_status"] == "down":
#                 overall = "down"
#             elif room_status["overall_status"] == "degraded" and overall != "down":
#                 overall = "degraded"

#         return {"overall_status": overall, "rooms": rooms}

#     def _persist_storage_locked(self) -> None:
#         payload: Dict[str, Dict[str, Dict[str, str]]] = {}
#         for room_id, cameras in self._rooms.items():
#             payload[room_id] = {
#                 camera_id: {"url": cam.url, "role": cam.role}
#                 for camera_id, cam in cameras.items()
#             }

#         os.makedirs(os.path.dirname(self.storage_path) or ".", exist_ok=True)
#         tmp_path = f"{self.storage_path}.tmp"
#         with open(tmp_path, "w", encoding="utf-8") as handle:
#             json.dump(payload, handle, indent=2, sort_keys=True)
#         os.replace(tmp_path, self.storage_path)

#     def _load_storage(self) -> None:
#         if not os.path.exists(self.storage_path):
#             return
#         try:
#             with open(self.storage_path, "r", encoding="utf-8") as handle:
#                 payload = json.load(handle)
#         except (OSError, json.JSONDecodeError):
#             return

#         if not isinstance(payload, dict):
#             return
#         for room_id, cameras in payload.items():
#             if not isinstance(room_id, str) or not isinstance(cameras, dict):
#                 continue
#             self._rooms[room_id] = {}
#             for camera_id, url in cameras.items():
#                 if not isinstance(camera_id, str):
#                     continue
#                 if isinstance(url, dict):
#                     camera_url = url.get("url")
#                     role = url.get("role", "other")
#                 else:
#                     camera_url = url
#                     role = "other"
#                 if not isinstance(camera_url, str):
#                     continue
#                 self._rooms[room_id][camera_id] = CameraEntry(
#                     url=camera_url,
#                     role=role if isinstance(role, str) else "other",
#                     ingestor=StreamIngestor(
#                         f"{room_id}:{camera_id}",
#                         camera_url,
#                         self.frame_width,
#                         self.frame_height,
#                         self.target_fps,
#                         self.backoff_min_seconds,
#                         self.backoff_max_seconds,
#                         self.read_timeout_seconds,
#                         self.use_ffmpeg,
#                         self.rtsp_transport,
#                     ),
#                 )


import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2

from app.stream_ingestor import StreamIngestor

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Camera entry
# ------------------------------------------------------------------
@dataclass
class CameraEntry:
    url: str
    role: str
    ingestor: StreamIngestor


# ------------------------------------------------------------------
# Stream Manager
# ------------------------------------------------------------------
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

        self._rooms: Dict[str, Dict[str, CameraEntry]] = {}
        self._lock = threading.Lock()

        self._load_storage()

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            for cameras in self._rooms.values():
                for entry in cameras.values():
                    entry.ingestor.start()
        logger.info("stream_manager.start rooms=%d", len(self._rooms))

    def stop(self) -> None:
        with self._lock:
            for cameras in self._rooms.values():
                for entry in cameras.values():
                    entry.ingestor.stop()
        logger.info("stream_manager.stop")

    # --------------------------------------------------------------
    # Room & camera management
    # --------------------------------------------------------------
    def register_room(self, room_id: str) -> bool:
        with self._lock:
            if room_id in self._rooms:
                return False
            self._rooms[room_id] = {}
            self._persist_storage_locked()
            logger.info("stream_manager.room_registered room_id=%s", room_id)
            return True

    def add_camera(self, room_id: str, camera_id: str, url: str, role: str) -> bool:
        with self._lock:
            if room_id not in self._rooms:
                return False

            cameras = self._rooms[room_id]
            if camera_id in cameras:
                return False

            ingestor = StreamIngestor(
                stream_id=f"{room_id}:{camera_id}",
                url=url,
                frame_width=self.frame_width,
                frame_height=self.frame_height,
                target_fps=self.target_fps,
                backoff_min_seconds=self.backoff_min_seconds,
                backoff_max_seconds=self.backoff_max_seconds,
                read_timeout_seconds=self.read_timeout_seconds,
                use_ffmpeg=self.use_ffmpeg,
            )

            cameras[camera_id] = CameraEntry(
                url=url,
                role=role,
                ingestor=ingestor,
            )

            ingestor.start()
            self._persist_storage_locked()
            logger.info(
                "stream_manager.camera_added room_id=%s camera_id=%s role=%s url=%s",
                room_id,
                camera_id,
                role,
                url,
            )
            return True

    def remove_camera(self, room_id: str, camera_id: str) -> bool:
        with self._lock:
            cameras = self._rooms.get(room_id)
            if cameras is None:
                return False

            entry = cameras.pop(camera_id, None)
            if entry is None:
                return False

            entry.ingestor.stop()
            self._persist_storage_locked()
            logger.info(
                "stream_manager.camera_removed room_id=%s camera_id=%s",
                room_id,
                camera_id,
            )
            return True

    def remove_room(self, room_id: str) -> bool:
        with self._lock:
            cameras = self._rooms.pop(room_id, None)
            if cameras is None:
                return False

            for entry in cameras.values():
                entry.ingestor.stop()

            self._persist_storage_locked()
            logger.info("stream_manager.room_removed room_id=%s", room_id)
            return True

    # --------------------------------------------------------------
    # Queries
    # --------------------------------------------------------------
    def list_rooms(self) -> Dict[str, object]:
        with self._lock:
            return {
                room_id: {camera_id: entry.role for camera_id, entry in cameras.items()}
                for room_id, cameras in self._rooms.items()
            }

    def list_camera_entries(self) -> Dict[str, Dict[str, CameraEntry]]:
        with self._lock:
            return {room_id: dict(cameras) for room_id, cameras in self._rooms.items()}

    def get_snapshot(
        self,
        room_id: str,
        camera_id: str,
    ) -> Tuple[Optional["cv2.typing.MatLike"], Optional[float]]:
        with self._lock:
            cameras = self._rooms.get(room_id)
            if cameras is None:
                logger.debug(
                    "stream_manager.snapshot_miss room_id=%s camera_id=%s reason=room_missing",
                    room_id,
                    camera_id,
                )
                return None, None

            entry = cameras.get(camera_id)
            if entry is None:
                logger.debug(
                    "stream_manager.snapshot_miss room_id=%s camera_id=%s reason=camera_missing",
                    room_id,
                    camera_id,
                )
                return None, None

        frame, ts = entry.ingestor.snapshot()
        if frame is None or ts is None:
            logger.debug(
                "stream_manager.snapshot_empty room_id=%s camera_id=%s",
                room_id,
                camera_id,
            )
        else:
            logger.debug(
                "stream_manager.snapshot_ok room_id=%s camera_id=%s ts=%.3f",
                room_id,
                camera_id,
                ts,
            )
        return frame, ts

    # --------------------------------------------------------------
    # Health checks
    # --------------------------------------------------------------
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

        for camera_id, entry in camera_items:
            _, ts = entry.ingestor.snapshot()
            status = self._camera_status(entry.ingestor.metrics, ts)
            statuses[camera_id] = status

            if status["health"] == "down":
                overall = "down"
            elif status["health"] == "degraded" and overall != "down":
                overall = "degraded"

        return {
            "overall_status": overall,
            "cameras": statuses,
        }

    def health(self) -> Dict[str, object]:
        with self._lock:
            room_items = list(self._rooms.items())

        rooms: Dict[str, object] = {}
        overall = "healthy"

        for room_id, _ in room_items:
            room_status = self.room_health(room_id)
            rooms[room_id] = room_status

            if room_status is None:
                continue
            if room_status["overall_status"] == "down":
                overall = "down"
            elif room_status["overall_status"] == "degraded" and overall != "down":
                overall = "degraded"

        return {
            "overall_status": overall,
            "rooms": rooms,
        }

    # --------------------------------------------------------------
    # Persistence
    # --------------------------------------------------------------
    def _persist_storage_locked(self) -> None:
        payload: Dict[str, Dict[str, Dict[str, str]]] = {}

        for room_id, cameras in self._rooms.items():
            payload[room_id] = {
                camera_id: {
                    "url": cam.url,
                    "role": cam.role,
                }
                for camera_id, cam in cameras.items()
            }

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

            for camera_id, data in cameras.items():
                if not isinstance(camera_id, str):
                    continue

                if isinstance(data, dict):
                    camera_url = data.get("url")
                    role = data.get("role", "other")
                else:
                    camera_url = data
                    role = "other"

                if not isinstance(camera_url, str):
                    continue

                self._rooms[room_id][camera_id] = CameraEntry(
                    url=camera_url,
                    role=role if isinstance(role, str) else "other",
                    ingestor=StreamIngestor(
                        stream_id=f"{room_id}:{camera_id}",
                        url=camera_url,
                        frame_width=self.frame_width,
                        frame_height=self.frame_height,
                        target_fps=self.target_fps,
                        backoff_min_seconds=self.backoff_min_seconds,
                        backoff_max_seconds=self.backoff_max_seconds,
                        read_timeout_seconds=self.read_timeout_seconds,
                        use_ffmpeg=self.use_ffmpeg,
                    ),
                )
