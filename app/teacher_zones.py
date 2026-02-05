import json
import os
import threading
from typing import Dict, Optional


class TeacherZoneConfig:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._mtime: Optional[float] = None
        self._default = "top"
        self._by_role: Dict[str, str] = {}
        self._by_camera: Dict[str, str] = {}

    def direction_for(self, room_id: str, camera_id: str, role: str) -> str:
        self._load_if_needed()
        key = f"{room_id}:{camera_id}"
        with self._lock:
            if key in self._by_camera:
                return self._by_camera[key]
            if role in self._by_role:
                return self._by_role[role]
            return self._default

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
        default = payload.get("default")
        by_role = payload.get("by_role", {})
        by_camera = payload.get("by_camera", {})
        loaded_role: Dict[str, str] = {}
        loaded_camera: Dict[str, str] = {}
        if isinstance(by_role, dict):
            for key, val in by_role.items():
                if isinstance(key, str) and isinstance(val, str):
                    loaded_role[key] = val
        if isinstance(by_camera, dict):
            for key, val in by_camera.items():
                if isinstance(key, str) and isinstance(val, str):
                    loaded_camera[key] = val
        with self._lock:
            if isinstance(default, str) and default:
                self._default = default
            self._by_role = loaded_role
            self._by_camera = loaded_camera
            try:
                self._mtime = os.path.getmtime(self.path)
            except OSError:
                self._mtime = None
