import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

import cv2
import numpy as np

from app.stream_manager import CameraEntry, StreamManager

logger = logging.getLogger(__name__)

@dataclass
class CameraGateState:
    room_id: str
    camera_id: str
    role: str
    sample_interval: float
    window_size: int
    active_enter: float
    active_exit: float
    spike_enter: float
    spike_exit: float
    spike_cooldown_seconds: float
    stale_seconds: float
    downsample_width: int
    downsample_height: int
    state: str = "IDLE"
    last_motion_score: float = 0.0
    last_state_change: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    last_spike_time: Optional[float] = None
    window: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    prev_frame: Optional[np.ndarray] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None


class MotionGateManager:
    def __init__(
        self,
        stream_manager: StreamManager,
        sample_interval: float,
        window_size: int,
        active_enter: float,
        active_exit: float,
        spike_enter: float,
        spike_exit: float,
        spike_cooldown_seconds: float,
        stale_seconds: float,
        downsample_width: int,
        downsample_height: int,
    ) -> None:
        self.stream_manager = stream_manager
        self.sample_interval = sample_interval
        self.window_size = window_size
        self.active_enter = active_enter
        self.active_exit = active_exit
        self.spike_enter = spike_enter
        self.spike_exit = spike_exit
        self.spike_cooldown_seconds = spike_cooldown_seconds
        self.stale_seconds = stale_seconds
        self.downsample_width = downsample_width
        self.downsample_height = downsample_height
        self._gates: Dict[str, Dict[str, CameraGateState]] = {}
        self._lock = threading.Lock()

    def bootstrap_from_stream_manager(self) -> None:
        entries = self.stream_manager.list_camera_entries()
        for room_id, cameras in entries.items():
            for camera_id, entry in cameras.items():
                self.add_camera(room_id, camera_id, entry.role)

    def add_camera(self, room_id: str, camera_id: str, role: str) -> None:
        with self._lock:
            room = self._gates.setdefault(room_id, {})
            if camera_id in room:
                return
            gate = CameraGateState(
                room_id=room_id,
                camera_id=camera_id,
                role=role,
                sample_interval=self.sample_interval,
                window_size=self.window_size,
                active_enter=self.active_enter,
                active_exit=self.active_exit,
                spike_enter=self.spike_enter,
                spike_exit=self.spike_exit,
                spike_cooldown_seconds=self.spike_cooldown_seconds,
                stale_seconds=self.stale_seconds,
                downsample_width=self.downsample_width,
                downsample_height=self.downsample_height,
                window=deque(maxlen=self.window_size),
            )
            gate.thread = threading.Thread(
                target=self._run_gate, args=(gate,), daemon=True
            )
            gate.thread.start()
            room[camera_id] = gate

    def remove_camera(self, room_id: str, camera_id: str) -> None:
        with self._lock:
            room = self._gates.get(room_id)
            if room is None:
                return
            gate = room.pop(camera_id, None)
        if gate is None:
            return
        gate.stop_event.set()
        if gate.thread and gate.thread.is_alive():
            gate.thread.join(timeout=2.0)

    def remove_room(self, room_id: str) -> None:
        with self._lock:
            room = self._gates.pop(room_id, None)
        if room is None:
            return
        for gate in room.values():
            gate.stop_event.set()
            if gate.thread and gate.thread.is_alive():
                gate.thread.join(timeout=2.0)

    def activity(self, room_id: str, camera_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            room = self._gates.get(room_id)
            if room is None:
                return None
            gate = room.get(camera_id)
        if gate is None:
            return None
        with gate.lock:
            return self._serialize_gate(gate)

    def room_activity(self, room_id: str) -> Optional[Dict[str, object]]:
        with self._lock:
            room = self._gates.get(room_id)
            if room is None:
                return None
            gates = list(room.values())
        cameras: Dict[str, object] = {}
        for gate in gates:
            with gate.lock:
                cameras[gate.camera_id] = self._serialize_gate(gate)
        return {"cameras": cameras}

    def all_activity(self) -> Dict[str, object]:
        with self._lock:
            room_items = list(self._gates.items())
        output: Dict[str, object] = {}
        for room_id, gates in room_items:
            cameras: Dict[str, object] = {}
            for gate in gates.values():
                with gate.lock:
                    cameras[gate.camera_id] = self._serialize_gate(gate)
            output[room_id] = {"cameras": cameras}
        return {"rooms": output}

    def _serialize_gate(self, gate: CameraGateState) -> Dict[str, object]:
        now = time.time()
        return {
            "camera_id": gate.camera_id,
            "camera_role": gate.role,
            "activity_state": gate.state,
            "motion_score": gate.last_motion_score,
            "timestamp": gate.last_update,
            "seconds_since_state_change": max(0.0, now - gate.last_state_change),
        }

    def _run_gate(self, gate: CameraGateState) -> None:
        next_time = time.monotonic()
        while not gate.stop_event.is_set():
            now = time.monotonic()
            if now < next_time:
                time.sleep(min(0.1, next_time - now))
                continue
            next_time = now + gate.sample_interval

            frame, frame_ts = self.stream_manager.get_snapshot(
                gate.room_id, gate.camera_id
            )
            if frame_ts is None or frame is None:
                logger.debug(
                    "motion_gate.frame_missing room_id=%s camera_id=%s",
                    gate.room_id,
                    gate.camera_id,
                )
                self._set_idle(gate)
                continue

            age = time.time() - frame_ts
            if age > gate.stale_seconds:
                logger.debug(
                    "motion_gate.frame_stale room_id=%s camera_id=%s age=%.2f",
                    gate.room_id,
                    gate.camera_id,
                    age,
                )
                self._set_idle(gate)
                continue

            processed = self._preprocess(frame, gate.downsample_width, gate.downsample_height)
            if processed is None:
                logger.debug(
                    "motion_gate.preprocess_failed room_id=%s camera_id=%s",
                    gate.room_id,
                    gate.camera_id,
                )
                self._set_idle(gate)
                continue

            with gate.lock:
                if gate.prev_frame is None:
                    gate.prev_frame = processed
                    gate.last_motion_score = 0.0
                    gate.last_update = time.time()
                    gate.window.append(0.0)
                    self._transition_state(gate, 0.0)
                    continue

                diff = np.abs(processed - gate.prev_frame)
                score = float(np.mean(diff))
                score = max(0.0, min(1.0, score))
                gate.prev_frame = processed
                gate.window.append(score)
                aggregated = float(np.mean(gate.window)) if gate.window else score
                gate.last_motion_score = aggregated
                gate.last_update = time.time()
                self._transition_state(gate, aggregated)

    def _set_idle(self, gate: CameraGateState) -> None:
        with gate.lock:
            gate.prev_frame = None
            gate.window.clear()
            gate.last_motion_score = 0.0
            gate.last_update = time.time()
            if gate.state != "IDLE":
                gate.state = "IDLE"
                gate.last_state_change = gate.last_update
                gate.last_spike_time = None

    def _transition_state(self, gate: CameraGateState, score: float) -> None:
        now = time.time()
        prev_state = gate.state
        state = gate.state
        if state == "SPIKE":
            if gate.last_spike_time is not None:
                if now - gate.last_spike_time < gate.spike_cooldown_seconds:
                    return
            if score >= gate.spike_exit:
                return
            if score >= gate.active_enter:
                gate.state = "ACTIVE"
            else:
                gate.state = "IDLE"
            gate.last_state_change = now
            if gate.state != prev_state:
                logger.info(
                    "motion_gate.state room_id=%s camera_id=%s %s->%s score=%.3f",
                    gate.room_id,
                    gate.camera_id,
                    prev_state,
                    gate.state,
                    score,
                )
            return

        if score >= gate.spike_enter:
            gate.state = "SPIKE"
            gate.last_spike_time = now
            gate.last_state_change = now
            if gate.state != prev_state:
                logger.info(
                    "motion_gate.state room_id=%s camera_id=%s %s->%s score=%.3f",
                    gate.room_id,
                    gate.camera_id,
                    prev_state,
                    gate.state,
                    score,
                )
            return

        if state == "ACTIVE":
            if score < gate.active_exit:
                gate.state = "IDLE"
                gate.last_state_change = now
                if gate.state != prev_state:
                    logger.info(
                        "motion_gate.state room_id=%s camera_id=%s %s->%s score=%.3f",
                        gate.room_id,
                        gate.camera_id,
                        prev_state,
                        gate.state,
                        score,
                    )
            return

        if score >= gate.active_enter:
            gate.state = "ACTIVE"
            gate.last_state_change = now
            if gate.state != prev_state:
                logger.info(
                    "motion_gate.state room_id=%s camera_id=%s %s->%s score=%.3f",
                    gate.room_id,
                    gate.camera_id,
                    prev_state,
                    gate.state,
                    score,
                )

    @staticmethod
    def _preprocess(
        frame: "cv2.typing.MatLike", width: int, height: int
    ) -> Optional[np.ndarray]:
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except Exception:
            return None
        small = cv2.resize(gray, (width, height))
        small = small.astype(np.float32)
        mean = float(np.mean(small))
        std = float(np.std(small))
        if std < 1e-6:
            std = 1.0
        normalized = (small - mean) / std
        normalized = np.clip(np.abs(normalized) / 3.0, 0.0, 1.0)
        return normalized
