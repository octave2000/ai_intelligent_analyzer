import logging
import math
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.face_identifier import FaceIdentifier, FaceMatch
from app.attendance_manager import AttendanceManager
from app.yolo_detector import YoloDetector, YoloDetection
from app.overlay_store import OverlayStore
from app.stream_manager import StreamManager
from app.attention_manager import TrackSummary

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float


@dataclass
class ObjectDetection:
    object_type: str
    category: str
    risk_level: str
    confidence: float
    bbox: Tuple[int, int, int, int]


@dataclass
class TrackState:
    track_id: int
    bbox: Tuple[int, int, int, int]
    last_seen: float
    hits: int = 1
    history: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))
    appearance: Optional[np.ndarray] = None
    global_id: Optional[int] = None
    role: str = "unknown"
    role_confidence: float = 0.0
    last_orientation: Optional[str] = None
    identity_id: Optional[str] = None
    identity_name: Optional[str] = None
    identity_role: Optional[str] = None
    identity_score: float = 0.0
    last_identity_time: float = 0.0
    last_body_movement_emit: float = 0.0
    posture: str = "upright"
    upright_height_ema: float = 0.0
    down_since: float = 0.0
    bowing_since: float = 0.0
    role_teacher_evidence: float = 0.0
    role_student_evidence: float = 0.0
    last_sleep_emit: float = 0.0
    last_device_emit: float = 0.0
    last_phone_emit: float = 0.0


@dataclass
class ObjectTrack:
    track_id: int
    detection: ObjectDetection
    last_seen: float
    hits: int = 1
    emitted: bool = False


@dataclass
class GlobalPerson:
    global_id: int
    appearance: np.ndarray
    last_seen: float


@dataclass
class CameraPerceptionState:
    room_id: str
    camera_id: str
    role: str = "other"
    tracks: Dict[int, TrackState] = field(default_factory=dict)
    object_tracks: Dict[int, ObjectTrack] = field(default_factory=dict)
    next_track_id: int = 1
    next_object_id: int = 1
    last_run: float = 0.0
    last_frame_source_timestamp: Optional[float] = None
    last_frame_timestamp: Optional[float] = None
    last_frame_timestamp_offset_seconds: float = 0.0
    last_frame_timestamp_stabilizer_skew_seconds: float = 0.0
    last_frame_age_seconds: Optional[float] = None
    last_frame_transport_delay_seconds: Optional[float] = None
    timestamp_delay_ema_seconds: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    proximity_state: Dict[Tuple[int, int], Tuple[bool, float, bool]] = field(default_factory=dict)
    group_state: Dict[frozenset, Tuple[int, float, bool]] = field(default_factory=dict)
    next_group_id: int = 1
    association_last: Dict[Tuple[int, int], float] = field(default_factory=dict)
    last_attempt_at: float = 0.0
    last_processed_at: float = 0.0
    last_processing_ms: float = 0.0
    last_error: Optional[str] = None
    last_skip_event_at: float = 0.0
    last_inflight_tick_at: float = 0.0
    last_face_infer_at: float = 0.0
    in_flight: bool = False
    yolo_cached: List[YoloDetection] = field(default_factory=list)
    yolo_cached_at: float = 0.0
    yolo_in_flight: bool = False
    last_yolo_submit_at: float = 0.0
    last_yolo_inference_ms: float = 0.0


class GlobalIdentityResolver:
    def __init__(self, similarity_threshold: float, max_age_seconds: float) -> None:
        self._next_id = 1
        self._people: Dict[int, GlobalPerson] = {}
        self._lock = threading.Lock()
        self._similarity_threshold = similarity_threshold
        self._max_age_seconds = max_age_seconds

    def assign(self, camera_id: str, appearance: Optional[np.ndarray]) -> int:
        now = time.time()
        if appearance is None:
            with self._lock:
                new_id = self._next_id
                self._next_id += 1
                self._people[new_id] = GlobalPerson(new_id, np.zeros((1, 1)), now)
                return new_id

        with self._lock:
            best_id = None
            best_score = 0.0
            for pid, person in self._people.items():
                if now - person.last_seen > self._max_age_seconds:
                    continue
                score = _hist_similarity(person.appearance, appearance)
                if score > best_score:
                    best_score = score
                    best_id = pid

            if best_id is None or best_score < self._similarity_threshold:
                new_id = self._next_id
                self._next_id += 1
                self._people[new_id] = GlobalPerson(new_id, appearance, now)
                return new_id

            person = self._people[best_id]
            person.appearance = 0.7 * person.appearance + 0.3 * appearance
            person.last_seen = now
            return best_id

    def refresh(self, global_id: int, appearance: Optional[np.ndarray]) -> None:
        if appearance is None:
            return
        with self._lock:
            person = self._people.get(global_id)
            if person is None:
                return
            person.appearance = 0.8 * person.appearance + 0.2 * appearance
            person.last_seen = time.time()


class PerceptionManager:
    def __init__(
        self,
        stream_manager: StreamManager,
        active_interval_seconds: float,
        stale_seconds: float,
        track_ttl_seconds: float,
        object_ttl_seconds: float,
        object_persist_frames: int,
        person_iou_threshold: float,
        object_iou_threshold: float,
        global_similarity_threshold: float,
        global_max_age_seconds: float,
        uniform_hsv_low: Tuple[int, int, int],
        uniform_hsv_high: Tuple[int, int, int],
        uniform_min_ratio: float,
        student_top_hsv_low: Tuple[int, int, int],
        student_top_hsv_high: Tuple[int, int, int],
        student_bottom_hsv_low: Tuple[int, int, int],
        student_bottom_hsv_high: Tuple[int, int, int],
        student_bottom_hsv_low_2: Tuple[int, int, int],
        student_bottom_hsv_high_2: Tuple[int, int, int],
        student_top_min_ratio: float,
        student_bottom_min_ratio: float,
        student_top_only_min_ratio: float,
        student_top_only_max_bottom_ratio: float,
        student_seated_max_height_ratio: float,
        teacher_min_hits: int,
        role_decision_margin: float,
        teacher_height_ratio: float,
        orientation_motion_threshold: float,
        body_movement_enabled: bool,
        body_movement_min_delta_pixels: float,
        body_movement_emit_interval_seconds: float,
        posture_height_ema_alpha: float,
        sleep_bow_ratio_threshold: float,
        sleep_bow_aspect_min: float,
        sleep_min_seconds: float,
        sleep_emit_interval_seconds: float,
        device_usage_emit_interval_seconds: float,
        phone_usage_emit_interval_seconds: float,
        identity_min_interval_seconds: float,
        identity_sticky_score: float,
        proximity_distance_ratio: float,
        proximity_duration_seconds: float,
        group_distance_ratio: float,
        group_duration_seconds: float,
        detection_width: int,
        detection_height: int,
        event_queue_maxlen: int,
        exam_mode: bool,
        max_cameras_per_tick: int,
        event_max_frame_age_seconds: float,
        event_timestamp_offset_seconds: float,
        event_timestamp_stabilize_alpha: float,
        event_timestamp_stabilize_max_correction_seconds: float,
        event_timestamp_round_seconds: float,
        dual_detect_test: bool,
        people_detector_mode: str = "auto",
        yolo_workers: int = 1,
        yolo_submit_interval_seconds: float = 0.8,
        yolo_cache_ttl_seconds: float = 0.0,
        pipeline_tag: str = "p1",
        face_identifier: Optional[FaceIdentifier] = None,
        attendance: Optional[AttendanceManager] = None,
        yolo_detector: Optional[YoloDetector] = None,
        overlay_store: Optional[OverlayStore] = None,
        object_allowlist: Tuple[str, ...] = (),
        object_priority: Tuple[str, ...] = (),
        object_risky: Tuple[str, ...] = (),
        object_label_map: Optional[Dict[str, Dict[str, str]]] = None,
        attention_manager: Optional[object] = None,
    ) -> None:
        self.stream_manager = stream_manager
        self.active_interval_seconds = active_interval_seconds
        self.stale_seconds = stale_seconds
        self.track_ttl_seconds = track_ttl_seconds
        self.object_ttl_seconds = object_ttl_seconds
        self.object_persist_frames = object_persist_frames
        self.person_iou_threshold = person_iou_threshold
        self.object_iou_threshold = object_iou_threshold
        self.uniform_hsv_low = uniform_hsv_low
        self.uniform_hsv_high = uniform_hsv_high
        self.uniform_min_ratio = uniform_min_ratio
        self.student_top_hsv_low = student_top_hsv_low
        self.student_top_hsv_high = student_top_hsv_high
        self.student_bottom_hsv_low = student_bottom_hsv_low
        self.student_bottom_hsv_high = student_bottom_hsv_high
        self.student_bottom_hsv_low_2 = student_bottom_hsv_low_2
        self.student_bottom_hsv_high_2 = student_bottom_hsv_high_2
        self.student_top_min_ratio = max(0.01, min(1.0, student_top_min_ratio))
        self.student_bottom_min_ratio = max(0.01, min(1.0, student_bottom_min_ratio))
        self.student_top_only_min_ratio = max(
            0.01, min(1.0, student_top_only_min_ratio)
        )
        self.student_top_only_max_bottom_ratio = max(
            0.0, min(1.0, student_top_only_max_bottom_ratio)
        )
        self.student_seated_max_height_ratio = max(
            0.05, min(1.0, student_seated_max_height_ratio)
        )
        self.teacher_min_hits = max(1, teacher_min_hits)
        self.role_decision_margin = max(0.0, min(0.4, role_decision_margin))
        self.teacher_height_ratio = max(0.05, min(1.0, teacher_height_ratio))
        self.orientation_motion_threshold = orientation_motion_threshold
        self.body_movement_enabled = body_movement_enabled
        self.body_movement_min_delta_pixels = max(0.0, body_movement_min_delta_pixels)
        self.body_movement_emit_interval_seconds = max(
            0.0, body_movement_emit_interval_seconds
        )
        self.posture_height_ema_alpha = max(0.01, min(1.0, posture_height_ema_alpha))
        self.sleep_bow_ratio_threshold = max(0.2, min(1.5, sleep_bow_ratio_threshold))
        self.sleep_bow_aspect_min = max(0.05, min(2.0, sleep_bow_aspect_min))
        self.sleep_min_seconds = max(0.1, sleep_min_seconds)
        self.sleep_emit_interval_seconds = max(0.0, sleep_emit_interval_seconds)
        self.device_usage_emit_interval_seconds = max(
            0.0, device_usage_emit_interval_seconds
        )
        self.phone_usage_emit_interval_seconds = max(
            0.0, phone_usage_emit_interval_seconds
        )
        self.identity_min_interval_seconds = max(0.05, identity_min_interval_seconds)
        self.face_detect_interval_seconds = max(
            0.6, self.identity_min_interval_seconds * 2.0
        )
        self.identity_sticky_score = max(0.0, min(1.0, identity_sticky_score))
        self.proximity_distance_ratio = proximity_distance_ratio
        self.proximity_duration_seconds = proximity_duration_seconds
        self.group_distance_ratio = group_distance_ratio
        self.group_duration_seconds = group_duration_seconds
        self.detection_width = detection_width
        self.detection_height = detection_height
        self.event_queue_maxlen = max(1000, event_queue_maxlen)
        self.exam_mode = exam_mode
        self.max_cameras_per_tick = max(1, max_cameras_per_tick)
        self.event_max_frame_age_seconds = max(0.0, event_max_frame_age_seconds)
        self.event_timestamp_offset_seconds = event_timestamp_offset_seconds
        self.event_timestamp_stabilize_alpha = max(
            0.0, min(1.0, event_timestamp_stabilize_alpha)
        )
        self.event_timestamp_stabilize_max_correction_seconds = max(
            0.0, event_timestamp_stabilize_max_correction_seconds
        )
        self.event_timestamp_round_seconds = max(0.0, event_timestamp_round_seconds)
        self.dual_detect_test = dual_detect_test
        detector_mode = people_detector_mode.strip().lower()
        if detector_mode not in ("auto", "yolo_only", "hog_only"):
            detector_mode = "auto"
        self.people_detector_mode = detector_mode
        self.yolo_workers = max(1, yolo_workers)
        self.yolo_submit_interval_seconds = max(0.05, yolo_submit_interval_seconds)
        if yolo_cache_ttl_seconds <= 0.0:
            self.yolo_cache_ttl_seconds = max(1.0, self.active_interval_seconds * 8.0)
        else:
            self.yolo_cache_ttl_seconds = max(0.5, yolo_cache_ttl_seconds)
        self.pipeline_tag = pipeline_tag
        self.face_identifier = face_identifier
        self.attendance = attendance
        self.yolo_detector = yolo_detector
        self.overlay_store = overlay_store
        self.object_allowlist = {
            item.strip().lower()
            for item in object_allowlist
            if isinstance(item, str) and item.strip()
        }
        self.object_priority = {
            item.strip().lower()
            for item in object_priority
            if isinstance(item, str) and item.strip()
        }
        self.object_risky = {
            item.strip().lower()
            for item in object_risky
            if isinstance(item, str) and item.strip()
        }
        self.object_label_map = object_label_map or {}
        self.attention_manager = attention_manager

        self._cameras: Dict[str, Dict[str, CameraPerceptionState]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._process_pool: Optional[ThreadPoolExecutor] = None
        self._yolo_pool: Optional[ThreadPoolExecutor] = None
        self._yolo_submit_interval_seconds = self.yolo_submit_interval_seconds
        self._yolo_cache_ttl_seconds = self.yolo_cache_ttl_seconds
        self._events: Deque[Dict[str, object]] = deque(maxlen=self.event_queue_maxlen)
        self._events_lock = threading.Lock()
        self._resolver = GlobalIdentityResolver(
            similarity_threshold=global_similarity_threshold,
            max_age_seconds=global_max_age_seconds,
        )
        self._camera_order: List[Tuple[str, str]] = []
        self._camera_index = 0
        self._person_id_map: Dict[str, str] = {}
        self._hog = cv2.HOGDescriptor()
        detector_fn = getattr(cv2, "HOGDescriptor_getDefaultPeopleDetector", None)
        if detector_fn is not None:
            try:
                self._hog.setSVMDetector(detector_fn())
            except Exception:
                pass

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        if self._process_pool is None:
            self._process_pool = ThreadPoolExecutor(
                max_workers=self.max_cameras_per_tick,
                thread_name_prefix="perception-worker",
            )
        if self._yolo_pool is None and self.yolo_detector is not None and self.yolo_detector.ready():
            self._yolo_pool = ThreadPoolExecutor(
                max_workers=self.yolo_workers,
                thread_name_prefix="perception-yolo",
            )
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        pool = self._process_pool
        self._process_pool = None
        if pool is not None:
            pool.shutdown(wait=True)
        yolo_pool = self._yolo_pool
        self._yolo_pool = None
        if yolo_pool is not None:
            yolo_pool.shutdown(wait=True)

    def bootstrap_from_stream_manager(self) -> None:
        entries = self.stream_manager.list_camera_entries()
        for room_id, cameras in entries.items():
            for camera_id, entry in cameras.items():
                self.add_camera(room_id, camera_id, entry.role)

    def add_camera(self, room_id: str, camera_id: str, role: str = "other") -> None:
        with self._lock:
            room = self._cameras.setdefault(room_id, {})
            if camera_id in room:
                return
            room[camera_id] = CameraPerceptionState(
                room_id=room_id,
                camera_id=camera_id,
                role=role,
            )
            self._refresh_camera_order_locked()

    def remove_camera(self, room_id: str, camera_id: str) -> None:
        with self._lock:
            room = self._cameras.get(room_id)
            if room is None:
                return
            room.pop(camera_id, None)
            if not room:
                self._cameras.pop(room_id, None)
            self._refresh_camera_order_locked()

    def remove_room(self, room_id: str) -> None:
        with self._lock:
            self._cameras.pop(room_id, None)
            self._refresh_camera_order_locked()

    def get_events(
        self,
        limit: int = 200,
        since: Optional[float] = None,
        room_id: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        limit = max(1, min(1000, limit))
        results: List[Dict[str, object]] = []
        with self._events_lock:
            events = list(self._events)
        for event in reversed(events):
            if since is not None and _event_cursor_ts(event) <= since:
                continue
            if room_id is not None:
                event_room = event.get("room_id")
                if not isinstance(event_room, str) or event_room != room_id:
                    continue
            if camera_id is not None:
                event_camera = event.get("camera_id")
                if not isinstance(event_camera, str) or event_camera != camera_id:
                    continue
            results.append(event)
            if len(results) >= limit:
                break
        results.reverse()
        return results

    def health(self) -> Dict[str, object]:
        with self._lock:
            room_items = list(self._cameras.items())
        with self._events_lock:
            event_count = len(self._events)
        rooms: Dict[str, object] = {}
        total_cameras = 0
        for room_id, cameras in room_items:
            cam_payload: Dict[str, object] = {}
            for camera_id, state in cameras.items():
                with state.lock:
                    cam_payload[camera_id] = {
                        "last_attempt_at": state.last_attempt_at,
                        "last_processed_at": state.last_processed_at,
                        "last_processing_ms": state.last_processing_ms,
                        "last_error": state.last_error,
                        "tracks": len(state.tracks),
                        "object_tracks": len(state.object_tracks),
                    }
            rooms[room_id] = {"cameras": cam_payload}
            total_cameras += len(cameras)
        return {
            "event_queue_size": event_count,
            "camera_count": total_cameras,
            "rooms": rooms,
        }

    def _refresh_camera_order_locked(self) -> None:
        self._camera_order = [
            (room_id, camera_id)
            for room_id, cameras in self._cameras.items()
            for camera_id in cameras.keys()
        ]
        if self._camera_order:
            self._camera_index %= len(self._camera_order)
        else:
            self._camera_index = 0

    def _emit(
        self,
        event: Dict[str, object],
        frame: Optional["cv2.typing.MatLike"] = None,
    ) -> None:
        if "emitted_at" not in event:
            event["emitted_at"] = time.time()
        event["pipeline"] = self.pipeline_tag
        with self._events_lock:
            self._events.append(event)
        logger.info(
            "perception.event room_id=%s camera_id=%s event_type=%s",
            event.get("room_id"),
            event.get("camera_id"),
            event.get("event_type"),
        )
        if self.overlay_store is not None:
            room_id = event.get("room_id")
            camera_id = event.get("camera_id")
            if isinstance(room_id, str) and isinstance(camera_id, str):
                self.overlay_store.add_event(room_id, camera_id, event, frame=frame)

    def _emit_frame_skipped(
        self,
        state: CameraPerceptionState,
        reason: str,
        confidence: float = 0.35,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        now = time.time()
        with state.lock:
            if now - state.last_skip_event_at < 1.0:
                return
            state.last_skip_event_at = now
        payload: Dict[str, object] = {
            "timestamp": now,
            "room_id": state.room_id,
            "camera_id": state.camera_id,
            "global_person_id": None,
            "person_id": None,
            "event_type": "frame_skipped",
            "confidence": float(max(0.0, min(1.0, confidence))),
            "skip_reason": reason,
        }
        if extra:
            payload.update(extra)
        self._emit(payload)

    def _emit_inflight_tick(
        self,
        state: CameraPerceptionState,
        in_flight_seconds: float,
    ) -> None:
        now = time.time()
        with state.lock:
            if now - state.last_inflight_tick_at < 1.0:
                return
            state.last_inflight_tick_at = now
            detections_count = len(state.tracks)
            objects_count = len(state.object_tracks)
            last_processing_ms = state.last_processing_ms
            last_frame_timestamp = state.last_frame_timestamp
            last_frame_age_seconds = state.last_frame_age_seconds
        payload: Dict[str, object] = {
            "timestamp": now,
            "room_id": state.room_id,
            "camera_id": state.camera_id,
            "global_person_id": None,
            "person_id": None,
            "event_type": "frame_tick",
            "confidence": 1.0,
            "detections_count": detections_count,
            "objects_count": objects_count,
            "processing_in_flight": True,
            "in_flight_seconds": round(in_flight_seconds, 3),
            "last_processing_ms": round(last_processing_ms, 2),
            "person_detection_source": "cached_tracks",
        }
        if last_frame_timestamp is not None:
            payload["frame_timestamp"] = last_frame_timestamp
        if last_frame_age_seconds is not None:
            payload["frame_age_seconds"] = last_frame_age_seconds
        self._emit(payload)

    def emit_external_event(
        self,
        event: Dict[str, object],
        frame: Optional["cv2.typing.MatLike"] = None,
    ) -> None:
        if "emitted_at" not in event:
            event["emitted_at"] = time.time()
        if "pipeline" not in event:
            event["pipeline"] = self.pipeline_tag
        with self._events_lock:
            self._events.append(event)
        if self.overlay_store is not None:
            room_id = event.get("room_id")
            camera_id = event.get("camera_id")
            if isinstance(room_id, str) and isinstance(camera_id, str):
                self.overlay_store.add_event(room_id, camera_id, event, frame=frame)

    @staticmethod
    def _unknown_person_id(track: TrackState) -> Optional[str]:
        if track.identity_id:
            return None
        if track.global_id is None:
            return None
        return f"unknown:{track.global_id}"

    def _person_id_for_track(self, track: TrackState) -> Optional[str]:
        if track.identity_id:
            return track.identity_id
        unknown_id = self._unknown_person_id(track)
        if unknown_id is None:
            return None
        mapped = self._person_id_map.get(unknown_id)
        return mapped or unknown_id

    def _run(self) -> None:
        while not self._stop_event.is_set():
            with self._lock:
                camera_order = list(self._camera_order)
                start_index = self._camera_index

            now = time.monotonic()
            dispatched = 0
            checked = 0
            total = len(camera_order)
            idx = start_index
            while checked < total and dispatched < self.max_cameras_per_tick:
                room_id, camera_id = camera_order[idx]
                idx = (idx + 1) % total
                checked += 1
                with self._lock:
                    state = self._cameras.get(room_id, {}).get(camera_id)
                if state is None:
                    continue
                interval = self.active_interval_seconds
                in_flight_for = 0.0
                with state.lock:
                    if state.in_flight:
                        base_attempt = state.last_attempt_at if state.last_attempt_at > 0.0 else state.last_run
                        in_flight_for = max(0.0, now - base_attempt)
                        logger.debug(
                            "perception.skip room_id=%s camera_id=%s reason=in_flight",
                            room_id,
                            camera_id,
                        )
                    elif now - state.last_run < interval:
                        logger.info(
                            "perception.skip room_id=%s camera_id=%s reason=interval_throttle",
                            room_id,
                            camera_id,
                        )
                        continue
                    else:
                        state.last_run = now
                        state.in_flight = True
                if in_flight_for > 0.0:
                    if in_flight_for >= 1.0:
                        self._emit_inflight_tick(state, in_flight_for)
                    continue
                logger.debug(
                    "perception.process_dispatch room_id=%s camera_id=%s",
                    room_id,
                    camera_id,
                )
                if not self._submit_camera(state):
                    with state.lock:
                        state.in_flight = False
                    continue
                dispatched += 1
            with self._lock:
                if total > 0:
                    self._camera_index = idx
                if total > dispatched and dispatched >= self.max_cameras_per_tick:
                    logger.info(
                        "perception.skip reason=camera_throttled total=%d dispatched=%d max_per_tick=%d",
                        total,
                        dispatched,
                        self.max_cameras_per_tick,
                    )

            time.sleep(0.1)

    def _submit_camera(self, state: CameraPerceptionState) -> bool:
        pool = self._process_pool
        if pool is None:
            self._process_camera_wrapper(state)
            return True
        try:
            pool.submit(self._process_camera_wrapper, state)
            return True
        except RuntimeError:
            logger.exception(
                "perception.dispatch_failed room_id=%s camera_id=%s",
                state.room_id,
                state.camera_id,
            )
            return False

    def _process_camera_wrapper(self, state: CameraPerceptionState) -> None:
        try:
            self._process_camera(state)
        finally:
            with state.lock:
                state.in_flight = False

    def _submit_yolo_if_due(
        self,
        state: CameraPerceptionState,
        frame: "cv2.typing.MatLike",
    ) -> None:
        detector = self.yolo_detector
        if detector is None or not detector.ready():
            return
        now = time.monotonic()
        with state.lock:
            if state.yolo_in_flight:
                return
            if now - state.last_yolo_submit_at < self._yolo_submit_interval_seconds:
                return
            try:
                frame_copy = frame.copy()
            except Exception:
                return
            state.yolo_in_flight = True
            state.last_yolo_submit_at = now
        pool = self._yolo_pool
        if pool is None:
            self._run_yolo_task(state, frame_copy)
            return
        try:
            pool.submit(self._run_yolo_task, state, frame_copy)
        except RuntimeError:
            with state.lock:
                state.yolo_in_flight = False

    def _run_yolo_task(
        self,
        state: CameraPerceptionState,
        frame: "cv2.typing.MatLike",
    ) -> None:
        detector = self.yolo_detector
        if detector is None or not detector.ready():
            with state.lock:
                state.yolo_in_flight = False
            return
        start = time.perf_counter()
        detections: List[YoloDetection] = []
        try:
            detections = detector.detect(frame)
        except Exception:
            detections = []
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        with state.lock:
            state.yolo_cached = detections
            state.yolo_cached_at = time.monotonic()
            state.last_yolo_inference_ms = elapsed_ms
            state.yolo_in_flight = False

    def _get_cached_yolo(
        self,
        state: CameraPerceptionState,
    ) -> Tuple[List[YoloDetection], Optional[float], float]:
        now = time.monotonic()
        with state.lock:
            if not state.yolo_cached:
                return [], None, state.last_yolo_inference_ms
            age = max(0.0, now - state.yolo_cached_at)
            if age > self._yolo_cache_ttl_seconds:
                return [], age, state.last_yolo_inference_ms
            return list(state.yolo_cached), age, state.last_yolo_inference_ms

    def _process_camera(self, state: CameraPerceptionState) -> None:
        start = time.time()
        success = False
        error: Optional[str] = None
        with state.lock:
            state.last_attempt_at = start
        try:
            frame, ts, arrival_ts = self.stream_manager.get_snapshot_meta(
                state.room_id, state.camera_id
            )
            if frame is None or ts is None or arrival_ts is None:
                logger.info(
                    "perception.skip room_id=%s camera_id=%s reason=frame_missing",
                    state.room_id,
                    state.camera_id,
                )
                self._emit_frame_skipped(state, "frame_missing", confidence=0.25)
                error = "frame_missing"
                return
            now = time.time()
            frame_arrival_age_seconds = now - arrival_ts
            if frame_arrival_age_seconds > self.stale_seconds:
                logger.info(
                    "perception.skip room_id=%s camera_id=%s reason=frame_stale arrival_age=%.3f",
                    state.room_id,
                    state.camera_id,
                    frame_arrival_age_seconds,
                )
                self._emit_frame_skipped(
                    state,
                    "frame_stale",
                    confidence=0.3,
                    extra={
                        "arrival_age_seconds": frame_arrival_age_seconds,
                        "stale_seconds": self.stale_seconds,
                    },
                )
                error = "frame_stale"
                return
            if (
                self.event_max_frame_age_seconds > 0.0
                and frame_arrival_age_seconds > self.event_max_frame_age_seconds
            ):
                logger.info(
                    "perception.skip room_id=%s camera_id=%s reason=frame_too_old arrival_age=%.3f max_age=%.3f",
                    state.room_id,
                    state.camera_id,
                    frame_arrival_age_seconds,
                    self.event_max_frame_age_seconds,
                )
                self._emit_frame_skipped(
                    state,
                    "frame_too_old",
                    confidence=0.3,
                    extra={
                        "arrival_age_seconds": frame_arrival_age_seconds,
                        "max_frame_age_seconds": self.event_max_frame_age_seconds,
                    },
                )
                error = "frame_too_old"
                return
            source_ts = ts
            event_ts, stabilizer_skew = self._stable_event_timestamp(
                state,
                source_ts,
                arrival_ts,
            )
            frame_age_seconds = now - event_ts
            frame_transport_delay_seconds = arrival_ts - source_ts
            effective_offset_seconds = event_ts - source_ts
            with state.lock:
                state.last_frame_source_timestamp = source_ts
                state.last_frame_timestamp = event_ts
                state.last_frame_timestamp_offset_seconds = effective_offset_seconds
                state.last_frame_timestamp_stabilizer_skew_seconds = stabilizer_skew
                state.last_frame_age_seconds = frame_age_seconds
                state.last_frame_transport_delay_seconds = frame_transport_delay_seconds

            perf_start = time.perf_counter()
            faces: List[FaceMatch] = []
            if self.face_identifier and self.face_identifier.ready():
                should_run_face = False
                face_now = time.monotonic()
                with state.lock:
                    if face_now - state.last_face_infer_at >= self.face_detect_interval_seconds:
                        state.last_face_infer_at = face_now
                        should_run_face = True
                if should_run_face:
                    try:
                        faces = self.face_identifier.detect_and_identify(frame)
                    except Exception:
                        faces = []
            face_ms = (time.perf_counter() - perf_start) * 1000.0
            logger.debug(
                "perception.detect_faces room_id=%s camera_id=%s faces=%d",
                state.room_id,
                state.camera_id,
                len(faces),
            )

            detect_start = time.perf_counter()
            yolo_cache_age_seconds: Optional[float] = None
            yolo_last_inference_ms = 0.0
            yolo_detections: List[YoloDetection] = []
            person_detection_source = "none"
            yolo_ready = self.yolo_detector is not None and self.yolo_detector.ready()
            if self.people_detector_mode == "hog_only":
                detections = self._detect_people_hog(frame)
                person_detection_source = "hog_only"
            elif yolo_ready:
                self._submit_yolo_if_due(state, frame)
                yolo_detections, yolo_cache_age_seconds, yolo_last_inference_ms = (
                    self._get_cached_yolo(state)
                )
                if yolo_detections:
                    detections = self._detect_people(
                        frame,
                        yolo_detections=yolo_detections,
                    )
                    person_detection_source = "yolo_cache"
                else:
                    detections = []
                    person_detection_source = "yolo_pending"
            elif self.people_detector_mode == "yolo_only":
                detections = []
                person_detection_source = "yolo_unavailable"
            else:
                detections = self._detect_people_hog(frame)
                person_detection_source = "hog"
            secondary_people: Optional[List[Detection]] = None
            if self.dual_detect_test:
                if self.people_detector_mode == "yolo_only":
                    secondary_people = []
                elif yolo_ready:
                    secondary_people = self._detect_people_hog(frame)
                else:
                    secondary_people = (
                        self._detect_people_yolo(frame, yolo_detections=yolo_detections)
                        if self.yolo_detector is not None and self.yolo_detector.ready()
                        else []
                    )
            logger.debug(
                "perception.detect_people room_id=%s camera_id=%s count=%d",
                state.room_id,
                state.camera_id,
                len(detections),
            )
            detect_ms = (time.perf_counter() - detect_start) * 1000.0
            if secondary_people is not None:
                logger.info(
                    "perception.dual_detect room_id=%s camera_id=%s primary=%d secondary=%d",
                    state.room_id,
                    state.camera_id,
                    len(detections),
                    len(secondary_people),
                )
            track_start = time.perf_counter()
            with state.lock:
                self._update_tracks(state, frame, detections, faces)
                objects = self._detect_objects(
                    frame,
                    yolo_detections=yolo_detections,
                )
                logger.debug(
                    "perception.detect_objects room_id=%s camera_id=%s count=%d",
                    state.room_id,
                    state.camera_id,
                    len(objects),
                )
                self._update_object_tracks(state, objects, frame)
                self._associate_objects(state, frame)
                self._update_proximity(state)
                self._update_groups(state)
                if not detections and not objects:
                    logger.info(
                        "perception.skip room_id=%s camera_id=%s reason=no_detections",
                        state.room_id,
                        state.camera_id,
                    )
                self._emit(
                    _event(
                        state,
                        "frame_tick",
                        1.0,
                        None,
                        {
                            "frame_timestamp": event_ts,
                            "detections_count": len(detections),
                            "objects_count": len(objects),
                            "face_inference_ms": round(face_ms, 2),
                            "person_detection_ms": round(detect_ms, 2),
                            "tracking_emit_ms": round(
                                (time.perf_counter() - track_start) * 1000.0,
                                2,
                            ),
                            "yolo_cached_count": len(yolo_detections),
                            "yolo_cache_age_seconds": (
                                round(yolo_cache_age_seconds, 3)
                                if yolo_cache_age_seconds is not None
                                else None
                            ),
                            "yolo_last_inference_ms": round(
                                yolo_last_inference_ms,
                                2,
                            ),
                            "person_detection_source": person_detection_source,
                            "people_detector_mode": self.people_detector_mode,
                            "yolo_ready": yolo_ready,
                            "processing_elapsed_ms": round(
                                (time.perf_counter() - perf_start) * 1000.0,
                                2,
                            ),
                            "secondary_detections_count": len(secondary_people)
                            if secondary_people is not None
                            else None,
                        },
                    )
                )
                if self.overlay_store is not None and (detections or objects):
                    self.overlay_store.add_snapshot_all(
                        state.room_id,
                        state.camera_id,
                        frame,
                        timestamp=event_ts,
                    )
                if self.attention_manager is not None:
                    track_summaries = [
                        TrackSummary(
                            bbox=track.bbox,
                            global_id=track.global_id,
                            person_id=self._person_id_for_track(track),
                            role=track.role,
                        )
                        for track in state.tracks.values()
                    ]
                    self.attention_manager.submit(
                        state.room_id,
                        state.camera_id,
                        state.role,
                        frame,
                        event_ts,
                        track_summaries,
                    )
            success = True
        except Exception as exc:
            error = f"exception:{exc}"
            logger.exception(
                "perception.process_failed room_id=%s camera_id=%s",
                state.room_id,
                state.camera_id,
            )
            self._emit_frame_skipped(
                state,
                "process_exception",
                confidence=0.2,
                extra={"error": str(exc)},
            )
        finally:
            self._update_processing_stats(state, start, success, error)

    def _stable_event_timestamp(
        self,
        state: CameraPerceptionState,
        source_ts: float,
        arrival_ts: float,
    ) -> Tuple[float, float]:
        event_ts = source_ts + self.event_timestamp_offset_seconds
        skew = 0.0
        if (
            self.event_timestamp_stabilize_alpha > 0.0
            and math.isfinite(source_ts)
            and math.isfinite(arrival_ts)
        ):
            observed_delay = arrival_ts - source_ts
            delay_ema = state.timestamp_delay_ema_seconds
            if delay_ema is None or not math.isfinite(delay_ema):
                delay_ema = observed_delay
            else:
                alpha = self.event_timestamp_stabilize_alpha
                delay_ema = (1.0 - alpha) * delay_ema + alpha * observed_delay
            state.timestamp_delay_ema_seconds = delay_ema
            skew = observed_delay - delay_ema
            max_correction = self.event_timestamp_stabilize_max_correction_seconds
            if max_correction > 0.0:
                if skew > max_correction:
                    skew = max_correction
                elif skew < -max_correction:
                    skew = -max_correction
            event_ts += skew
        if self.event_timestamp_round_seconds > 0.0:
            step = self.event_timestamp_round_seconds
            event_ts = round(event_ts / step) * step
        return event_ts, skew

    @staticmethod
    def _update_processing_stats(
        state: CameraPerceptionState,
        start: float,
        success: bool,
        error: Optional[str],
    ) -> None:
        end = time.time()
        with state.lock:
            state.last_processing_ms = (end - start) * 1000.0
            if success:
                state.last_processed_at = end
                state.last_error = None
            else:
                state.last_error = error
            processing_ms = state.last_processing_ms
        if processing_ms >= 1500.0:
            logger.warning(
                "perception.slow_camera room_id=%s camera_id=%s processing_ms=%.1f success=%s error=%s",
                state.room_id,
                state.camera_id,
                processing_ms,
                success,
                error,
            )

    def _detect_people(
        self,
        frame: "cv2.typing.MatLike",
        yolo_detections: Optional[List[YoloDetection]] = None,
    ) -> List[Detection]:
        detector = self.yolo_detector
        if detector is not None and detector.ready():
            return self._detect_people_yolo(frame, yolo_detections=yolo_detections)
        return self._detect_people_hog(frame)

    def _detect_people_yolo(
        self,
        frame: "cv2.typing.MatLike",
        yolo_detections: Optional[List[YoloDetection]] = None,
    ) -> List[Detection]:
        detections: List[Detection] = []
        raw_detections = (
            yolo_detections if yolo_detections is not None else self.yolo_detector.detect(frame)
        )
        for det in raw_detections:
            if det.label:  # defensive
                if det.label == "person":
                    detections.append(Detection(det.bbox, det.confidence))
        return detections

    def _detect_people_hog(self, frame: "cv2.typing.MatLike") -> List[Detection]:
        h, w = frame.shape[:2]
        scale_x = w / float(self.detection_width)
        scale_y = h / float(self.detection_height)
        resized = cv2.resize(frame, (self.detection_width, self.detection_height))
        rects, weights = self._hog.detectMultiScale(
            resized, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        detections: List[Detection] = []
        for (x, y, rw, rh), weight in zip(rects, weights):
            if rh <= 0 or rw <= 0:
                continue
            x1 = int(x * scale_x)
            y1 = int(y * scale_y)
            x2 = int((x + rw) * scale_x)
            y2 = int((y + rh) * scale_y)
            conf = float(weight) if weight is not None else 0.5
            detections.append(Detection((x1, y1, x2, y2), min(1.0, conf)))
        return detections

    def _update_tracks(
        self,
        state: CameraPerceptionState,
        frame: "cv2.typing.MatLike",
        detections: List[Detection],
        faces: List[FaceMatch],
    ) -> None:
        now = time.time()
        tracks = state.tracks
        matches = _match_detections_to_tracks(detections, list(tracks.values()), self.person_iou_threshold)

        matched_track_ids = set()
        matched_detection_ids = set()
        detection_track_ids: Dict[int, int] = {}

        for det_idx, track_id in matches.items():
            detection = detections[det_idx]
            track = tracks[track_id]
            track.bbox = detection.bbox
            track.last_seen = now
            track.hits += 1
            centroid = _bbox_center(detection.bbox)
            track.history.append(centroid)
            track.appearance = _appearance_hist(frame, detection.bbox)
            if track.global_id is None:
                track.global_id = self._resolver.assign(state.camera_id, track.appearance)
            else:
                self._resolver.refresh(track.global_id, track.appearance)
            self._update_identity(state, track, faces)
            self._update_role(state, track, frame)
            self._emit(
                _event(
                    state,
                    "person_tracked",
                    detection.confidence,
                    track.global_id,
                    {
                        "track_id": track.track_id,
                        "bbox": detection.bbox,
                        "person_id": self._person_id_for_track(track),
                        "person_name": track.identity_name,
                        "person_role": self._role_for_track(track),
                    },
                )
            , frame=frame)
            self._update_body_movement(state, track, detection, frame)
            self._update_orientation(state, track)
            self._update_posture_state(state, track, frame)
            matched_track_ids.add(track_id)
            matched_detection_ids.add(det_idx)
            detection_track_ids[det_idx] = track_id

        for idx, detection in enumerate(detections):
            if idx in matched_detection_ids:
                continue
            track_id = state.next_track_id
            state.next_track_id += 1
            track = TrackState(
                track_id=track_id,
                bbox=detection.bbox,
                last_seen=now,
                hits=1,
            )
            track.history.append(_bbox_center(detection.bbox))
            track.appearance = _appearance_hist(frame, detection.bbox)
            track.global_id = self._resolver.assign(state.camera_id, track.appearance)
            self._update_identity(state, track, faces)
            tracks[track_id] = track
            self._update_role(state, track, frame)
            self._emit(
                _event(
                    state,
                    "person_detected",
                    detection.confidence,
                    track.global_id,
                    {
                        "track_id": track.track_id,
                        "bbox": detection.bbox,
                        "person_id": self._person_id_for_track(track),
                        "person_name": track.identity_name,
                        "person_role": self._role_for_track(track),
                    },
                )
            , frame=frame)
            self._update_orientation(state, track)
            self._update_posture_state(state, track, frame)

        expired = [
            track_id
            for track_id, track in tracks.items()
            if now - track.last_seen > self.track_ttl_seconds
        ]
        for track_id in expired:
            track = tracks.pop(track_id)
            self._emit(
                _event(
                    state,
                    "person_lost",
                    0.6,
                    track.global_id,
                    {
                        "track_id": track.track_id,
                        "person_id": self._person_id_for_track(track),
                        "person_name": track.identity_name,
                        "person_role": self._role_for_track(track),
                    },
                )
            )

    def _update_role(self, state: CameraPerceptionState, track: TrackState, frame: "cv2.typing.MatLike") -> None:
        student_top_ratio, student_bottom_ratio = self._student_uniform_ratios(
            frame,
            track.bbox,
        )
        x1, y1, x2, y2 = track.bbox
        frame_height = (
            max(1, int(frame.shape[0]))
            if hasattr(frame, "shape") and len(frame.shape) >= 2
            else max(1, self.detection_height)
        )
        frame_width = (
            max(1, int(frame.shape[1]))
            if hasattr(frame, "shape") and len(frame.shape) >= 2
            else max(1, self.detection_width)
        )
        bbox_height = max(1, y2 - y1)
        bbox_width = max(1, x2 - x1)
        height_ratio = bbox_height / float(frame_height)
        bbox_aspect_ratio = bbox_height / float(bbox_width)
        frame_diag = max(1.0, math.hypot(frame_width, frame_height))
        motion_strength = self._track_motion_strength(track, frame_diag)

        height_ratios: List[float] = []
        for candidate in state.tracks.values():
            _cx1, cy1, _cx2, cy2 = candidate.bbox
            candidate_height = max(1, cy2 - cy1)
            height_ratios.append(candidate_height / float(frame_height))
        if height_ratios:
            ordered = sorted(height_ratios)
            mid = len(ordered) // 2
            if len(ordered) % 2 == 1:
                median_height_ratio = ordered[mid]
            else:
                median_height_ratio = (ordered[mid - 1] + ordered[mid]) / 2.0
        else:
            median_height_ratio = height_ratio
        relative_tallness = height_ratio / max(0.01, median_height_ratio)

        has_student_top = student_top_ratio >= self.student_top_min_ratio
        has_student_bottom = student_bottom_ratio >= self.student_bottom_min_ratio
        student_full_uniform = has_student_top and has_student_bottom

        top_only_candidate = (
            student_top_ratio >= self.student_top_only_min_ratio
            and student_bottom_ratio <= self.student_top_only_max_bottom_ratio
        )
        seated_top_only_student = (
            top_only_candidate
            and height_ratio <= self.student_seated_max_height_ratio
        )

        student_score = 0.0
        student_reason = ""
        if student_full_uniform:
            top_strength = min(
                1.0, student_top_ratio / max(0.01, self.student_top_min_ratio)
            )
            bottom_strength = min(
                1.0, student_bottom_ratio / max(0.01, self.student_bottom_min_ratio)
            )
            student_score = min(
                0.95, 0.56 + (0.2 * top_strength) + (0.2 * bottom_strength)
            )
            student_reason = "uniform_top_bottom"
        elif seated_top_only_student:
            top_only_strength = min(
                1.0, student_top_ratio / max(0.01, self.student_top_only_min_ratio)
            )
            seated_strength = min(
                1.0,
                max(
                    0.0,
                    (self.student_seated_max_height_ratio - height_ratio)
                    / max(0.01, self.student_seated_max_height_ratio),
                ),
            )
            student_score = min(
                0.88,
                0.5 + (0.24 * top_only_strength) + (0.1 * seated_strength),
            )
            student_reason = "uniform_top_only_seated"
        if student_score > 0.0:
            stillness_strength = max(0.0, 1.0 - motion_strength)
            if seated_top_only_student:
                student_score = min(0.95, student_score + (0.08 * stillness_strength))
            elif student_full_uniform and height_ratio <= self.student_seated_max_height_ratio * 1.12:
                student_score = min(0.95, student_score + (0.05 * stillness_strength))

        top_deficit = max(
            0.0,
            (self.student_top_min_ratio - student_top_ratio)
            / max(0.01, self.student_top_min_ratio),
        )
        bottom_deficit = max(
            0.0,
            (self.student_bottom_min_ratio - student_bottom_ratio)
            / max(0.01, self.student_bottom_min_ratio),
        )
        non_uniform_strength = min(1.0, (0.35 * top_deficit) + (0.65 * bottom_deficit))
        white_top_without_red = has_student_top and not has_student_bottom
        standing_candidate = (
            height_ratio >= self.teacher_height_ratio
            or (
                track.hits >= self.teacher_min_hits
                and relative_tallness >= 1.28
                and height_ratio >= max(0.12, self.student_seated_max_height_ratio * 0.55)
            )
        )

        teacher_candidate = (
            track.hits >= self.teacher_min_hits
            and not seated_top_only_student
            and (
                (white_top_without_red and standing_candidate)
                or non_uniform_strength >= 0.25
                or (standing_candidate and relative_tallness >= 1.25)
            )
        )
        teacher_score = 0.0
        teacher_reason = "non_uniform_profile"
        if teacher_candidate:
            absolute_height_strength = min(
                1.0, height_ratio / max(0.01, self.teacher_height_ratio)
            )
            relative_height_strength = min(
                1.0, max(0.0, (relative_tallness - 1.0) / 0.6)
            )
            height_strength = max(absolute_height_strength, relative_height_strength)
            mobility_bonus = (
                0.12 * motion_strength if standing_candidate else 0.06 * motion_strength
            )
            if white_top_without_red and standing_candidate:
                teacher_reason = "white_top_standing"
                teacher_score = min(
                    0.94,
                    0.56
                    + (0.14 * height_strength)
                    + (0.16 * min(1.0, bottom_deficit))
                    + mobility_bonus,
                )
            else:
                teacher_score = min(
                    0.93,
                    0.5
                    + (0.28 * non_uniform_strength)
                    + (0.1 * height_strength)
                    + mobility_bonus,
                )
            if relative_tallness >= 1.35:
                if teacher_reason == "non_uniform_profile":
                    teacher_reason = "tall_non_uniform_profile"
                teacher_score = min(
                    0.95,
                    teacher_score
                    + (
                        0.04
                        * min(1.0, max(0.0, (relative_tallness - 1.35) / 0.35))
                    ),
                )

        very_tall_teacher_candidate = (
            track.hits >= self.teacher_min_hits
            and not seated_top_only_student
            and relative_tallness >= 1.45
            and height_ratio >= max(0.15, self.student_seated_max_height_ratio * 0.65)
        )
        if very_tall_teacher_candidate:
            tallness_boost = min(
                1.0,
                max(0.0, (relative_tallness - 1.45) / 1.0),
            )
            motion_boost = min(1.0, motion_strength / 0.25)
            teacher_score = min(
                0.97,
                max(teacher_score, 0.64 + (0.2 * tallness_boost) + (0.12 * motion_boost)),
            )
            if teacher_reason in ("non_uniform_profile", "tall_non_uniform_profile"):
                teacher_reason = "very_tall_active_candidate"
            if student_full_uniform and motion_strength >= 0.08:
                student_score = max(0.0, student_score - (0.12 + (0.06 * tallness_boost)))

        identity_role = (
            track.identity_role.strip().lower()
            if isinstance(track.identity_role, str)
            else ""
        )
        identity_bias = 0.0
        if identity_role in ("teacher", "student"):
            identity_strength = min(1.0, max(0.0, track.identity_score))
            identity_bias = 0.72 + (0.24 * identity_strength)
            if identity_role == "teacher":
                teacher_score = max(teacher_score, identity_bias)
                if teacher_score >= student_score:
                    teacher_reason = "identity_teacher"
            else:
                student_score = max(student_score, identity_bias)
                if student_score >= teacher_score:
                    student_reason = "identity_student"

        evidence_decay = 0.84
        track.role_student_evidence = min(
            6.0, (track.role_student_evidence * evidence_decay) + student_score
        )
        track.role_teacher_evidence = min(
            6.0, (track.role_teacher_evidence * evidence_decay) + teacher_score
        )
        evidence_total = track.role_student_evidence + track.role_teacher_evidence
        if evidence_total > 0.0:
            student_temporal = track.role_student_evidence / evidence_total
            teacher_temporal = track.role_teacher_evidence / evidence_total
        else:
            student_temporal = 0.0
            teacher_temporal = 0.0

        raw_student_score = student_score
        raw_teacher_score = teacher_score
        student_score = (0.64 * student_score) + (0.36 * student_temporal)
        teacher_score = (0.64 * teacher_score) + (0.36 * teacher_temporal)

        margin = self.role_decision_margin
        role_metrics = {
            "student_top_ratio": round(student_top_ratio, 3),
            "student_bottom_ratio": round(student_bottom_ratio, 3),
            "student_top_deficit": round(top_deficit, 3),
            "student_bottom_deficit": round(bottom_deficit, 3),
            "height_ratio": round(height_ratio, 3),
            "bbox_aspect_ratio": round(bbox_aspect_ratio, 3),
            "relative_tallness": round(relative_tallness, 3),
            "median_height_ratio": round(median_height_ratio, 3),
            "motion_strength": round(motion_strength, 3),
            "student_full_uniform": student_full_uniform,
            "seated_top_only_student": seated_top_only_student,
            "teacher_standing_candidate": standing_candidate,
            "identity_role": identity_role if identity_role else None,
            "identity_bias": round(identity_bias, 3),
            "raw_student_score": round(raw_student_score, 3),
            "raw_teacher_score": round(raw_teacher_score, 3),
            "student_temporal": round(student_temporal, 3),
            "teacher_temporal": round(teacher_temporal, 3),
            "student_evidence": round(track.role_student_evidence, 3),
            "teacher_evidence": round(track.role_teacher_evidence, 3),
        }

        if student_score > 0.0 and student_score >= teacher_score + margin:
            if track.role != "student":
                switch_guard = track.role_confidence + (margin * 0.5)
                if (
                    student_score < switch_guard
                    and track.role_student_evidence
                    <= track.role_teacher_evidence * 1.08
                ):
                    return
            if not student_reason:
                if track.role_student_evidence > track.role_teacher_evidence * 1.2:
                    student_reason = "temporal_student_consensus"
                else:
                    student_reason = "student_profile"
            self._emit_role_assignment(
                state=state,
                track=track,
                role="student",
                confidence=student_score,
                frame=frame,
                reason=student_reason,
                metrics=role_metrics,
            )
            return

        if teacher_score > 0.0 and teacher_score >= student_score + margin:
            if track.role != "teacher":
                switch_guard = track.role_confidence + (margin * 0.5)
                if (
                    teacher_score < switch_guard
                    and track.role_teacher_evidence
                    <= track.role_student_evidence * 1.08
                ):
                    return
            if (
                teacher_reason == "non_uniform_profile"
                and track.role_teacher_evidence > track.role_student_evidence * 1.2
            ):
                teacher_reason = "temporal_teacher_consensus"
            self._emit_role_assignment(
                state=state,
                track=track,
                role="teacher",
                confidence=teacher_score,
                frame=frame,
                reason=teacher_reason,
                metrics=role_metrics,
            )

    def _emit_role_assignment(
        self,
        state: CameraPerceptionState,
        track: TrackState,
        role: str,
        confidence: float,
        frame: "cv2.typing.MatLike",
        reason: str,
        metrics: Optional[Dict[str, object]] = None,
    ) -> None:
        confidence = float(max(0.0, min(1.0, confidence)))
        if role == track.role and confidence <= track.role_confidence:
            return
        track.role = role
        track.role_confidence = confidence
        payload: Dict[str, object] = {
            "track_id": track.track_id,
            "role": track.role,
            "role_reason": reason,
            "person_id": self._person_id_for_track(track),
            "person_name": track.identity_name,
            "person_role": track.role,
        }
        if metrics:
            payload.update(metrics)
        self._emit(
            _event(
                state,
                "role_assigned",
                track.role_confidence,
                track.global_id,
                payload,
            ),
            frame=frame,
        )

    def _student_uniform_ratios(
        self,
        frame: "cv2.typing.MatLike",
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[float, float]:
        top_bbox = _bbox_vertical_slice(bbox, 0.0, 0.58)
        bottom_bbox = _bbox_vertical_slice(bbox, 0.52, 1.0)
        top_ratio = _uniform_ratio(
            frame,
            top_bbox,
            self.student_top_hsv_low,
            self.student_top_hsv_high,
        )
        bottom_ratio_1 = _uniform_ratio(
            frame,
            bottom_bbox,
            self.student_bottom_hsv_low,
            self.student_bottom_hsv_high,
        )
        bottom_ratio_2 = _uniform_ratio(
            frame,
            bottom_bbox,
            self.student_bottom_hsv_low_2,
            self.student_bottom_hsv_high_2,
        )
        return top_ratio, max(bottom_ratio_1, bottom_ratio_2)

    @staticmethod
    def _track_motion_strength(track: TrackState, frame_diag: float) -> float:
        if frame_diag <= 0.0 or len(track.history) < 2:
            return 0.0
        points = list(track.history)
        distance = 0.0
        for idx in range(1, len(points)):
            px, py = points[idx - 1]
            cx, cy = points[idx]
            distance += math.hypot(cx - px, cy - py)
        return min(1.0, distance / max(1.0, frame_diag * 0.35))

    def _update_identity(
        self, state: CameraPerceptionState, track: TrackState, faces: List[FaceMatch]
    ) -> None:
        if not faces:
            return
        now = time.time()
        if now - track.last_identity_time < self.identity_min_interval_seconds:
            return
        if track.identity_id is not None and track.identity_score >= self.identity_sticky_score:
            return
        match = _match_face_to_track(faces, track.bbox)
        if match is None or match.person_id is None or match.name is None:
            return
        unknown_id = self._unknown_person_id(track)
        track.identity_id = match.person_id
        track.identity_name = match.name
        track.identity_role = match.role
        track.identity_score = match.score
        track.last_identity_time = now
        if unknown_id and unknown_id != match.person_id:
            with self._lock:
                mapped = self._person_id_map.get(unknown_id)
                if mapped != match.person_id:
                    self._person_id_map[unknown_id] = match.person_id
                    self._emit(
                        _event(
                            state,
                            "identity_resolved",
                            max(0.5, match.score),
                            track.global_id,
                            {
                                "person_id": match.person_id,
                                "previous_person_id": unknown_id,
                                "person_name": match.name,
                                "person_role": self._role_for_track(track),
                                "identity_role": match.role,
                            },
                        )
                    )
        if self.attendance:
            self.attendance.mark_present(
                person_id=match.person_id,
                name=match.name,
                role=match.role or "unknown",
                camera_id=state.camera_id,
                timestamp=now,
            )

    def _update_orientation(self, state: CameraPerceptionState, track: TrackState) -> None:
        if len(track.history) < 2:
            return
        (x1, y1), (x2, y2) = track.history[-2], track.history[-1]
        dx = x2 - x1
        dy = y2 - y1
        orientation = "forward"
        if abs(dx) >= self.orientation_motion_threshold and abs(dx) >= abs(dy):
            orientation = "right" if dx > 0 else "left"
        elif dy >= self.orientation_motion_threshold:
            orientation = "down"
        if orientation != track.last_orientation:
            track.last_orientation = orientation
            self._emit(
                _event(
                    state,
                    "head_orientation_changed",
                    0.5,
                    track.global_id,
                    {
                        "track_id": track.track_id,
                        "orientation": orientation,
                        "person_id": self._person_id_for_track(track),
                        "person_name": track.identity_name,
                        "person_role": self._role_for_track(track),
                    },
                )
            )

    def _update_body_movement(
        self,
        state: CameraPerceptionState,
        track: TrackState,
        detection: Detection,
        frame: "cv2.typing.MatLike",
    ) -> None:
        if not self.body_movement_enabled:
            return
        if len(track.history) < 2:
            return
        now = time.time()
        if (
            self.body_movement_emit_interval_seconds > 0.0
            and now - track.last_body_movement_emit
            < self.body_movement_emit_interval_seconds
        ):
            return
        (x1, y1), (x2, y2) = track.history[-2], track.history[-1]
        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        if distance < self.body_movement_min_delta_pixels:
            return
        track.last_body_movement_emit = now
        confidence = min(1.0, 0.3 + min(0.7, distance / 40.0))
        self._emit(
            _event(
                state,
                "body_movement",
                confidence,
                track.global_id,
                {
                    "track_id": track.track_id,
                    "bbox": detection.bbox,
                    "dx_pixels": dx,
                    "dy_pixels": dy,
                    "distance_pixels": distance,
                    "person_id": self._person_id_for_track(track),
                    "person_name": track.identity_name,
                    "person_role": self._role_for_track(track),
                },
            ),
            frame=frame,
        )

    def _update_posture_state(
        self,
        state: CameraPerceptionState,
        track: TrackState,
        frame: "cv2.typing.MatLike",
    ) -> None:
        x1, y1, x2, y2 = track.bbox
        height = max(1, y2 - y1)
        width = max(1, x2 - x1)
        now = time.time()
        orientation = track.last_orientation or "forward"
        role = self._role_for_track(track)

        if orientation != "down":
            if track.upright_height_ema <= 0.0:
                track.upright_height_ema = float(height)
            else:
                alpha = self.posture_height_ema_alpha
                track.upright_height_ema = (
                    (1.0 - alpha) * track.upright_height_ema + alpha * float(height)
                )
            track.down_since = 0.0
            track.bowing_since = 0.0
            if track.posture != "upright":
                track.posture = "upright"
                self._emit(
                    _event(
                        state,
                        "posture_changed",
                        0.55,
                        track.global_id,
                        {
                            "track_id": track.track_id,
                            "posture": "upright",
                            "orientation": orientation,
                            "person_id": self._person_id_for_track(track),
                            "person_name": track.identity_name,
                            "person_role": role,
                            "role": role,
                        },
                    ),
                    frame=frame,
                )
            return

        if track.down_since <= 0.0:
            track.down_since = now
        baseline_height = (
            track.upright_height_ema
            if track.upright_height_ema > 1.0
            else float(height)
        )
        bow_ratio = min(2.0, float(height) / max(1.0, baseline_height))
        aspect_ratio = width / float(height)
        is_bowing = (
            bow_ratio <= self.sleep_bow_ratio_threshold
            and aspect_ratio >= self.sleep_bow_aspect_min
        )
        posture = "bowing" if is_bowing else "upright"

        if posture != track.posture:
            track.posture = posture
            self._emit(
                _event(
                    state,
                    "posture_changed",
                    0.6 if is_bowing else 0.55,
                    track.global_id,
                    {
                        "track_id": track.track_id,
                        "posture": posture,
                        "orientation": orientation,
                        "bow_ratio": bow_ratio,
                            "aspect_ratio": aspect_ratio,
                            "person_id": self._person_id_for_track(track),
                            "person_name": track.identity_name,
                            "person_role": role,
                            "role": role,
                        },
                    ),
                frame=frame,
            )

        if not is_bowing:
            track.bowing_since = 0.0
            return
        if track.bowing_since <= 0.0:
            track.bowing_since = now

        bow_duration = now - track.bowing_since
        down_duration = now - track.down_since
        sleep_duration = min(bow_duration, down_duration)
        if sleep_duration < self.sleep_min_seconds:
            return
        if now - track.last_sleep_emit < self.sleep_emit_interval_seconds:
            return
        track.last_sleep_emit = now
        confidence = min(1.0, 0.5 + min(0.5, sleep_duration / 20.0))
        self._emit(
            _event(
                state,
                "sleeping_suspected",
                confidence,
                track.global_id,
                {
                    "track_id": track.track_id,
                    "posture": "bowing",
                    "orientation": orientation,
                    "bow_ratio": bow_ratio,
                    "aspect_ratio": aspect_ratio,
                    "bowing_duration_seconds": bow_duration,
                    "head_down_duration_seconds": down_duration,
                    "sleep_duration_seconds": sleep_duration,
                    "person_id": self._person_id_for_track(track),
                    "person_name": track.identity_name,
                    "person_role": role,
                    "role": role,
                },
            ),
            frame=frame,
        )

    def _detect_objects(
        self,
        frame: "cv2.typing.MatLike",
        yolo_detections: Optional[List[YoloDetection]] = None,
    ) -> List[ObjectDetection]:
        if self.yolo_detector is not None and self.yolo_detector.ready():
            return self._detect_objects_yolo(frame, yolo_detections=yolo_detections)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections: List[ObjectDetection] = []
        h, w = frame.shape[:2]
        frame_area = float(h * w)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 0.002 * frame_area:
                continue
            x, y, rw, rh = cv2.boundingRect(cnt)
            if rw <= 0 or rh <= 0:
                continue
            aspect = rw / float(rh)
            area_ratio = area / frame_area
            if area_ratio < 0.0008:
                continue

            mean_val = float(np.mean(gray[y : y + rh, x : x + rw]))
            obj = self._classify_object(area_ratio, aspect, mean_val)
            if obj is None:
                continue
            if not self._object_allowed(obj.object_type):
                continue
            obj.bbox = (x, y, x + rw, y + rh)
            detections.append(obj)
        return detections

    def _detect_objects_yolo(
        self,
        frame: "cv2.typing.MatLike",
        yolo_detections: Optional[List[YoloDetection]] = None,
    ) -> List[ObjectDetection]:
        detector = self.yolo_detector
        if detector is None:
            return []
        detections: List[ObjectDetection] = []
        raw_detections = (
            yolo_detections if yolo_detections is not None else detector.detect(frame)
        )
        for det in raw_detections:
            obj = self._map_yolo_label(det)
            if obj is None:
                continue
            if not self._object_allowed(obj.object_type):
                continue
            obj.bbox = det.bbox
            detections.append(obj)
        return detections

    def _map_yolo_label(self, det: YoloDetection) -> Optional[ObjectDetection]:
        label = det.label
        if label == "person":
            return None
        mapped = self.object_label_map.get(label)
        if isinstance(mapped, dict):
            object_type = mapped.get("object_type", label)
            category = mapped.get("category", "other")
            risk_level = mapped.get("risk_level", "low")
            if isinstance(object_type, str) and object_type.strip():
                object_type_value = object_type.strip().lower()
                return ObjectDetection(
                    object_type=object_type_value,
                    category=category if isinstance(category, str) else "other",
                    risk_level=risk_level if isinstance(risk_level, str) else "low",
                    confidence=det.confidence,
                    bbox=det.bbox,
                )
        return _map_yolo_label(det)

    def _classify_object(
        self, area_ratio: float, aspect: float, brightness: float
    ) -> Optional[ObjectDetection]:
        if aspect >= 6.0 and area_ratio < 0.01:
            return ObjectDetection(
                object_type="knife_like",
                category="suspicious",
                risk_level="high",
                confidence=0.25,
                bbox=(0, 0, 0, 0),
            )
        if area_ratio < 0.01 and 0.3 <= aspect <= 3.0:
            return ObjectDetection(
                object_type="phone",
                category="devices",
                risk_level="low",
                confidence=0.3,
                bbox=(0, 0, 0, 0),
            )
        if area_ratio < 0.02 and aspect <= 0.25:
            return ObjectDetection(
                object_type="test_tube",
                category="lab",
                risk_level="medium",
                confidence=0.25,
                bbox=(0, 0, 0, 0),
            )
        if 0.01 <= area_ratio < 0.05 and 0.6 <= aspect <= 1.8:
            if brightness > 170:
                return ObjectDetection(
                    object_type="paper",
                    category="academic",
                    risk_level="low",
                    confidence=0.35,
                    bbox=(0, 0, 0, 0),
                )
            return ObjectDetection(
                object_type="notebook",
                category="academic",
                risk_level="low",
                confidence=0.3,
                bbox=(0, 0, 0, 0),
            )
        if 0.02 <= area_ratio < 0.06 and 0.7 <= aspect <= 1.3 and brightness < 140:
            return ObjectDetection(
                object_type="beaker",
                category="lab",
                risk_level="medium",
                confidence=0.25,
                bbox=(0, 0, 0, 0),
            )
        if area_ratio >= 0.05 and aspect >= 1.2:
            return ObjectDetection(
                object_type="laptop",
                category="devices",
                risk_level="low",
                confidence=0.4,
                bbox=(0, 0, 0, 0),
            )
        if area_ratio >= 0.05 and 0.6 <= aspect <= 1.2:
            return ObjectDetection(
                object_type="tablet",
                category="devices",
                risk_level="low",
                confidence=0.35,
                bbox=(0, 0, 0, 0),
            )
        if area_ratio >= 0.08 and aspect < 2.5:
            return ObjectDetection(
                object_type="backpack",
                category="personal",
                risk_level="low",
                confidence=0.3,
                bbox=(0, 0, 0, 0),
            )
        if 0.01 <= area_ratio < 0.04 and aspect >= 1.5:
            return ObjectDetection(
                object_type="pouch",
                category="personal",
                risk_level="low",
                confidence=0.25,
                bbox=(0, 0, 0, 0),
            )
        return None

    def _update_object_tracks(
        self,
        state: CameraPerceptionState,
        detections: List[ObjectDetection],
        frame: "cv2.typing.MatLike",
    ) -> None:
        now = time.time()
        tracks = state.object_tracks
        matches = _match_object_detections(detections, list(tracks.values()), self.object_iou_threshold)

        matched_detection_ids = set()

        for det_idx, track_id in matches.items():
            detection = detections[det_idx]
            track = tracks[track_id]
            track.detection = detection
            track.last_seen = now
            track.hits += 1
            matched_detection_ids.add(det_idx)

        for idx, detection in enumerate(detections):
            if idx in matched_detection_ids:
                continue
            track_id = state.next_object_id
            state.next_object_id += 1
            track = ObjectTrack(
                track_id=track_id,
                detection=detection,
                last_seen=now,
                hits=1,
            )
            tracks[track_id] = track

        expired = [
            track_id
            for track_id, track in tracks.items()
            if now - track.last_seen > self.object_ttl_seconds
        ]
        for track_id in expired:
            tracks.pop(track_id, None)

        for track in tracks.values():
            if not track.emitted and track.hits < self.object_persist_frames:
                continue
            detection = self._apply_object_flags(track.detection)
            object_type = _object_key(detection.object_type)
            track.emitted = True
            self._emit(
                _event(
                    state,
                    "object_detected",
                    detection.confidence,
                    None,
                    {
                        "object_type": detection.object_type,
                        "category": detection.category,
                        "risk_level": detection.risk_level,
                        "bbox": detection.bbox,
                        "object_track_id": track.track_id,
                        "priority": object_type in self.object_priority,
                        "risky": object_type in self.object_risky,
                    },
                )
            , frame=frame)

    def _associate_objects(
        self,
        state: CameraPerceptionState,
        frame: "cv2.typing.MatLike",
    ) -> None:
        if not state.tracks or not state.object_tracks:
            return
        for obj_track in state.object_tracks.values():
            best_track = None
            best_score = 0.0
            for track in state.tracks.values():
                iou = _bbox_iou(track.bbox, obj_track.detection.bbox)
                if iou > best_score:
                    best_score = iou
                    best_track = track
            if best_track is None:
                continue
            detection = self._apply_object_flags(obj_track.detection)
            object_type = _object_key(detection.object_type)
            self._emit(
                _event(
                    state,
                    "object_associated",
                    min(0.9, detection.confidence + 0.2),
                    best_track.global_id,
                    {
                        "track_id": best_track.track_id,
                        "object_track_id": obj_track.track_id,
                        "object_type": detection.object_type,
                        "category": detection.category,
                        "risk_level": detection.risk_level,
                        "bbox": detection.bbox,
                        "priority": object_type in self.object_priority,
                        "risky": object_type in self.object_risky,
                        "person_id": self._person_id_for_track(best_track),
                        "person_name": best_track.identity_name,
                        "person_role": self._role_for_track(best_track),
                    },
                )
            , frame=frame)
            self._maybe_emit_device_usage(state, best_track, detection, frame)

        if not self.exam_mode:
            return
        for obj_track in state.object_tracks.values():
            if obj_track.detection.object_type != "paper":
                continue
            x1, y1, x2, y2 = obj_track.detection.bbox
            area = max(1, (x2 - x1) * (y2 - y1))
            if area <= 0:
                continue
            if area < 3000:
                self._emit(
                    _event(
                        state,
                        "object_detected",
                        0.4,
                        None,
                        {
                            "object_type": "concealed_paper",
                            "category": "suspicious",
                            "risk_level": "medium",
                            "bbox": obj_track.detection.bbox,
                            "priority": "concealed_paper" in self.object_priority,
                            "risky": "concealed_paper" in self.object_risky,
                        },
                    )
                , frame=frame)

    @staticmethod
    def _role_for_track(track: TrackState) -> str:
        return track.role

    def _maybe_emit_device_usage(
        self,
        state: CameraPerceptionState,
        track: TrackState,
        detection: ObjectDetection,
        frame: "cv2.typing.MatLike",
    ) -> None:
        if detection.object_type not in {"phone", "tablet", "laptop", "device"}:
            return
        now = time.time()
        role = self._role_for_track(track)
        person_id = self._person_id_for_track(track)
        if now - track.last_device_emit >= self.device_usage_emit_interval_seconds:
            track.last_device_emit = now
            self._emit(
                _event(
                    state,
                    "device_usage_detected",
                    min(1.0, detection.confidence + 0.1),
                    track.global_id,
                    {
                        "track_id": track.track_id,
                        "object_type": detection.object_type,
                        "category": detection.category,
                        "risk_level": detection.risk_level,
                        "bbox": detection.bbox,
                        "person_id": person_id,
                        "person_name": track.identity_name,
                        "person_role": role,
                        "role": role,
                    },
                ),
                frame=frame,
        )
        if detection.object_type != "phone":
            return
        if now - track.last_phone_emit < self.phone_usage_emit_interval_seconds:
            return
        track.last_phone_emit = now
        if role == "teacher":
            event_type = "teacher_phone_usage"
        elif role == "student":
            event_type = "student_phone_usage"
        else:
            event_type = "phone_usage_detected"
        self._emit(
            _event(
                state,
                event_type,
                min(1.0, detection.confidence + 0.15),
                track.global_id,
                {
                    "track_id": track.track_id,
                    "object_type": detection.object_type,
                    "category": detection.category,
                    "risk_level": detection.risk_level,
                    "bbox": detection.bbox,
                    "person_id": person_id,
                    "person_name": track.identity_name,
                    "person_role": role,
                    "role": role,
                },
            ),
            frame=frame,
        )

    def _update_proximity(self, state: CameraPerceptionState) -> None:
        tracks = list(state.tracks.values())
        if len(tracks) < 2:
            return
        h_ratio = self.proximity_distance_ratio
        for i, t1 in enumerate(tracks):
            for t2 in tracks[i + 1 :]:
                c1 = _bbox_center(t1.bbox)
                c2 = _bbox_center(t2.bbox)
                dist = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                frame_diag = math.hypot(
                    self.detection_width, self.detection_height
                )
                threshold = frame_diag * h_ratio
                if t1.track_id < t2.track_id:
                    key: Tuple[int, int] = (t1.track_id, t2.track_id)
                else:
                    key = (t2.track_id, t1.track_id)
                close, since, emitted = state.proximity_state.get(
                    key, (False, time.time(), False)
                )
                now = time.time()
                if dist <= threshold:
                    if not close:
                        state.proximity_state[key] = (True, now, False)
                    elif not emitted and now - since >= self.proximity_duration_seconds:
                        state.proximity_state[key] = (True, since, True)
                        self._emit(
                            _event(
                                state,
                                "proximity_event",
                                0.5,
                                None,
                                {
                                    "track_ids": [t1.track_id, t2.track_id],
                                    "global_ids": [t1.global_id, t2.global_id],
                                    "person_ids": [
                                        self._person_id_for_track(t1),
                                        self._person_id_for_track(t2),
                                    ],
                                    "distance": dist,
                                    "status": "close",
                                    "duration_seconds": now - since,
                                },
                            )
                        )
                else:
                    if close:
                        duration = now - since
                        state.proximity_state[key] = (False, now, False)
                        self._emit(
                            _event(
                                state,
                                "proximity_event",
                                0.5,
                                None,
                                {
                                    "track_ids": [t1.track_id, t2.track_id],
                                    "global_ids": [t1.global_id, t2.global_id],
                                    "person_ids": [
                                        self._person_id_for_track(t1),
                                        self._person_id_for_track(t2),
                                    ],
                                    "distance": dist,
                                    "status": "separated",
                                    "duration_seconds": duration,
                                },
                            )
                        )

    def _update_groups(self, state: CameraPerceptionState) -> None:
        tracks = list(state.tracks.values())
        if len(tracks) < 2:
            return
        now = time.time()
        groups: List[List[TrackState]] = _cluster_groups(
            tracks,
            self.group_distance_ratio,
            (self.detection_width, self.detection_height),
        )

        active_keys = set()
        for group in groups:
            if len(group) < 3:
                continue
            members_unique = _unique_values(t.global_id for t in group)
            person_ids_unique = _unique_values(
                self._person_id_for_track(t) for t in group
            )
            member_ids: frozenset[int] = frozenset(t.track_id for t in group)
            active_keys.add(member_ids)
            existing = state.group_state.get(member_ids)
            if existing is None:
                group_id = state.next_group_id
                state.next_group_id += 1
                state.group_state[member_ids] = (group_id, now, False)
            else:
                group_id, since, emitted = existing
                duration = now - since
                if duration >= self.group_duration_seconds and not emitted:
                    self._emit(
                        _event(
                            state,
                            "group_formed",
                            0.6,
                            None,
                            {
                                "group_id": group_id,
                                "members": person_ids_unique,
                                "person_ids": person_ids_unique,
                                "member_global_ids": members_unique,
                                "track_ids": [t.track_id for t in group],
                                "duration_seconds": duration,
                            },
                        )
                    )
                    state.group_state[member_ids] = (group_id, since, True)
                elif duration >= self.group_duration_seconds and emitted:
                    self._emit(
                        _event(
                            state,
                            "group_updated",
                            0.6,
                            None,
                            {
                                "group_id": group_id,
                                "members": person_ids_unique,
                                "person_ids": person_ids_unique,
                                "member_global_ids": members_unique,
                                "track_ids": [t.track_id for t in group],
                                "duration_seconds": duration,
                            },
                        )
                    )

        for key in list(state.group_state.keys()):
            if key not in active_keys:
                state.group_state.pop(key, None)

    def _object_allowed(self, object_type: str) -> bool:
        key = _object_key(object_type)
        if not key:
            return False
        if not self.object_allowlist:
            return True
        return key in self.object_allowlist

    def _apply_object_flags(self, detection: ObjectDetection) -> ObjectDetection:
        object_type = _object_key(detection.object_type)
        if object_type in self.object_risky and detection.risk_level != "high":
            detection = ObjectDetection(
                object_type=detection.object_type,
                category=detection.category,
                risk_level="high",
                confidence=detection.confidence,
                bbox=detection.bbox,
            )
        return detection


def _event(
    state: CameraPerceptionState,
    event_type: str,
    confidence: float,
    global_person_id: Optional[int],
    payload: Dict[str, object],
) -> Dict[str, object]:
    person_id = payload.pop("person_id", None)
    if person_id is None and global_person_id is not None:
        person_id = f"unknown:{global_person_id}"
    timestamp = (
        state.last_frame_timestamp
        if state.last_frame_timestamp is not None
        else time.time()
    )
    data: Dict[str, object] = {
        "timestamp": timestamp,
        "room_id": state.room_id,
        "camera_id": state.camera_id,
        "global_person_id": global_person_id,
        "person_id": person_id,
        "event_type": event_type,
        "confidence": float(max(0.0, min(1.0, confidence))),
    }
    if state.last_frame_timestamp is not None:
        data["frame_timestamp"] = state.last_frame_timestamp
    if state.last_frame_source_timestamp is not None:
        data["frame_source_timestamp"] = state.last_frame_source_timestamp
    if state.last_frame_timestamp_offset_seconds != 0.0:
        data["timestamp_offset_seconds"] = state.last_frame_timestamp_offset_seconds
    if state.last_frame_timestamp_stabilizer_skew_seconds != 0.0:
        data["timestamp_stabilizer_skew_seconds"] = (
            state.last_frame_timestamp_stabilizer_skew_seconds
        )
    if state.last_frame_age_seconds is not None:
        data["frame_age_seconds"] = state.last_frame_age_seconds
    if state.last_frame_transport_delay_seconds is not None:
        data["frame_transport_delay_seconds"] = state.last_frame_transport_delay_seconds
    data.update(payload)
    return data


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _unique_values(values) -> List[object]:
    seen = set()
    output: List[object] = []
    for value in values:
        if value is None:
            continue
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _bbox_vertical_slice(
    bbox: Tuple[int, int, int, int],
    start_ratio: float,
    end_ratio: float,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    if y2 <= y1:
        return bbox
    start = max(0.0, min(1.0, start_ratio))
    end = max(start, min(1.0, end_ratio))
    height = y2 - y1
    ys = y1 + int(height * start)
    ye = y1 + int(height * end)
    ys = max(y1, min(y2 - 1, ys))
    ye = max(ys + 1, min(y2, ye))
    return (x1, ys, x2, ye)


def _object_key(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = max(1, (ax2 - ax1) * (ay2 - ay1))
    b_area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(a_area + b_area - inter)


def _match_detections_to_tracks(
    detections: List[Detection],
    tracks: List[TrackState],
    iou_threshold: float,
) -> Dict[int, int]:
    matches: Dict[int, int] = {}
    used_tracks = set()
    for det_idx, det in enumerate(detections):
        best_iou = 0.0
        best_track = None
        for track in tracks:
            if track.track_id in used_tracks:
                continue
            iou = _bbox_iou(det.bbox, track.bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track and best_iou >= iou_threshold:
            matches[det_idx] = best_track.track_id
            used_tracks.add(best_track.track_id)
    return matches


def _match_object_detections(
    detections: List[ObjectDetection],
    tracks: List[ObjectTrack],
    iou_threshold: float,
) -> Dict[int, int]:
    matches: Dict[int, int] = {}
    used_tracks = set()
    for det_idx, det in enumerate(detections):
        best_iou = 0.0
        best_track = None
        det_type = _object_key(det.object_type)
        for track in tracks:
            if track.track_id in used_tracks:
                continue
            if _object_key(track.detection.object_type) != det_type:
                continue
            iou = _bbox_iou(det.bbox, track.detection.bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track and best_iou >= iou_threshold:
            matches[det_idx] = best_track.track_id
            used_tracks.add(best_track.track_id)
    return matches


def _appearance_hist(frame: "cv2.typing.MatLike", bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    score = cv2.compareHist(a.astype(np.float32), b.astype(np.float32), cv2.HISTCMP_CORREL)
    return float((score + 1.0) / 2.0)


def _uniform_ratio(
    frame: "cv2.typing.MatLike",
    bbox: Tuple[int, int, int, int],
    hsv_low: Tuple[int, int, int],
    hsv_high: Tuple[int, int, int],
) -> float:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(1, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(1, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array(hsv_low, dtype=np.uint8)
    upper = np.array(hsv_high, dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    ratio = float(np.mean(mask > 0))
    return ratio


def _cluster_groups(
    tracks: List[TrackState],
    distance_ratio: float,
    frame_size: Tuple[int, int],
) -> List[List[TrackState]]:
    width, height = frame_size
    threshold = math.hypot(width, height) * distance_ratio
    groups: List[List[TrackState]] = []
    for track in tracks:
        placed = False
        c = _bbox_center(track.bbox)
        for group in groups:
            if any(
                math.hypot(c[0] - _bbox_center(t.bbox)[0], c[1] - _bbox_center(t.bbox)[1])
                <= threshold
                for t in group
            ):
                group.append(track)
                placed = True
                break
        if not placed:
            groups.append([track])
    return groups


def _match_face_to_track(
    faces: List[FaceMatch], person_bbox: Tuple[int, int, int, int]
) -> Optional[FaceMatch]:
    best = None
    best_iou = 0.0
    for face in faces:
        iou = _bbox_iou(face.bbox, person_bbox)
        if iou > best_iou:
            best_iou = iou
            best = face
    if best_iou < 0.05:
        return None
    return best


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


def _event_cursor_ts(event: Dict[str, object]) -> float:
    emitted = _get_float(event, "emitted_at")
    event_ts = _get_float(event, "timestamp")
    if emitted > 0.0 and event_ts > 0.0:
        return max(emitted, event_ts)
    if emitted > 0.0:
        return emitted
    return event_ts




def _map_yolo_label(det: YoloDetection) -> Optional[ObjectDetection]:
    label = det.label
    if label == "person":
        return None
    if label == "cell phone":
        return ObjectDetection(
            object_type="phone",
            category="devices",
            risk_level="low",
            confidence=det.confidence,
            bbox=det.bbox,
        )
    if label == "laptop":
        return ObjectDetection(
            object_type="laptop",
            category="devices",
            risk_level="low",
            confidence=det.confidence,
            bbox=det.bbox,
        )
    if label == "book":
        return ObjectDetection(
            object_type="book",
            category="academic",
            risk_level="low",
            confidence=det.confidence,
            bbox=det.bbox,
        )
    if label == "backpack":
        return ObjectDetection(
            object_type="backpack",
            category="personal",
            risk_level="low",
            confidence=det.confidence,
            bbox=det.bbox,
        )
    if label in ("knife", "scissors"):
        return ObjectDetection(
            object_type=label,
            category="suspicious",
            risk_level="high",
            confidence=det.confidence,
            bbox=det.bbox,
        )
    if label in ("handbag", "suitcase"):
        return ObjectDetection(
            object_type="pouch",
            category="personal",
            risk_level="low",
            confidence=det.confidence,
            bbox=det.bbox,
        )
    if label in ("tablet", "tv", "remote", "keyboard", "mouse"):
        return ObjectDetection(
            object_type="device",
            category="devices",
            risk_level="low",
            confidence=det.confidence,
            bbox=det.bbox,
        )
    return None
