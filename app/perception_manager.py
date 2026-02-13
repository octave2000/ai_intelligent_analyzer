import logging
import math
import threading
import time
from collections import deque
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
        self.teacher_height_ratio = teacher_height_ratio
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
        self.pipeline_tag = pipeline_tag
        self.face_identifier = face_identifier
        self.attendance = attendance
        self.yolo_detector = yolo_detector
        self.overlay_store = overlay_store
        self.object_allowlist = set(object_allowlist)
        self.object_priority = set(object_priority)
        self.object_risky = set(object_risky)
        self.object_label_map = object_label_map or {}
        self.attention_manager = attention_manager

        self._cameras: Dict[str, Dict[str, CameraPerceptionState]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
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
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

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
            processed = 0
            checked = 0
            total = len(camera_order)
            idx = start_index
            while checked < total and processed < self.max_cameras_per_tick:
                room_id, camera_id = camera_order[idx]
                idx = (idx + 1) % total
                checked += 1
                with self._lock:
                    state = self._cameras.get(room_id, {}).get(camera_id)
                if state is None:
                    continue
                interval = self.active_interval_seconds
                if now - state.last_run < interval:
                    logger.info(
                        "perception.skip room_id=%s camera_id=%s reason=interval_throttle",
                        room_id,
                        camera_id,
                    )
                    continue
                state.last_run = now
                logger.debug(
                    "perception.process_start room_id=%s camera_id=%s",
                    room_id,
                    camera_id,
                )
                self._process_camera(state)
                processed += 1
            with self._lock:
                if total > 0:
                    self._camera_index = idx
                if total > processed and processed >= self.max_cameras_per_tick:
                    logger.info(
                        "perception.skip reason=camera_throttled total=%d processed=%d max_per_tick=%d",
                        total,
                        processed,
                        self.max_cameras_per_tick,
                    )

            time.sleep(0.1)

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

            faces: List[FaceMatch] = []
            if self.face_identifier and self.face_identifier.ready():
                try:
                    faces = self.face_identifier.detect_and_identify(frame)
                except Exception:
                    faces = []
            logger.debug(
                "perception.detect_faces room_id=%s camera_id=%s faces=%d",
                state.room_id,
                state.camera_id,
                len(faces),
            )

            detections = self._detect_people(frame)
            secondary_people: Optional[List[Detection]] = None
            if self.dual_detect_test:
                if self.yolo_detector is not None and self.yolo_detector.ready():
                    secondary_people = self._detect_people_hog(frame)
                else:
                    secondary_people = (
                        self._detect_people_yolo(frame)
                        if self.yolo_detector is not None
                        else []
                    )
            logger.debug(
                "perception.detect_people room_id=%s camera_id=%s count=%d",
                state.room_id,
                state.camera_id,
                len(detections),
            )
            if secondary_people is not None:
                logger.info(
                    "perception.dual_detect room_id=%s camera_id=%s primary=%d secondary=%d",
                    state.room_id,
                    state.camera_id,
                    len(detections),
                    len(secondary_people),
                )
            with state.lock:
                self._update_tracks(state, frame, detections, faces)
                objects = self._detect_objects(frame)
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

    def _detect_people(self, frame: "cv2.typing.MatLike") -> List[Detection]:
        detector = self.yolo_detector
        if detector is not None and detector.ready():
            return self._detect_people_yolo(frame)
        return self._detect_people_hog(frame)

    def _detect_people_yolo(self, frame: "cv2.typing.MatLike") -> List[Detection]:
        detections: List[Detection] = []
        for det in self.yolo_detector.detect(frame):
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
                        "person_role": track.identity_role,
                    },
                )
            , frame=frame)
            self._update_role(state, track, frame)
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
                        "person_role": track.identity_role,
                    },
                )
            , frame=frame)
            self._update_role(state, track, frame)
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
                        "person_role": track.identity_role,
                    },
                )
            )

    def _update_role(self, state: CameraPerceptionState, track: TrackState, frame: "cv2.typing.MatLike") -> None:
        if track.identity_role in ("teacher", "student"):
            if track.identity_role != track.role or track.identity_score > track.role_confidence:
                track.role = track.identity_role
                track.role_confidence = min(1.0, max(0.7, track.identity_score))
                self._emit(
                    _event(
                        state,
                        "role_assigned",
                        track.role_confidence,
                        track.global_id,
                        {
                            "track_id": track.track_id,
                            "role": track.role,
                            "person_id": self._person_id_for_track(track),
                            "person_name": track.identity_name,
                            "person_role": track.identity_role,
                        },
                    )
                , frame=frame)
            return
        if track.role == "student" and track.role_confidence >= 0.7:
            return
        uniform_ratio = _uniform_ratio(frame, track.bbox, self.uniform_hsv_low, self.uniform_hsv_high)
        if uniform_ratio >= self.uniform_min_ratio:
            track.role = "student"
            track.role_confidence = min(1.0, 0.7 + uniform_ratio)
            self._emit(
                _event(
                    state,
                    "role_assigned",
                    track.role_confidence,
                    track.global_id,
                    {
                        "track_id": track.track_id,
                        "role": track.role,
                        "person_id": self._person_id_for_track(track),
                        "person_name": track.identity_name,
                        "person_role": track.identity_role,
                    },
                )
            , frame=frame)
            return

        height = max(1, track.bbox[3] - track.bbox[1])
        frame_height = frame.shape[0]
        height_ratio = height / float(frame_height)
        if height_ratio >= self.teacher_height_ratio and track.hits >= 5:
            confidence = min(0.6, height_ratio)
            if confidence > track.role_confidence:
                track.role = "teacher"
                track.role_confidence = confidence
                self._emit(
                    _event(
                        state,
                        "role_assigned",
                        track.role_confidence,
                        track.global_id,
                        {
                            "track_id": track.track_id,
                            "role": track.role,
                            "person_id": self._person_id_for_track(track),
                            "person_name": track.identity_name,
                            "person_role": track.identity_role,
                        },
                    )
                , frame=frame)

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
                                "person_role": match.role,
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
                        "person_role": track.identity_role,
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
                    "person_role": track.identity_role,
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
                            "person_role": track.identity_role,
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
                        "person_role": track.identity_role,
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
                    "person_role": track.identity_role,
                    "role": role,
                },
            ),
            frame=frame,
        )

    def _detect_objects(self, frame: "cv2.typing.MatLike") -> List[ObjectDetection]:
        if self.yolo_detector is not None and self.yolo_detector.ready():
            return self._detect_objects_yolo(frame)
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

    def _detect_objects_yolo(self, frame: "cv2.typing.MatLike") -> List[ObjectDetection]:
        detector = self.yolo_detector
        if detector is None:
            return []
        detections: List[ObjectDetection] = []
        for det in detector.detect(frame):
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
                return ObjectDetection(
                    object_type=object_type.strip(),
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

        matched_track_ids = set()
        matched_detection_ids = set()
        detection_track_ids: Dict[int, int] = {}

        for det_idx, track_id in matches.items():
            detection = detections[det_idx]
            track = tracks[track_id]
            track.detection = detection
            track.last_seen = now
            track.hits += 1
            matched_track_ids.add(track_id)
            matched_detection_ids.add(det_idx)
            detection_track_ids[det_idx] = track_id

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
            detection_track_ids[idx] = track_id

        expired = [
            track_id
            for track_id, track in tracks.items()
            if now - track.last_seen > self.object_ttl_seconds
        ]
        for track_id in expired:
            tracks.pop(track_id, None)

        for det_idx, detection in enumerate(detections):
            detection = self._apply_object_flags(detection)
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
                        "object_track_id": detection_track_ids.get(det_idx),
                        "priority": detection.object_type in self.object_priority,
                        "risky": detection.object_type in self.object_risky,
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
                        "priority": detection.object_type in self.object_priority,
                        "risky": detection.object_type in self.object_risky,
                        "person_id": self._person_id_for_track(best_track),
                        "person_name": best_track.identity_name,
                        "person_role": best_track.identity_role,
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
        if isinstance(track.identity_role, str) and track.identity_role:
            return track.identity_role
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
                        "person_role": track.identity_role,
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
                    "person_role": track.identity_role,
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
        return True

    def _apply_object_flags(self, detection: ObjectDetection) -> ObjectDetection:
        if detection.object_type in self.object_risky and detection.risk_level != "high":
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
        for track in tracks:
            if track.track_id in used_tracks:
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
