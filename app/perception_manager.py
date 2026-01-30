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
    tracks: Dict[int, TrackState] = field(default_factory=dict)
    object_tracks: Dict[int, ObjectTrack] = field(default_factory=dict)
    next_track_id: int = 1
    next_object_id: int = 1
    last_run: float = 0.0
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
        proximity_distance_ratio: float,
        proximity_duration_seconds: float,
        group_distance_ratio: float,
        group_duration_seconds: float,
        detection_width: int,
        detection_height: int,
        exam_mode: bool,
        max_cameras_per_tick: int,
        face_identifier: Optional[FaceIdentifier] = None,
        attendance: Optional[AttendanceManager] = None,
        yolo_detector: Optional[YoloDetector] = None,
        overlay_store: Optional[OverlayStore] = None,
        object_allowlist: Tuple[str, ...] = (),
        object_priority: Tuple[str, ...] = (),
        object_risky: Tuple[str, ...] = (),
        object_label_map: Optional[Dict[str, Dict[str, str]]] = None,
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
        self.proximity_distance_ratio = proximity_distance_ratio
        self.proximity_duration_seconds = proximity_duration_seconds
        self.group_distance_ratio = group_distance_ratio
        self.group_duration_seconds = group_duration_seconds
        self.detection_width = detection_width
        self.detection_height = detection_height
        self.exam_mode = exam_mode
        self.max_cameras_per_tick = max(1, max_cameras_per_tick)
        self.face_identifier = face_identifier
        self.attendance = attendance
        self.yolo_detector = yolo_detector
        self.overlay_store = overlay_store
        self.object_allowlist = set(object_allowlist)
        self.object_priority = set(object_priority)
        self.object_risky = set(object_risky)
        self.object_label_map = object_label_map or {}

        self._cameras: Dict[str, Dict[str, CameraPerceptionState]] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._events: Deque[Dict[str, object]] = deque(maxlen=2000)
        self._resolver = GlobalIdentityResolver(
            similarity_threshold=global_similarity_threshold,
            max_age_seconds=global_max_age_seconds,
        )
        self._camera_order: List[Tuple[str, str]] = []
        self._camera_index = 0
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
            for camera_id in cameras.keys():
                self.add_camera(room_id, camera_id)

    def add_camera(self, room_id: str, camera_id: str) -> None:
        with self._lock:
            room = self._cameras.setdefault(room_id, {})
            if camera_id in room:
                return
            room[camera_id] = CameraPerceptionState(room_id=room_id, camera_id=camera_id)
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
        for event in reversed(self._events):
            if since is not None and _get_float(event, "timestamp") <= since:
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

            time.sleep(0.1)

    def _process_camera(self, state: CameraPerceptionState) -> None:
        start = time.time()
        success = False
        error: Optional[str] = None
        with state.lock:
            state.last_attempt_at = start
        try:
            frame, ts = self.stream_manager.get_snapshot(state.room_id, state.camera_id)
            if frame is None or ts is None:
                logger.debug(
                    "perception.frame_missing room_id=%s camera_id=%s",
                    state.room_id,
                    state.camera_id,
                )
                error = "frame_missing"
                return
            if time.time() - ts > self.stale_seconds:
                logger.debug(
                    "perception.frame_stale room_id=%s camera_id=%s",
                    state.room_id,
                    state.camera_id,
                )
                error = "frame_stale"
                return

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
            logger.debug(
                "perception.detect_people room_id=%s camera_id=%s count=%d",
                state.room_id,
                state.camera_id,
                len(detections),
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
                if self.overlay_store is not None:
                    annotations = []
                    for det in detections:
                        annotations.append(
                            {
                                "bbox": det.bbox,
                                "label": f"person:{det.confidence:.2f}",
                                "confidence": det.confidence,
                            }
                        )
                    for obj in objects:
                        annotations.append(
                            {
                                "bbox": obj.bbox,
                                "label": f"{obj.object_type}:{obj.confidence:.2f}",
                                "confidence": obj.confidence,
                            }
                        )
                    if annotations:
                        self.overlay_store.add_snapshot_all(
                            state.room_id,
                            state.camera_id,
                            annotations,
                            frame,
                            timestamp=ts,
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
            detections = []
            for det in detector.detect(frame):
                if det.label:  # defensive
                    if det.label == "person":
                        detections.append(Detection(det.bbox, det.confidence))
            return detections
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
                    },
                )
            , frame=frame)
            self._update_role(state, track, frame)
            self._update_orientation(state, track)
            matched_track_ids.add(track_id)
            matched_detection_ids.add(det_idx)

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
                    },
                )
            , frame=frame)
            self._update_role(state, track, frame)
            self._update_orientation(state, track)

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
                    {"track_id": track.track_id},
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
                        {"track_id": track.track_id, "role": track.role},
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
                    {"track_id": track.track_id, "role": track.role},
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
                        {"track_id": track.track_id, "role": track.role},
                    )
                , frame=frame)

    def _update_identity(
        self, state: CameraPerceptionState, track: TrackState, faces: List[FaceMatch]
    ) -> None:
        if not faces:
            return
        now = time.time()
        if now - track.last_identity_time < 2.0:
            return
        match = _match_face_to_track(faces, track.bbox)
        if match is None or match.person_id is None or match.name is None:
            return
        track.identity_id = match.person_id
        track.identity_name = match.name
        track.identity_role = match.role
        track.identity_score = match.score
        track.last_identity_time = now
        if self.attendance:
            self.attendance.mark_present(
                person_id=match.person_id,
                name=match.name,
                role=match.role or "unknown",
                camera_id=state.camera_id,
                timestamp=now,
            )

    def _update_orientation(self, state: CameraPerceptionState, track: TrackState) -> None:
        if not self.exam_mode:
            return
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
                    {"track_id": track.track_id, "orientation": orientation},
                )
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

        for det_idx, track_id in matches.items():
            detection = detections[det_idx]
            track = tracks[track_id]
            track.detection = detection
            track.last_seen = now
            track.hits += 1
            matched_track_ids.add(track_id)
            matched_detection_ids.add(det_idx)

        for idx, detection in enumerate(detections):
            if idx in matched_detection_ids:
                continue
            track_id = state.next_object_id
            state.next_object_id += 1
            track = ObjectTrack(track_id=track_id, detection=detection, last_seen=now, hits=1)
            tracks[track_id] = track

        expired = [
            track_id
            for track_id, track in tracks.items()
            if now - track.last_seen > self.object_ttl_seconds
        ]
        for track_id in expired:
            tracks.pop(track_id, None)

        for track in tracks.values():
            if track.hits >= self.object_persist_frames and not track.emitted:
                track.emitted = True
                detection = self._apply_object_flags(track.detection)
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
            if obj_track.hits < self.object_persist_frames:
                continue
            best_track = None
            best_score = 0.0
            for track in state.tracks.values():
                iou = _bbox_iou(track.bbox, obj_track.detection.bbox)
                if iou > best_score:
                    best_score = iou
                    best_track = track
            if best_track is None:
                continue
            if best_score < 0.05:
                continue
            key: Tuple[int, int] = (obj_track.track_id, best_track.track_id)
            last_emit = state.association_last.get(key, 0.0)
            now = time.time()
            if now - last_emit < 2.0:
                continue
            state.association_last[key] = now
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
                    },
                )
            , frame=frame)

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
                if not self._object_allowed("concealed_paper"):
                    continue
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
            if len(group) < 2:
                continue
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
                                "members": [t.global_id for t in group],
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
                                "members": [t.global_id for t in group],
                                "track_ids": [t.track_id for t in group],
                                "duration_seconds": duration,
                            },
                        )
                    )

        for key in list(state.group_state.keys()):
            if key not in active_keys:
                state.group_state.pop(key, None)

    def _object_allowed(self, object_type: str) -> bool:
        if not self.object_allowlist:
            return True
        return object_type in self.object_allowlist

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
    data: Dict[str, object] = {
        "timestamp": time.time(),
        "room_id": state.room_id,
        "camera_id": state.camera_id,
        "global_person_id": global_person_id,
        "event_type": event_type,
        "confidence": float(max(0.0, min(1.0, confidence))),
    }
    data.update(payload)
    return data


def _bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


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
