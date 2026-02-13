import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.teacher_zones import TeacherZoneConfig

logger = logging.getLogger(__name__)


@dataclass
class TrackSummary:
    bbox: Tuple[int, int, int, int]
    global_id: Optional[int]
    person_id: Optional[str]
    role: str


@dataclass
class CameraAttentionState:
    room_id: str
    camera_id: str
    role: str
    frame: Optional["cv2.typing.MatLike"] = None
    timestamp: Optional[float] = None
    tracks: List[TrackSummary] = field(default_factory=list)
    updated_at: float = 0.0
    last_processed_at: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)


class AttentionManager:
    def __init__(
        self,
        interval_seconds: float,
        downscale_width: int,
        downscale_height: int,
        max_faces: int,
        zone_config: TeacherZoneConfig,
        emit_callback,
    ) -> None:
        self.interval_seconds = max(0.2, interval_seconds)
        self.downscale_width = max(160, downscale_width)
        self.downscale_height = max(120, downscale_height)
        self.max_faces = max(1, max_faces)
        self.zone_config = zone_config
        self._emit_callback = emit_callback
        self._cameras: Dict[str, CameraAttentionState] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._mp_face = None

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

    def submit(
        self,
        room_id: str,
        camera_id: str,
        role: str,
        frame: "cv2.typing.MatLike",
        timestamp: float,
        tracks: List[TrackSummary],
    ) -> None:
        key = f"{room_id}:{camera_id}"
        with self._lock:
            state = self._cameras.get(key)
            if state is None:
                state = CameraAttentionState(
                    room_id=room_id,
                    camera_id=camera_id,
                    role=role,
                )
                self._cameras[key] = state
        with state.lock:
            state.frame = frame
            state.timestamp = timestamp
            state.tracks = tracks
            state.updated_at = time.time()

    def _run(self) -> None:
        self._init_mp()
        while not self._stop_event.is_set():
            with self._lock:
                states = list(self._cameras.values())
            now = time.time()
            for state in states:
                if now - state.last_processed_at < self.interval_seconds:
                    continue
                with state.lock:
                    frame = state.frame
                    timestamp = state.timestamp
                    tracks = list(state.tracks)
                    state.last_processed_at = now
                if frame is None or timestamp is None or not tracks:
                    continue
                try:
                    events = self._process_frame(state, frame, timestamp, tracks)
                    for event in events:
                        self._emit_event(event)
                except Exception:
                    logger.exception(
                        "attention.process_failed room_id=%s camera_id=%s",
                        state.room_id,
                        state.camera_id,
                    )
            time.sleep(0.05)

    def _init_mp(self) -> None:
        try:
            import mediapipe as mp  # type: ignore
        except Exception:
            self._mp_face = None
            logger.warning("attention.unavailable reason=mediapipe_missing")
            return
        self._mp_face = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_faces,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def _process_frame(
        self,
        state: CameraAttentionState,
        frame: "cv2.typing.MatLike",
        timestamp: float,
        tracks: List[TrackSummary],
    ) -> List[Dict[str, object]]:
        if self._mp_face is None:
            return []
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.downscale_width, self.downscale_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        results = self._mp_face.process(rgb)
        if not results.multi_face_landmarks:
            return []

        scale_x = self.downscale_width / float(w)
        scale_y = self.downscale_height / float(h)
        track_boxes = [
            (
                _scale_bbox(t.bbox, scale_x, scale_y),
                t,
            )
            for t in tracks
        ]
        direction = self.zone_config.direction_for(
            state.room_id, state.camera_id, state.role
        )
        events: List[Dict[str, object]] = []

        for face in results.multi_face_landmarks:
            bbox = _face_bbox(face, self.downscale_width, self.downscale_height)
            if bbox is None:
                continue
            yaw, pitch, roll = _estimate_head_pose(
                face, self.downscale_width, self.downscale_height
            )
            if yaw is None or pitch is None or roll is None:
                continue
            track = _match_track(bbox, track_boxes)
            if track is None:
                continue
            focus_mode = _focus_mode(yaw, pitch, direction)
            events.append(
                {
                    "timestamp": timestamp,
                    "room_id": state.room_id,
                    "camera_id": state.camera_id,
                    "global_person_id": track.global_id,
                    "person_id": track.person_id,
                    "event_type": "attention_observation",
                    "confidence": 0.7,
                    "yaw": yaw,
                    "pitch": pitch,
                    "roll": roll,
                    "focus_mode": focus_mode,
                    "teacher_direction": direction,
                }
            )
        return events

    def _emit_event(self, event: Dict[str, object]) -> None:
        if self._emit_callback is not None:
            self._emit_callback(event)


def _scale_bbox(
    bbox: Tuple[int, int, int, int], sx: float, sy: float
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy))


def _match_track(
    face_bbox: Tuple[int, int, int, int],
    track_boxes: List[Tuple[Tuple[int, int, int, int], TrackSummary]],
) -> Optional[TrackSummary]:
    best = None
    best_iou = 0.0
    for bbox, track in track_boxes:
        iou = _bbox_iou(face_bbox, bbox)
        if iou > best_iou:
            best_iou = iou
            best = track
    if best_iou < 0.08:
        return None
    return best


def _face_bbox(face, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    xs = []
    ys = []
    for lm in face.landmark:
        xs.append(lm.x * w)
        ys.append(lm.y * h)
    if not xs or not ys:
        return None
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    return (max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))


def _estimate_head_pose(face, w: int, h: int) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    # 2D landmarks
    idx = {
        "nose": 1,
        "chin": 152,
        "left_eye": 33,
        "right_eye": 263,
        "left_mouth": 61,
        "right_mouth": 291,
    }
    points_2d = []
    for key in ("nose", "chin", "left_eye", "right_eye", "left_mouth", "right_mouth"):
        lm = face.landmark[idx[key]]
        points_2d.append((lm.x * w, lm.y * h))

    # Approximate 3D model points
    points_3d = [
        (0.0, 0.0, 0.0),        # nose
        (0.0, -63.6, -12.5),    # chin
        (-43.3, 32.7, -26.0),   # left eye
        (43.3, 32.7, -26.0),    # right eye
        (-28.9, -28.9, -24.1),  # left mouth
        (28.9, -28.9, -24.1),   # right mouth
    ]
    camera_matrix = np.array(
        [[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], dtype=np.float32
    )
    dist_coeffs = np.zeros((4, 1))
    success, rot_vec, _ = cv2.solvePnP(
        np.array(points_3d, dtype=np.float32),
        np.array(points_2d, dtype=np.float32),
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None, None
    rot_mat, _ = cv2.Rodrigues(rot_vec)
    yaw, pitch, roll = _rotation_matrix_to_euler_angles(rot_mat)
    return yaw, pitch, roll


def _rotation_matrix_to_euler_angles(rmat: np.ndarray) -> Tuple[float, float, float]:
    sy = (rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0]) ** 0.5
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rmat[2, 1], rmat[2, 2])
        y = np.arctan2(-rmat[2, 0], sy)
        z = np.arctan2(rmat[1, 0], rmat[0, 0])
    else:
        x = np.arctan2(-rmat[1, 2], rmat[1, 1])
        y = np.arctan2(-rmat[2, 0], sy)
        z = 0.0
    return float(np.degrees(y)), float(np.degrees(x)), float(np.degrees(z))


def _focus_mode(yaw: float, pitch: float, direction: str) -> str:
    if pitch > 15.0:
        return "notebook"
    if direction == "top":
        if abs(yaw) < 20.0 and pitch < 10.0:
            return "teacher"
    if direction == "left" and yaw < -15.0:
        return "teacher"
    if direction == "right" and yaw > 15.0:
        return "teacher"
    if direction == "top-left" and yaw < -10.0 and pitch < 10.0:
        return "teacher"
    if direction == "top-right" and yaw > 10.0 and pitch < 10.0:
        return "teacher"
    return "unknown"


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
