import os
from typing import Optional

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(".env", override=False)


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_face_ctx_id(name: str, default: str = "auto") -> int:
    raw = os.getenv(name, default).strip().lower()
    if raw in ("auto", ""):
        try:
            import onnxruntime as ort  # type: ignore[import-not-found]

            return 0 if ort.get_device().upper() == "GPU" else -1
        except Exception:
            return -1
    try:
        return int(raw)
    except ValueError:
        return -1


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_ffmpeg_mode() -> str:
    raw = os.getenv("USE_FFMPEG", "auto").strip().lower()
    if raw in ("1", "true", "yes", "force", "on"):
        return "force"
    if raw in ("0", "false", "no", "disable", "off"):
        return "disable"
    return "auto"


def _get_mode(name: str, default: str) -> str:
    raw = os.getenv(name, default).strip().lower()
    if raw in ("force", "on", "1", "true", "yes"):
        return "force"
    if raw in ("disable", "off", "0", "false", "no"):
        return "disable"
    return "auto"


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_int_tuple(name: str, default: tuple) -> tuple:
    raw = os.getenv(name)
    if raw is None:
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 3:
        return default
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return default


def _get_str_tuple(name: str, default: tuple) -> tuple:
    raw = os.getenv(name)
    if raw is None:
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        return default
    return tuple(parts)


def _load_object_config(
    path: str,
    default_allowlist: tuple,
    default_priority: tuple,
    default_risky: tuple,
    default_label_map: dict,
) -> tuple:
    if not path:
        return default_allowlist, default_priority, default_risky, default_label_map
    try:
        import json

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return default_allowlist, default_priority, default_risky, default_label_map

    if not isinstance(payload, dict):
        return default_allowlist, default_priority, default_risky, default_label_map

    def _read_list(key: str, fallback: tuple) -> tuple:
        value = payload.get(key)
        if not isinstance(value, list):
            return fallback
        items = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return tuple(items) if items else fallback

    allowlist = _read_list("allowlist", default_allowlist)
    priority = _read_list("priority", default_priority)
    risky = _read_list("risky", default_risky)

    label_map_raw = payload.get("label_map", {})
    label_map = default_label_map
    if isinstance(label_map_raw, dict):
        cleaned: dict = {}
        for key, value in label_map_raw.items():
            if not isinstance(key, str) or not isinstance(value, dict):
                continue
            object_type = value.get("object_type", key)
            category = value.get("category", "other")
            risk_level = value.get("risk_level", "low")
            if not isinstance(object_type, str) or not object_type.strip():
                continue
            if not isinstance(category, str) or not category.strip():
                category = "other"
            if not isinstance(risk_level, str) or not risk_level.strip():
                risk_level = "low"
            cleaned[key] = {
                "object_type": object_type.strip(),
                "category": category.strip(),
                "risk_level": risk_level.strip(),
            }
        if cleaned:
            label_map = cleaned

    return allowlist, priority, risky, label_map


def _load_mode_config(path: str, default_exam_mode: bool) -> bool:
    if not path:
        return default_exam_mode
    try:
        import json

        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return default_exam_mode
    if not isinstance(payload, dict):
        return default_exam_mode
    value = payload.get("exam_mode", default_exam_mode)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(value, (int, float)):
        return bool(value)
    return default_exam_mode


class Settings:
    frame_width: int
    frame_height: int
    target_fps: int
    stale_threshold_seconds: float
    backoff_min_seconds: float
    backoff_max_seconds: float
    read_timeout_seconds: float
    use_ffmpeg: str
    storage_path: str
    perception_active_interval_seconds: float
    perception_stale_seconds: float
    perception_track_ttl_seconds: float
    perception_object_ttl_seconds: float
    perception_object_persist_frames: int
    perception_person_iou_threshold: float
    perception_object_iou_threshold: float
    perception_global_similarity_threshold: float
    perception_global_max_age_seconds: float
    perception_uniform_hsv_low: tuple
    perception_uniform_hsv_high: tuple
    perception_uniform_min_ratio: float
    perception_teacher_height_ratio: float
    perception_orientation_motion_threshold: float
    perception_identity_min_interval_seconds: float
    perception_identity_sticky_score: float
    perception_proximity_distance_ratio: float
    perception_proximity_duration_seconds: float
    perception_group_distance_ratio: float
    perception_group_duration_seconds: float
    perception_detection_width: int
    perception_detection_height: int
    perception_exam_mode: bool
    perception_max_cameras_per_tick: int
    roster_path: str
    attendance_path: str
    face_similarity_threshold: float
    face_det_min_score: float
    face_enhance_enable: bool
    face_enhance_gamma: float
    face_enhance_clahe: bool
    face_enhance_denoise: bool
    face_enhance_sharpen: bool
    face_enhance_upscale_enable: bool
    face_enhance_upscale_min_dim: int
    face_enhance_upscale_max_dim: int
    face_model_name: str
    face_model_root: Optional[str]
    face_ctx_id: int
    yolo_mode: str
    yolo_model_path: str
    yolo_conf_threshold: float
    yolo_iou_threshold: float
    yolo_device: str
    overlay_path: str
    overlay_retention_seconds: float
    overlay_flush_interval_seconds: float
    overlay_person_conf_threshold: float
    overlay_object_conf_threshold: float
    overlay_disk_retention_seconds: float
    overlay_cleanup_interval_seconds: float
    overlay_snapshot_enabled: bool
    overlay_snapshot_path: str
    overlay_snapshot_all: bool
    overlay_snapshot_min_interval_seconds: float
    rtsp_transport: str
    object_allowlist: tuple
    object_priority: tuple
    object_risky: tuple
    object_label_map: dict
    object_config_path: str
    mode_config_path: str
    inference_cheating_window_seconds: float
    inference_cheating_emit_interval_seconds: float
    inference_teacher_window_seconds: float
    inference_teacher_emit_interval_seconds: float
    inference_participation_window_seconds: float
    inference_participation_emit_interval_seconds: float
    inference_sync_turn_window_seconds: float
    inference_fight_window_seconds: float
    inference_fight_emit_interval_seconds: float
    inference_fight_motion_threshold: float
    inference_fight_proximity_threshold: int

    def __init__(self) -> None:
        self.frame_width = _get_int("FRAME_WIDTH", 640)
        self.frame_height = _get_int("FRAME_HEIGHT", 360)
        self.target_fps = _get_int("TARGET_FPS", 5)
        self.stale_threshold_seconds = _get_float("STALE_THRESHOLD_SECONDS", 2.5)
        self.backoff_min_seconds = _get_float("BACKOFF_MIN_SECONDS", 1.0)
        self.backoff_max_seconds = _get_float("BACKOFF_MAX_SECONDS", 30.0)
        self.read_timeout_seconds = _get_float("READ_TIMEOUT_SECONDS", 10.0)
        self.use_ffmpeg = _get_ffmpeg_mode()
        self.storage_path = os.getenv("STORAGE_PATH", "data/rooms.json")
        self.perception_active_interval_seconds = _get_float(
            "PERCEPTION_ACTIVE_INTERVAL_SECONDS", 2.0
        )
        self.perception_stale_seconds = _get_float("PERCEPTION_STALE_SECONDS", 2.5)
        self.perception_track_ttl_seconds = _get_float("PERCEPTION_TRACK_TTL_SECONDS", 2.5)
        self.perception_object_ttl_seconds = _get_float("PERCEPTION_OBJECT_TTL_SECONDS", 3.0)
        self.perception_object_persist_frames = _get_int(
            "PERCEPTION_OBJECT_PERSIST_FRAMES", 2
        )
        self.perception_person_iou_threshold = _get_float(
            "PERCEPTION_PERSON_IOU_THRESHOLD", 0.2
        )
        self.perception_object_iou_threshold = _get_float(
            "PERCEPTION_OBJECT_IOU_THRESHOLD", 0.15
        )
        self.perception_global_similarity_threshold = _get_float(
            "PERCEPTION_GLOBAL_SIMILARITY_THRESHOLD", 0.85
        )
        self.perception_global_max_age_seconds = _get_float(
            "PERCEPTION_GLOBAL_MAX_AGE_SECONDS", 10.0
        )
        self.perception_uniform_hsv_low = _get_int_tuple(
            "PERCEPTION_UNIFORM_HSV_LOW", (0, 0, 0)
        )
        self.perception_uniform_hsv_high = _get_int_tuple(
            "PERCEPTION_UNIFORM_HSV_HIGH", (179, 255, 255)
        )
        self.perception_uniform_min_ratio = _get_float(
            "PERCEPTION_UNIFORM_MIN_RATIO", 1.1
        )
        self.perception_teacher_height_ratio = _get_float(
            "PERCEPTION_TEACHER_HEIGHT_RATIO", 0.6
        )
        self.perception_orientation_motion_threshold = _get_float(
            "PERCEPTION_ORIENTATION_MOTION_THRESHOLD", 10.0
        )
        self.perception_identity_min_interval_seconds = _get_float(
            "PERCEPTION_IDENTITY_MIN_INTERVAL_SECONDS", 0.3
        )
        self.perception_identity_sticky_score = _get_float(
            "PERCEPTION_IDENTITY_STICKY_SCORE", 0.4
        )
        self.perception_proximity_distance_ratio = _get_float(
            "PERCEPTION_PROXIMITY_DISTANCE_RATIO", 0.15
        )
        self.perception_proximity_duration_seconds = _get_float(
            "PERCEPTION_PROXIMITY_DURATION_SECONDS", 2.0
        )
        self.perception_group_distance_ratio = _get_float(
            "PERCEPTION_GROUP_DISTANCE_RATIO", 0.2
        )
        self.perception_group_duration_seconds = _get_float(
            "PERCEPTION_GROUP_DURATION_SECONDS", 3.0
        )
        self.perception_detection_width = _get_int("PERCEPTION_DETECTION_WIDTH", 640)
        self.perception_detection_height = _get_int("PERCEPTION_DETECTION_HEIGHT", 360)
        self.perception_max_cameras_per_tick = _get_int(
            "PERCEPTION_MAX_CAMERAS_PER_TICK", 8
        )
        self.mode_config_path = "data/modes.json"
        self.perception_exam_mode = _load_mode_config(
            self.mode_config_path, default_exam_mode=False
        )
        self.roster_path = os.getenv("ROSTER_PATH", "data/roster.json")
        self.attendance_path = os.getenv("ATTENDANCE_PATH", "data/attendance.json")
        self.face_similarity_threshold = _get_float(
            "FACE_SIMILARITY_THRESHOLD", 0.25
        )
        self.face_det_min_score = _get_float("FACE_DET_MIN_SCORE", 0.2)
        self.face_enhance_enable = _get_bool("FACE_ENHANCE_ENABLE", True)
        self.face_enhance_gamma = _get_float("FACE_ENHANCE_GAMMA", 1.6)
        self.face_enhance_clahe = _get_bool("FACE_ENHANCE_CLAHE", True)
        self.face_enhance_denoise = _get_bool("FACE_ENHANCE_DENOISE", False)
        self.face_enhance_sharpen = _get_bool("FACE_ENHANCE_SHARPEN", False)
        self.face_enhance_upscale_enable = _get_bool(
            "FACE_ENHANCE_UPSCALE_ENABLE", True
        )
        self.face_enhance_upscale_min_dim = _get_int(
            "FACE_ENHANCE_UPSCALE_MIN_DIM", 1080
        )
        self.face_enhance_upscale_max_dim = _get_int(
            "FACE_ENHANCE_UPSCALE_MAX_DIM", 1920
        )
        self.face_model_name = os.getenv("FACE_MODEL_NAME", "buffalo_s")
        self.face_model_root = os.getenv("FACE_MODEL_ROOT", "") or None
        self.face_ctx_id = _get_face_ctx_id("FACE_CTX_ID", "auto")
        self.yolo_mode = _get_mode("USE_YOLO", "auto")
        self.yolo_model_path = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
        self.yolo_conf_threshold = _get_float("YOLO_CONF_THRESHOLD", 0.35)
        self.yolo_iou_threshold = _get_float("YOLO_IOU_THRESHOLD", 0.45)
        self.yolo_device = os.getenv("YOLO_DEVICE", "auto").strip().lower()
        self.overlay_path = os.getenv("OVERLAY_PATH", "data/overlay")
        self.overlay_retention_seconds = _get_float("OVERLAY_RETENTION_SECONDS", 60.0)
        self.overlay_flush_interval_seconds = _get_float(
            "OVERLAY_FLUSH_INTERVAL_SECONDS", 1.0
        )
        self.overlay_person_conf_threshold = _get_float(
            "OVERLAY_PERSON_CONF_THRESHOLD", 0.7
        )
        self.overlay_object_conf_threshold = _get_float(
            "OVERLAY_OBJECT_CONF_THRESHOLD", 0.5
        )
        self.overlay_disk_retention_seconds = _get_float(
            "OVERLAY_DISK_RETENTION_SECONDS", 86400.0
        )
        self.overlay_cleanup_interval_seconds = _get_float(
            "OVERLAY_CLEANUP_INTERVAL_SECONDS", 60.0
        )
        self.overlay_snapshot_enabled = _get_bool(
            "OVERLAY_SNAPSHOT_ENABLED", False
        )
        self.overlay_snapshot_path = os.getenv(
            "OVERLAY_SNAPSHOT_PATH", "data/overlay_snapshots"
        )
        self.overlay_snapshot_all = _get_bool("OVERLAY_SNAPSHOT_ALL", False)
        self.overlay_snapshot_min_interval_seconds = _get_float(
            "OVERLAY_SNAPSHOT_MIN_INTERVAL_SECONDS", 1.0
        )
        self.rtsp_transport = os.getenv("RTSP_TRANSPORT", "tcp").strip().lower()
        self.object_config_path = os.getenv("OBJECT_CONFIG_PATH", "data/objects.json")
        allowlist, priority, risky, label_map = _load_object_config(
            self.object_config_path,
            default_allowlist=(
                "phone",
                "laptop",
                "tablet",
                "book",
                "paper",
                "notebook",
                "calculator",
                "knife",
                "knife_like",
                "concealed_paper",
                "beaker",
                "test_tube",
                "burner",
                "backpack",
                "pouch",
                "device",
            ),
            default_priority=("phone", "knife", "knife_like"),
            default_risky=("knife", "knife_like", "concealed_paper"),
            default_label_map={},
        )
        self.object_allowlist = allowlist
        self.object_priority = priority
        self.object_risky = risky
        self.object_label_map = label_map
        self.inference_cheating_window_seconds = _get_float(
            "INFERENCE_CHEATING_WINDOW_SECONDS", 60.0
        )
        self.inference_cheating_emit_interval_seconds = _get_float(
            "INFERENCE_CHEATING_EMIT_INTERVAL_SECONDS", 10.0
        )
        self.inference_teacher_window_seconds = _get_float(
            "INFERENCE_TEACHER_WINDOW_SECONDS", 120.0
        )
        self.inference_teacher_emit_interval_seconds = _get_float(
            "INFERENCE_TEACHER_EMIT_INTERVAL_SECONDS", 30.0
        )
        self.inference_participation_window_seconds = _get_float(
            "INFERENCE_PARTICIPATION_WINDOW_SECONDS", 120.0
        )
        self.inference_participation_emit_interval_seconds = _get_float(
            "INFERENCE_PARTICIPATION_EMIT_INTERVAL_SECONDS", 30.0
        )
        self.inference_sync_turn_window_seconds = _get_float(
            "INFERENCE_SYNC_TURN_WINDOW_SECONDS", 2.0
        )
        self.inference_fight_window_seconds = _get_float(
            "INFERENCE_FIGHT_WINDOW_SECONDS", 8.0
        )
        self.inference_fight_emit_interval_seconds = _get_float(
            "INFERENCE_FIGHT_EMIT_INTERVAL_SECONDS", 10.0
        )
        self.inference_fight_motion_threshold = _get_float(
            "INFERENCE_FIGHT_MOTION_THRESHOLD", 250.0
        )
        self.inference_fight_proximity_threshold = _get_int(
            "INFERENCE_FIGHT_PROXIMITY_THRESHOLD", 2
        )


settings = Settings()
