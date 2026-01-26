import os


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


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
    gate_sample_interval_seconds: float
    gate_window_size: int
    gate_active_enter: float
    gate_active_exit: float
    gate_spike_enter: float
    gate_spike_exit: float
    gate_spike_cooldown_seconds: float
    gate_stale_seconds: float
    gate_downsample_width: int
    gate_downsample_height: int

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
        self.gate_sample_interval_seconds = _get_float(
            "GATE_SAMPLE_INTERVAL_SECONDS", 1.0
        )
        self.gate_window_size = _get_int("GATE_WINDOW_SIZE", 5)
        self.gate_active_enter = _get_float("GATE_ACTIVE_ENTER", 0.08)
        self.gate_active_exit = _get_float("GATE_ACTIVE_EXIT", 0.04)
        self.gate_spike_enter = _get_float("GATE_SPIKE_ENTER", 0.35)
        self.gate_spike_exit = _get_float("GATE_SPIKE_EXIT", 0.2)
        self.gate_spike_cooldown_seconds = _get_float(
            "GATE_SPIKE_COOLDOWN_SECONDS", 3.0
        )
        self.gate_stale_seconds = _get_float("GATE_STALE_SECONDS", 2.5)
        self.gate_downsample_width = _get_int("GATE_DOWNSAMPLE_WIDTH", 64)
        self.gate_downsample_height = _get_int("GATE_DOWNSAMPLE_HEIGHT", 36)


settings = Settings()
