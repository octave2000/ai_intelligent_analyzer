import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

# ------------------------------------------------------------------
# Force FFmpeg RTSP transport correctly (NOT via URL query params)
# ------------------------------------------------------------------
# Use 'tcp' for stability, 'udp' if your network is extremely clean.
# Respect user-provided OpenCV FFmpeg options if already set.
_rtsp_transport = os.getenv("RTSP_TRANSPORT", "tcp").strip().lower()
if _rtsp_transport not in {"tcp", "udp", "http", "https"}:
    _rtsp_transport = "tcp"
_default_ffmpeg_capture_options = (
    f"rtsp_transport;{_rtsp_transport}"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|max_delay;0"
    "|reorder_queue_size;0"
)
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    _default_ffmpeg_capture_options,
)

# Optional but recommended for low-latency live streams
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "error"

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
@dataclass
class StreamMetrics:
    is_running: bool = False
    frames_received_total: int = 0
    restart_count: int = 0
    last_error_message: Optional[str] = None


# ------------------------------------------------------------------
# Stream Ingestor
# ------------------------------------------------------------------
class StreamIngestor:
    def __init__(
        self,
        stream_id: str,
        url: str,
        frame_width: int,
        frame_height: int,
        target_fps: int,
        backoff_min_seconds: float,
        backoff_max_seconds: float,
        read_timeout_seconds: float,
        use_ffmpeg: str = "force",  # force | disable | auto
    ) -> None:
        self.stream_id = stream_id
        self.url = url
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = max(1, target_fps)

        self.backoff_min_seconds = max(0.1, backoff_min_seconds)
        self.backoff_max_seconds = max(self.backoff_min_seconds, backoff_max_seconds)
        self.read_timeout_seconds = max(0.5, read_timeout_seconds)

        self.use_ffmpeg = use_ffmpeg

        self.metrics = StreamMetrics()
        self._lock = threading.Lock()
        self._last_frame: Optional["cv2.typing.MatLike"] = None
        self._last_timestamp: Optional[float] = None
        self._last_arrival_timestamp: Optional[float] = None
        self._last_source_timestamp: Optional[float] = None
        self._source_epoch_wall: Optional[float] = None
        self._source_epoch_stream: Optional[float] = None
        self._max_backlog_seconds = 180.0

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        logger.info(
            "stream_ingestor.start stream_id=%s url=%s", self.stream_id, self.url
        )
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"StreamIngestor-{self.stream_id}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        logger.info("stream_ingestor.stop stream_id=%s", self.stream_id)

    def snapshot(
        self,
    ) -> Tuple[Optional["cv2.typing.MatLike"], Optional[float]]:
        with self._lock:
            if self._last_frame is None:
                return None, None
            ts = self._last_source_timestamp
            if ts is None:
                ts = self._last_arrival_timestamp
            return self._last_frame.copy(), ts

    def snapshot_meta(
        self,
    ) -> Tuple[Optional["cv2.typing.MatLike"], Optional[float], Optional[float]]:
        with self._lock:
            if self._last_frame is None:
                return None, None, None
            source_ts = self._last_source_timestamp
            if source_ts is None:
                source_ts = self._last_arrival_timestamp
            return self._last_frame.copy(), source_ts, self._last_arrival_timestamp

    # --------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------
    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        if self.use_ffmpeg == "force":
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        elif self.use_ffmpeg == "disable":
            cap = cv2.VideoCapture(self.url, cv2.CAP_ANY)
        else:
            cap = cv2.VideoCapture(self.url)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            cap.release()
            logger.warning(
                "stream_ingestor.open_failed stream_id=%s url=%s",
                self.stream_id,
                self.url,
            )
            return None

        # Reduce latency & buffering
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        return cap

    # --------------------------------------------------------------
    # Main loop
    # --------------------------------------------------------------
    def _run(self) -> None:
        self.metrics.is_running = True
        backoff = self.backoff_min_seconds
        target_interval = 1.0 / float(self.target_fps)

        while not self._stop_event.is_set():
            cap = self._open_capture()

            if cap is None:
                self.metrics.last_error_message = "Failed to open RTSP stream"
                self.metrics.restart_count += 1
                time.sleep(backoff)
                backoff = min(self.backoff_max_seconds, backoff * 2)
                continue

            backoff = self.backoff_min_seconds
            last_ok = time.time()
            next_store_time = time.monotonic()
            self._source_epoch_wall = None
            self._source_epoch_stream = None

            try:
                while not self._stop_event.is_set():
                    now = time.monotonic()
                    # Drain decoder buffer to reduce latency.
                    if now < next_store_time:
                        ok = cap.grab()
                        if ok:
                            last_ok = time.time()
                        else:
                            if time.time() - last_ok >= self.read_timeout_seconds:
                                self.metrics.last_error_message = (
                                    f"Read timeout ({self.read_timeout_seconds:.1f}s)"
                                )
                                self.metrics.restart_count += 1
                                break
                            time.sleep(0.01)
                        continue

                    ok, frame = cap.read()

                    if not ok or frame is None:
                        if time.time() - last_ok >= self.read_timeout_seconds:
                            self.metrics.last_error_message = (
                                f"Read timeout ({self.read_timeout_seconds:.1f}s)"
                            )
                            self.metrics.restart_count += 1
                            break
                        time.sleep(0.05)
                        continue

                    arrival_ts = time.time()
                    source_ts = self._estimate_source_timestamp(cap, arrival_ts)
                    last_ok = arrival_ts
                    next_store_time = now + target_interval

                    if (
                        frame.shape[1] != self.frame_width
                        or frame.shape[0] != self.frame_height
                    ):
                        frame = cv2.resize(
                            frame,
                            (self.frame_width, self.frame_height),
                            interpolation=cv2.INTER_LINEAR,
                        )

                    with self._lock:
                        self._last_frame = frame
                        self._last_arrival_timestamp = arrival_ts
                        self._last_source_timestamp = source_ts
                        self._last_timestamp = source_ts

                    self.metrics.frames_received_total += 1
                    self.metrics.last_error_message = None
                    logger.debug(
                        "stream_ingestor.frame stream_id=%s ts=%.3f arrival=%.3f",
                        self.stream_id,
                        source_ts,
                        arrival_ts,
                    )

            except Exception as exc:
                self.metrics.last_error_message = f"Stream error: {exc}"
                self.metrics.restart_count += 1

            finally:
                cap.release()

            if self._stop_event.is_set():
                break

            time.sleep(backoff)
            backoff = min(self.backoff_max_seconds, backoff * 2)

        self.metrics.is_running = False

    def _estimate_source_timestamp(
        self,
        cap: cv2.VideoCapture,
        arrival_ts: float,
    ) -> float:
        stream_seconds = self._read_stream_position_seconds(cap)
        if stream_seconds is None:
            return arrival_ts

        # Some backends expose absolute epoch seconds instead of relative stream position.
        if stream_seconds > 1e8:
            source_ts = stream_seconds
        else:
            if (
                self._source_epoch_wall is None
                or self._source_epoch_stream is None
                or stream_seconds + 1.0 < self._source_epoch_stream
            ):
                self._source_epoch_wall = arrival_ts - stream_seconds
            self._source_epoch_stream = stream_seconds
            source_ts = (self._source_epoch_wall or arrival_ts) + stream_seconds

        if not math.isfinite(source_ts):
            return arrival_ts
        if source_ts > arrival_ts + 2.0:
            return arrival_ts
        if arrival_ts - source_ts > self._max_backlog_seconds:
            return arrival_ts - self._max_backlog_seconds
        return source_ts

    @staticmethod
    def _read_stream_position_seconds(cap: cv2.VideoCapture) -> Optional[float]:
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if math.isfinite(pos_msec) and pos_msec > 0.0:
            return pos_msec / 1000.0
        pts_prop = getattr(cv2, "CAP_PROP_PTS", None)
        if pts_prop is None:
            return None
        pts = cap.get(pts_prop)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if math.isfinite(pts) and pts > 0.0 and math.isfinite(fps) and fps > 0.0:
            return pts / fps
        return None
