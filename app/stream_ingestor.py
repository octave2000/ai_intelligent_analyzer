import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2


@dataclass
class StreamMetrics:
    is_running: bool = False
    frames_received_total: int = 0
    restart_count: int = 0
    last_error_message: Optional[str] = None


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
        use_ffmpeg: str,
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
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"StreamIngestor-{self.stream_id}", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def snapshot(
        self,
    ) -> Tuple[Optional["cv2.typing.MatLike"], Optional[float]]:
        with self._lock:
            if self._last_frame is None:
                return None, None
            return self._last_frame, self._last_timestamp

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
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _run(self) -> None:
        self.metrics.is_running = True
        backoff = self.backoff_min_seconds
        target_interval = 1.0 / float(self.target_fps)

        while not self._stop_event.is_set():
            cap = self._open_capture()
            if cap is None:
                self.metrics.last_error_message = "Failed to open stream"
                self.metrics.restart_count += 1
                time.sleep(backoff)
                backoff = min(self.backoff_max_seconds, backoff * 2.0)
                continue

            backoff = self.backoff_min_seconds
            last_ok = time.time()
            next_frame_time = time.monotonic()

            try:
                while not self._stop_event.is_set():
                    now = time.monotonic()
                    if now < next_frame_time:
                        time.sleep(min(0.05, next_frame_time - now))
                        continue
                    next_frame_time = now + target_interval

                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if time.time() - last_ok >= self.read_timeout_seconds:
                            self.metrics.last_error_message = (
                                f"Read timed out after {self.read_timeout_seconds:.1f}s"
                            )
                            self.metrics.restart_count += 1
                            break
                        time.sleep(0.05)
                        continue

                    last_ok = time.time()
                    if (
                        frame.shape[1] != self.frame_width
                        or frame.shape[0] != self.frame_height
                    ):
                        frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                    with self._lock:
                        self._last_frame = frame
                        self._last_timestamp = last_ok
                    self.metrics.frames_received_total += 1
                    self.metrics.last_error_message = None
            except Exception as exc:
                self.metrics.last_error_message = f"Stream error: {exc}"
                self.metrics.restart_count += 1
            finally:
                cap.release()

            if self._stop_event.is_set():
                break
            time.sleep(backoff)
            backoff = min(self.backoff_max_seconds, backoff * 2.0)

        self.metrics.is_running = False
