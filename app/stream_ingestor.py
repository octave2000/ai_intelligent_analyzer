import os
import select
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FrameBuffer:
    frame: Optional["cv2.typing.MatLike"] = None
    timestamp: Optional[float] = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(self, frame: "cv2.typing.MatLike", timestamp: float) -> None:
        with self.lock:
            self.frame = frame
            self.timestamp = timestamp

    def read(self) -> Tuple[Optional["cv2.typing.MatLike"], Optional[float]]:
        with self.lock:
            return self.frame, self.timestamp


@dataclass
class StreamMetrics:
    is_running: bool = False
    last_frame_timestamp: Optional[float] = None
    frames_received_total: int = 0
    restart_count: int = 0
    last_error_message: Optional[str] = None


class StreamIngestor:
    def __init__(
        self,
        name: str,
        url: str,
        frame_width: int,
        frame_height: int,
        target_fps: int,
        backoff_min_seconds: float,
        backoff_max_seconds: float,
        read_timeout_seconds: float,
        use_ffmpeg: str,
    ) -> None:
        self.name = name
        self.url = url
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.target_fps = max(1, target_fps)
        self.backoff_min_seconds = backoff_min_seconds
        self.backoff_max_seconds = backoff_max_seconds
        self.read_timeout_seconds = read_timeout_seconds
        self.use_ffmpeg = use_ffmpeg
        self.metrics = StreamMetrics()
        self.buffer = FrameBuffer()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.metrics.is_running = False

    def snapshot(self) -> Tuple[Optional["cv2.typing.MatLike"], Optional[float]]:
        return self.buffer.read()

    def _should_use_ffmpeg(self) -> bool:
        if self.use_ffmpeg == "force":
            return True
        if self.use_ffmpeg == "disable":
            return False
        return self.url.startswith("rtsp://")

    def _run(self) -> None:
        backoff = self.backoff_min_seconds
        interval = 1.0 / float(self.target_fps)

        while not self._stop_event.is_set():
            capture = None
            try:
                self.metrics.is_running = True
                if self._should_use_ffmpeg():
                    capture = FfmpegCapture(
                        self.url,
                        self.frame_width,
                        self.frame_height,
                        self.target_fps,
                        self.read_timeout_seconds,
                    )
                    if not capture.is_opened:
                        raise RuntimeError("unable to open stream")
                else:
                    capture = cv2.VideoCapture(self.url)
                    if not capture.isOpened():
                        raise RuntimeError("unable to open stream")

                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.frame_width))
                    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.frame_height))

                self.metrics.last_error_message = None
                backoff = self.backoff_min_seconds

                while not self._stop_event.is_set():
                    start = time.time()
                    ok, frame = capture.read()
                    if not ok:
                        raise RuntimeError("failed to read frame")

                    timestamp = time.time()
                    if frame is not None:
                        resized = cv2.resize(frame, (self.frame_width, self.frame_height))
                        self.buffer.update(resized, timestamp)
                        self.metrics.last_frame_timestamp = timestamp
                        self.metrics.frames_received_total += 1

                    elapsed = time.time() - start
                    if elapsed < interval:
                        time.sleep(interval - elapsed)
            except Exception as exc:
                self.metrics.last_error_message = str(exc)
                self.metrics.restart_count += 1
                if capture is not None:
                    capture.release()
                self.metrics.is_running = False
                if self._stop_event.is_set():
                    break
                time.sleep(backoff)
                backoff = min(self.backoff_max_seconds, backoff * 2.0)
            finally:
                if capture is not None:
                    capture.release()


class FfmpegCapture:
    def __init__(
        self,
        url: str,
        width: int,
        height: int,
        target_fps: int,
        read_timeout_seconds: float,
    ) -> None:
        self.url = url
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.read_timeout_seconds = read_timeout_seconds
        self.frame_size = self.width * self.height * 3
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-rtsp_transport",
                "tcp",
                "-fflags",
                "nobuffer+discardcorrupt",
                "-flags",
                "low_delay",
                "-err_detect",
                "ignore_err",
                "-an",
                "-sn",
                "-dn",
                "-i",
                self.url,
                "-vf",
                f"scale={self.width}:{self.height}",
                "-r",
                str(self.target_fps),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    @property
    def is_opened(self) -> bool:
        return self.process.stdout is not None and self.process.poll() is None

    def read(self) -> Tuple[bool, Optional["cv2.typing.MatLike"]]:
        if not self.is_opened:
            return False, None
        raw = self._read_exact(self.frame_size, self.read_timeout_seconds)
        if raw is None:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8)
        frame = frame.reshape((self.height, self.width, 3))
        return True, frame

    def _read_exact(self, size: int, timeout: float) -> Optional[bytes]:
        if self.process.stdout is None:
            return None
        buf = bytearray()
        deadline = time.time() + timeout
        fd = self.process.stdout.fileno()
        while len(buf) < size:
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            rlist, _, _ = select.select([fd], [], [], remaining)
            if not rlist:
                return None
            chunk = os.read(fd, size - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def release(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
