from fastapi import FastAPI

from app.api import build_router
from app.config import settings
from app.motion_gate import MotionGateManager
from app.stream_manager import StreamManager


def create_app() -> FastAPI:
    app = FastAPI(title="Classroom AI - Part 1")
    manager = StreamManager(
        frame_width=settings.frame_width,
        frame_height=settings.frame_height,
        target_fps=settings.target_fps,
        backoff_min_seconds=settings.backoff_min_seconds,
        backoff_max_seconds=settings.backoff_max_seconds,
        stale_threshold_seconds=settings.stale_threshold_seconds,
        read_timeout_seconds=settings.read_timeout_seconds,
        use_ffmpeg=settings.use_ffmpeg,
        storage_path=settings.storage_path,
    )
    gate = MotionGateManager(
        stream_manager=manager,
        sample_interval=settings.gate_sample_interval_seconds,
        window_size=settings.gate_window_size,
        active_enter=settings.gate_active_enter,
        active_exit=settings.gate_active_exit,
        spike_enter=settings.gate_spike_enter,
        spike_exit=settings.gate_spike_exit,
        spike_cooldown_seconds=settings.gate_spike_cooldown_seconds,
        stale_seconds=settings.gate_stale_seconds,
        downsample_width=settings.gate_downsample_width,
        downsample_height=settings.gate_downsample_height,
    )
    gate.bootstrap_from_stream_manager()

    @app.on_event("startup")
    def _startup() -> None:
        manager.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        manager.stop()

    app.include_router(build_router(manager, gate))
    return app


app = create_app()
