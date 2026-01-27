from fastapi import FastAPI

from app.api import build_router
from app.attendance_manager import AttendanceManager
from app.config import settings
from app.face_identifier import FaceIdentifier
from app.inference_manager import InferenceManager
from app.motion_gate import MotionGateManager
from app.overlay_store import OverlayStore
from app.perception_manager import PerceptionManager
from app.stream_manager import StreamManager
from app.yolo_detector import YoloDetector


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
    attendance = AttendanceManager(storage_path=settings.attendance_path)
    overlay_store = OverlayStore(
        root_path=settings.overlay_path,
        retention_seconds=settings.overlay_retention_seconds,
        flush_interval_seconds=settings.overlay_flush_interval_seconds,
    )
    face_identifier = FaceIdentifier(
        roster_path=settings.roster_path,
        similarity_threshold=settings.face_similarity_threshold,
        model_name=settings.face_model_name,
        model_root=settings.face_model_root,
    )
    yolo_detector = None
    if settings.yolo_mode != "disable":
        yolo_candidate = YoloDetector(
            model_path=settings.yolo_model_path,
            conf_threshold=settings.yolo_conf_threshold,
            iou_threshold=settings.yolo_iou_threshold,
        )
        if yolo_candidate.ready() or settings.yolo_mode == "force":
            yolo_detector = yolo_candidate
    perception = PerceptionManager(
        stream_manager=manager,
        gate=gate,
        active_interval_seconds=settings.perception_active_interval_seconds,
        spike_interval_seconds=settings.perception_spike_interval_seconds,
        spike_burst_seconds=settings.perception_spike_burst_seconds,
        stale_seconds=settings.perception_stale_seconds,
        track_ttl_seconds=settings.perception_track_ttl_seconds,
        object_ttl_seconds=settings.perception_object_ttl_seconds,
        object_persist_frames=settings.perception_object_persist_frames,
        person_iou_threshold=settings.perception_person_iou_threshold,
        object_iou_threshold=settings.perception_object_iou_threshold,
        global_similarity_threshold=settings.perception_global_similarity_threshold,
        global_max_age_seconds=settings.perception_global_max_age_seconds,
        uniform_hsv_low=settings.perception_uniform_hsv_low,
        uniform_hsv_high=settings.perception_uniform_hsv_high,
        uniform_min_ratio=settings.perception_uniform_min_ratio,
        teacher_height_ratio=settings.perception_teacher_height_ratio,
        orientation_motion_threshold=settings.perception_orientation_motion_threshold,
        proximity_distance_ratio=settings.perception_proximity_distance_ratio,
        proximity_duration_seconds=settings.perception_proximity_duration_seconds,
        group_distance_ratio=settings.perception_group_distance_ratio,
        group_duration_seconds=settings.perception_group_duration_seconds,
        detection_width=settings.perception_detection_width,
        detection_height=settings.perception_detection_height,
        exam_mode=settings.perception_exam_mode,
        face_identifier=face_identifier,
        attendance=attendance,
        yolo_detector=yolo_detector,
        overlay_store=overlay_store,
    )
    perception.bootstrap_from_stream_manager()
    inference = InferenceManager(
        perception=perception,
        exam_mode=settings.perception_exam_mode,
        cheating_window_seconds=settings.inference_cheating_window_seconds,
        cheating_emit_interval_seconds=settings.inference_cheating_emit_interval_seconds,
        teacher_window_seconds=settings.inference_teacher_window_seconds,
        teacher_emit_interval_seconds=settings.inference_teacher_emit_interval_seconds,
        participation_window_seconds=settings.inference_participation_window_seconds,
        participation_emit_interval_seconds=settings.inference_participation_emit_interval_seconds,
        sync_turn_window_seconds=settings.inference_sync_turn_window_seconds,
        fight_window_seconds=settings.inference_fight_window_seconds,
        fight_emit_interval_seconds=settings.inference_fight_emit_interval_seconds,
        fight_motion_threshold=settings.inference_fight_motion_threshold,
        fight_proximity_threshold=settings.inference_fight_proximity_threshold,
    )

    @app.on_event("startup")
    def _startup() -> None:
        manager.start()
        perception.start()
        inference.start()
        overlay_store.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        manager.stop()
        perception.stop()
        inference.stop()
        overlay_store.stop()

    app.include_router(build_router(manager, gate, perception, inference, attendance))
    return app


app = create_app()
