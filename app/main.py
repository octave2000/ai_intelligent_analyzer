import logging
import os

from fastapi import FastAPI

from app.api import build_router
from app.attendance_manager import AttendanceManager
from app.config import settings
from app.face_identifier import FaceIdentifier
from app.inference_manager import InferenceManager
from app.overlay_store import OverlayStore
from app.perception_manager import PerceptionManager
from app.stream_manager import StreamManager
from app.yolo_detector import YoloDetector


def create_app() -> FastAPI:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
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
    attendance = AttendanceManager(storage_path=settings.attendance_path)
    overlay_store = OverlayStore(
        root_path=settings.overlay_path,
        retention_seconds=settings.overlay_retention_seconds,
        flush_interval_seconds=settings.overlay_flush_interval_seconds,
        person_conf_threshold=settings.overlay_person_conf_threshold,
        object_conf_threshold=settings.overlay_object_conf_threshold,
        disk_retention_seconds=settings.overlay_disk_retention_seconds,
        cleanup_interval_seconds=settings.overlay_cleanup_interval_seconds,
        snapshot_enabled=settings.overlay_snapshot_enabled,
        snapshot_path=settings.overlay_snapshot_path,
        snapshot_all=settings.overlay_snapshot_all,
        snapshot_min_interval_seconds=settings.overlay_snapshot_min_interval_seconds,
    )
    face_identifier = FaceIdentifier(
        roster_path=settings.roster_path,
        similarity_threshold=settings.face_similarity_threshold,
        det_min_score=settings.face_det_min_score,
        enhance_enable=settings.face_enhance_enable,
        enhance_gamma=settings.face_enhance_gamma,
        enhance_clahe=settings.face_enhance_clahe,
        enhance_denoise=settings.face_enhance_denoise,
        enhance_sharpen=settings.face_enhance_sharpen,
        enhance_upscale_enable=settings.face_enhance_upscale_enable,
        enhance_upscale_min_dim=settings.face_enhance_upscale_min_dim,
        enhance_upscale_max_dim=settings.face_enhance_upscale_max_dim,
        model_name=settings.face_model_name,
        model_root=settings.face_model_root,
        ctx_id=settings.face_ctx_id,
    )
    yolo_detector = None
    if settings.yolo_mode != "disable":
        yolo_candidate = YoloDetector(
            model_path=settings.yolo_model_path,
            conf_threshold=settings.yolo_conf_threshold,
            iou_threshold=settings.yolo_iou_threshold,
            device=settings.yolo_device,
        )
        if yolo_candidate.ready() or settings.yolo_mode == "force":
            yolo_detector = yolo_candidate
    if yolo_detector is None:
        logging.getLogger(__name__).info(
            "yolo_detector.disabled mode=%s", settings.yolo_mode
        )
    else:
        logging.getLogger(__name__).info(
            "yolo_detector.enabled mode=%s model_path=%s",
            settings.yolo_mode,
            settings.yolo_model_path,
        )
    perception = PerceptionManager(
        stream_manager=manager,
        active_interval_seconds=settings.perception_active_interval_seconds,
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
        identity_min_interval_seconds=settings.perception_identity_min_interval_seconds,
        identity_sticky_score=settings.perception_identity_sticky_score,
        proximity_distance_ratio=settings.perception_proximity_distance_ratio,
        proximity_duration_seconds=settings.perception_proximity_duration_seconds,
        group_distance_ratio=settings.perception_group_distance_ratio,
        group_duration_seconds=settings.perception_group_duration_seconds,
        detection_width=settings.perception_detection_width,
        detection_height=settings.perception_detection_height,
        exam_mode=settings.perception_exam_mode,
        max_cameras_per_tick=settings.perception_max_cameras_per_tick,
        face_identifier=face_identifier,
        attendance=attendance,
        yolo_detector=yolo_detector,
        overlay_store=overlay_store,
        object_allowlist=settings.object_allowlist,
        object_priority=settings.object_priority,
        object_risky=settings.object_risky,
        object_label_map=settings.object_label_map,
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

    app.include_router(
        build_router(
            manager,
            perception,
            inference,
            attendance,
            yolo_detector,
        )
    )
    return app


app = create_app()
