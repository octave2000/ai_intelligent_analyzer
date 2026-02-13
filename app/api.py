import time
from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from app.inference_manager import InferenceManager
from app.perception_manager import PerceptionManager
from app.stream_manager import StreamManager
from app.attendance_manager import AttendanceManager
from app.yolo_detector import YoloDetector


def _encode_jpeg(frame: "cv2.typing.MatLike") -> Optional[bytes]:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return encoded.tobytes()


def build_router(
    manager: StreamManager,
    perception: PerceptionManager,
    inference: InferenceManager,
    attendance: AttendanceManager,
    yolo_detector: Optional[YoloDetector],
) -> APIRouter:
    router = APIRouter()

    class RoomCreate(BaseModel):
        room_id: str = Field(..., min_length=1)

    class CameraCreate(BaseModel):
        camera_id: str = Field(..., min_length=1)
        url: str = Field(..., min_length=1)
        role: str = Field("other", pattern="^(front|back|other)$")

    @router.get("/health")
    def health() -> dict:
        payload = manager.health()
        payload["yolo"] = {
            "enabled": yolo_detector is not None,
            "ready": yolo_detector.ready() if yolo_detector is not None else False,
            "model_path": yolo_detector.model_path if yolo_detector is not None else None,
        }
        payload["perception"] = perception.health()
        payload["inference"] = inference.health()
        return payload

    @router.get("/rooms")
    def list_rooms() -> dict:
        return {"rooms": manager.list_rooms()}

    @router.post("/rooms")
    def register_room(payload: RoomCreate) -> dict:
        created = manager.register_room(payload.room_id)
        return {"room_id": payload.room_id, "created": created}

    @router.delete("/rooms/{room_id}")
    def delete_room(room_id: str) -> dict:
        removed = manager.remove_room(room_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Room not found")
        perception.remove_room(room_id)
        return {"room_id": room_id, "removed": True}

    @router.post("/rooms/{room_id}/cameras")
    def add_camera(room_id: str, payload: CameraCreate) -> dict:
        added = manager.add_camera(
            room_id, payload.camera_id, payload.url, payload.role
        )
        if not added:
            raise HTTPException(status_code=404, detail="Room not found or camera exists")
        perception.add_camera(room_id, payload.camera_id, payload.role)
        return {"room_id": room_id, "camera_id": payload.camera_id, "added": True}

    @router.delete("/rooms/{room_id}/cameras/{camera_id}")
    def delete_camera(room_id: str, camera_id: str) -> dict:
        removed = manager.remove_camera(room_id, camera_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Room or camera not found")
        perception.remove_camera(room_id, camera_id)
        return {"room_id": room_id, "camera_id": camera_id, "removed": True}

    @router.get("/rooms/{room_id}/health")
    def room_health(room_id: str) -> dict:
        status = manager.room_health(room_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Room not found")
        return status


    @router.get("/rooms/{room_id}/cameras/{camera_id}/snapshot")
    def snapshot(room_id: str, camera_id: str) -> Response:
        frame, _timestamp = manager.get_snapshot(room_id, camera_id)
        if frame is None:
            raise HTTPException(status_code=503, detail="No frame available")
        payload = _encode_jpeg(frame)
        if payload is None:
            raise HTTPException(status_code=500, detail="Failed to encode frame")
        return Response(content=payload, media_type="image/jpeg")


    @router.get("/perception/events")
    def perception_events(
        limit: int = 200,
        since: Optional[float] = None,
        room_id: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> dict:
        return {
            "events": perception.get_events(
                limit=limit, since=since, room_id=room_id, camera_id=camera_id
            )
        }

    @router.get("/inference/outputs")
    def inference_outputs(limit: int = 200, since: Optional[float] = None) -> dict:
        return {"outputs": inference.get_outputs(limit=limit, since=since)}

    @router.get("/inference/behavior")
    def inference_behavior_outputs(
        limit: int = 200,
        since: Optional[float] = None,
        room_id: Optional[str] = None,
    ) -> dict:
        raw_limit = max(200, min(5000, limit * 5))
        outputs = inference.get_outputs(limit=raw_limit, since=since)
        behavior_types = {
            "teacher_engagement",
            "teacher_device_usage",
            "teacher_student_interaction",
            "teacher_absence",
            "attention_summary",
            "participation_summary",
            "student_sleep_risk",
            "student_device_distraction",
            "student_behavior_summary",
            "offtask_movement",
            "group_participation_summary",
            "group_collaboration",
            "lesson_comprehensive_summary",
        }
        filtered = []
        for output in outputs:
            if output.get("type") not in behavior_types:
                continue
            if room_id is not None and output.get("room_id") != room_id:
                continue
            filtered.append(output)
        return {"outputs": filtered[-max(1, min(1000, limit)) :]}

    @router.get("/inference/lesson-summary")
    def inference_lesson_summary(
        limit: int = 20,
        since: Optional[float] = None,
        room_id: Optional[str] = None,
    ) -> dict:
        raw_limit = max(200, min(5000, limit * 20))
        outputs = inference.get_outputs(limit=raw_limit, since=since)
        summaries = []
        for output in outputs:
            if output.get("type") != "lesson_comprehensive_summary":
                continue
            if room_id is not None and output.get("room_id") != room_id:
                continue
            summaries.append(output)
        return {"summaries": summaries[-max(1, min(200, limit)) :]}

    @router.get("/attendance/today")
    def attendance_today() -> dict:
        date_key = time.strftime("%Y-%m-%d", time.localtime())
        return {"date": date_key, "attendance": attendance.get_attendance(date_key)}

    @router.get("/attendance/{date_key}")
    def attendance_by_date(date_key: str) -> dict:
        return {"date": date_key, "attendance": attendance.get_attendance(date_key)}

    @router.get("/dashboard/summary")
    def dashboard_summary(window_seconds: int = 300) -> dict:
        now = time.time()
        since = now - max(30, min(3600, window_seconds))
        outputs = inference.get_outputs(limit=500, since=since)
        output_counts: dict = {}
        for item in outputs:
            kind = item.get("type", "unknown")
            output_counts[kind] = output_counts.get(kind, 0) + 1

        date_key = time.strftime("%Y-%m-%d", time.localtime())
        return {
            "timestamp": now,
            "health": manager.health(),
            "yolo": {
                "enabled": yolo_detector is not None,
                "ready": yolo_detector.ready() if yolo_detector is not None else False,
                "model_path": yolo_detector.model_path if yolo_detector is not None else None,
            },
            "activity": perception.health(),
            "attendance": {
                "date": date_key,
                "count": len(attendance.get_attendance(date_key)),
            },
            "inference": {
                "window_seconds": int(max(30, min(3600, window_seconds))),
                "counts": output_counts,
                "recent": outputs[-20:],
            },
        }

    return router
