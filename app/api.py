from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field

from app.motion_gate import MotionGateManager
from app.perception_manager import PerceptionManager
from app.stream_manager import StreamManager


def _encode_jpeg(frame: "cv2.typing.MatLike") -> Optional[bytes]:
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return encoded.tobytes()


def build_router(
    manager: StreamManager, gate: MotionGateManager, perception: PerceptionManager
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
        return manager.health()

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
        gate.remove_room(room_id)
        perception.remove_room(room_id)
        return {"room_id": room_id, "removed": True}

    @router.post("/rooms/{room_id}/cameras")
    def add_camera(room_id: str, payload: CameraCreate) -> dict:
        added = manager.add_camera(
            room_id, payload.camera_id, payload.url, payload.role
        )
        if not added:
            raise HTTPException(status_code=404, detail="Room not found or camera exists")
        gate.add_camera(room_id, payload.camera_id, payload.role)
        perception.add_camera(room_id, payload.camera_id)
        return {"room_id": room_id, "camera_id": payload.camera_id, "added": True}

    @router.delete("/rooms/{room_id}/cameras/{camera_id}")
    def delete_camera(room_id: str, camera_id: str) -> dict:
        removed = manager.remove_camera(room_id, camera_id)
        if not removed:
            raise HTTPException(status_code=404, detail="Room or camera not found")
        gate.remove_camera(room_id, camera_id)
        perception.remove_camera(room_id, camera_id)
        return {"room_id": room_id, "camera_id": camera_id, "removed": True}

    @router.get("/rooms/{room_id}/health")
    def room_health(room_id: str) -> dict:
        status = manager.room_health(room_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Room not found")
        return status

    @router.get("/rooms/{room_id}/activity")
    def room_activity(room_id: str) -> dict:
        status = gate.room_activity(room_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Room not found")
        return status

    @router.get("/rooms/{room_id}/cameras/{camera_id}/activity")
    def camera_activity(room_id: str, camera_id: str) -> dict:
        status = gate.activity(room_id, camera_id)
        if status is None:
            raise HTTPException(status_code=404, detail="Room or camera not found")
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

    @router.get("/activity")
    def all_activity() -> dict:
        return gate.all_activity()

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

    return router
