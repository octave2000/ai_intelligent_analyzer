import json
import os
import sys
import time
from collections import defaultdict
from statistics import mean, median
from typing import Dict, List

import cv2

from app.face_identifier import FaceIdentifier
from app.yolo_detector import YoloDetector


def _load_rooms(path: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0}
    return {"count": len(values), "mean": mean(values), "median": median(values)}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/tune_thresholds.py <seconds>")
        return 1
    try:
        seconds = float(sys.argv[1])
    except ValueError:
        print("Seconds must be a number")
        return 1

    rooms_path = os.getenv("STORAGE_PATH", "data/rooms.json")
    rooms = _load_rooms(rooms_path)
    if not rooms:
        print("No rooms/cameras found. Add cameras first.")
        return 1

    face_identifier = FaceIdentifier(
        roster_path=os.getenv("ROSTER_PATH", "data/roster.json"),
        similarity_threshold=float(os.getenv("FACE_SIMILARITY_THRESHOLD", "0.35")),
        model_name=os.getenv("FACE_MODEL_NAME", "buffalo_s"),
        model_root=os.getenv("FACE_MODEL_ROOT") or None,
    )

    yolo_mode = os.getenv("USE_YOLO", "auto").strip().lower()
    yolo_detector = None
    if yolo_mode != "disable":
        yolo_detector = YoloDetector(
            model_path=os.getenv("YOLO_MODEL_PATH", "yolov8n.pt"),
            conf_threshold=float(os.getenv("YOLO_CONF_THRESHOLD", "0.35")),
            iou_threshold=float(os.getenv("YOLO_IOU_THRESHOLD", "0.45")),
        )

    face_scores: List[float] = []
    yolo_scores: Dict[str, List[float]] = defaultdict(list)

    end_time = time.time() + seconds
    while time.time() < end_time:
        for room_id, cams in rooms.items():
            for cam_id, meta in cams.items():
                url = meta.get("url")
                if not isinstance(url, str):
                    continue
                cap = cv2.VideoCapture(url)
                ok, frame = cap.read()
                cap.release()
                if not ok or frame is None:
                    continue

                if face_identifier.ready():
                    faces = face_identifier.detect_and_identify(frame)
                    for face in faces:
                        if face.score > 0:
                            face_scores.append(face.score)

                if yolo_detector and yolo_detector.ready():
                    detections = yolo_detector.detect(frame)
                    for det in detections:
                        yolo_scores[det.label].append(det.confidence)

        time.sleep(0.5)

    print("Face similarity stats:", _stats(face_scores))
    if face_scores:
        print("Suggested FACE_SIMILARITY_THRESHOLD ~", round(max(0.2, median(face_scores) * 0.9), 3))

    if yolo_scores:
        print("YOLO confidence stats by label:")
        for label, scores in yolo_scores.items():
            stats = _stats(scores)
            print(f"  {label}: {stats}")
        print("Suggested YOLO_CONF_THRESHOLD ~", round(0.25, 3))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
