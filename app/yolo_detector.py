from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class YoloDetection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class YoloDetector:
    def __init__(self, model_path: str, conf_threshold: float, iou_threshold: float) -> None:
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self._model = None
        self._names: List[str] = []
        self._ready = False
        self._load()

    def ready(self) -> bool:
        return self._ready

    def detect(self, frame: np.ndarray) -> List[YoloDetection]:
        if not self._ready:
            return []
        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        detections: List[YoloDetection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                label = self._names[cls_id] if 0 <= cls_id < len(self._names) else str(cls_id)
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                detections.append(YoloDetection(label=label, confidence=conf, bbox=(x1, y1, x2, y2)))
        return detections

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception:
            self._ready = False
            return
        try:
            model = YOLO(self.model_path)
            self._model = model
            self._names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
            self._ready = True
        except Exception:
            self._ready = False

