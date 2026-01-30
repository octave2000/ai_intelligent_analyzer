import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FaceMatch:
    bbox: Tuple[int, int, int, int]
    person_id: Optional[str]
    name: Optional[str]
    role: Optional[str]
    score: float
    embedding: np.ndarray


@dataclass
class RosterEntry:
    person_id: str
    name: str
    role: str
    embeddings: List[np.ndarray]


class FaceIdentifier:
    def __init__(
        self,
        roster_path: str,
        similarity_threshold: float,
        model_name: str,
        model_root: Optional[str],
        ctx_id: int = -1,
    ) -> None:
        self.roster_path = roster_path
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.model_root = model_root
        self.ctx_id = ctx_id
        self._app: Any = None
        self._roster: List[RosterEntry] = []
        self._ready = False

        self._load_models()
        self._load_roster()

    def ready(self) -> bool:
        return self._ready and bool(self._roster)

    def detect_and_identify(self, frame: "cv2.typing.MatLike") -> List[FaceMatch]:
        if not self.ready() or self._app is None:
            return []
        app = self._app
        faces = app.get(frame)
        matches: List[FaceMatch] = []
        for face in faces:
            bbox = _to_bbox(face.bbox)
            if bbox is None:
                continue
            embedding = face.embedding.astype(np.float32)
            person_id, name, role, score = self._match_embedding(embedding)
            matches.append(
                FaceMatch(
                    bbox=bbox,
                    person_id=person_id,
                    name=name,
                    role=role,
                    score=score,
                    embedding=embedding,
                )
            )
        return matches

    def _load_models(self) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore[import-not-found]
        except Exception:
            self._ready = False
            return

        try:
            model_root = self.model_root or ""
            app = FaceAnalysis(name=self.model_name, root=model_root)
            app.prepare(ctx_id=self.ctx_id)
            self._app = app
            self._ready = True
        except Exception:
            self._ready = False

    def _load_roster(self) -> None:
        if not os.path.exists(self.roster_path):
            return
        try:
            import json

            with open(self.roster_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return

        people = data.get("people") if isinstance(data, dict) else None
        if not isinstance(people, list):
            return
        roster: List[RosterEntry] = []
        for person in people:
            if not isinstance(person, dict):
                continue
            person_id = person.get("person_id")
            name = person.get("name")
            role = person.get("role", "unknown")
            images = person.get("images", [])
            embeddings_raw = person.get("embeddings", [])
            if not isinstance(person_id, str) or not isinstance(name, str):
                continue
            embeddings: List[np.ndarray] = []
            if isinstance(embeddings_raw, list):
                for emb in embeddings_raw:
                    if not isinstance(emb, list):
                        continue
                    arr = np.array(emb, dtype=np.float32)
                    if arr.size > 0:
                        embeddings.append(arr)
            if isinstance(images, list):
                for path in images:
                    if not isinstance(path, str):
                        continue
                    embedding = self._embed_image(path)
                    if embedding is not None:
                        embeddings.append(embedding)
            if embeddings:
                roster.append(
                    RosterEntry(
                        person_id=person_id,
                        name=name,
                        role=role if isinstance(role, str) else "unknown",
                        embeddings=embeddings,
                    )
                )
        self._roster = roster

    def _embed_image(self, path: str) -> Optional[np.ndarray]:
        if not self._ready or self._app is None:
            return None
        image = cv2.imread(path)
        if image is None:
            return None
        app = self._app
        faces = app.get(image)
        if not faces:
            return None
        embedding = faces[0].embedding.astype(np.float32)
        return embedding

    def _match_embedding(
        self, embedding: np.ndarray
    ) -> Tuple[Optional[str], Optional[str], Optional[str], float]:
        best_score = -1.0
        best_entry: Optional[RosterEntry] = None
        for entry in self._roster:
            for ref in entry.embeddings:
                score = _cosine_similarity(embedding, ref)
                if score > best_score:
                    best_score = score
                    best_entry = entry
        if best_entry is None or best_score < self.similarity_threshold:
            return None, None, None, best_score
        return best_entry.person_id, best_entry.name, best_entry.role, best_score


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _to_bbox(raw: object) -> Optional[Tuple[int, int, int, int]]:
    try:
        values = [int(v) for v in raw]  # type: ignore[assignment]
    except Exception:
        return None
    if len(values) != 4:
        return None
    return (values[0], values[1], values[2], values[3])
