import json
import os
import sys
from typing import Dict, List

import cv2
import numpy as np


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: python scripts/enroll_roster.py <input_roster.json> <output_roster.json>")
        return 1

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(input_path):
        print(f"Input roster not found: {input_path}")
        return 1

    try:
        from insightface.app import FaceAnalysis
    except Exception as exc:
        print(f"InsightFace not available: {exc}")
        return 1

    with open(input_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    people = data.get("people") if isinstance(data, dict) else None
    if not isinstance(people, list):
        print("Invalid roster format. Expected {\"people\": [...]} ")
        return 1

    model_name = os.getenv("FACE_MODEL_NAME", "buffalo_s")
    model_root = os.getenv("FACE_MODEL_ROOT") or None

    app = FaceAnalysis(name=model_name, root=model_root)
    app.prepare(ctx_id=-1)

    output_people: List[Dict[str, object]] = []
    for person in people:
        if not isinstance(person, dict):
            continue
        person_id = person.get("person_id")
        name = person.get("name")
        role = person.get("role", "unknown")
        images = person.get("images", [])
        if not isinstance(person_id, str) or not isinstance(name, str) or not isinstance(images, list):
            continue
        embeddings: List[List[float]] = []
        for image_path in images:
            if not isinstance(image_path, str):
                continue
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read {image_path}")
                continue
            faces = app.get(image)
            if not faces:
                print(f"No face detected in {image_path}")
                continue
            emb = faces[0].embedding.astype(np.float32)
            embeddings.append(emb.tolist())
        output_people.append(
            {
                "person_id": person_id,
                "name": name,
                "role": role if isinstance(role, str) else "unknown",
                "images": images,
                "embeddings": embeddings,
            }
        )

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({"people": output_people}, handle, indent=2, sort_keys=True)

    print(f"Wrote roster with embeddings to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
