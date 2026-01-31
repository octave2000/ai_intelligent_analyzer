#!/usr/bin/env python3
import argparse
import time
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.face_identifier import FaceIdentifier  # noqa: E402


def _build_identifier() -> FaceIdentifier:
    return FaceIdentifier(
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run face detection on a video and print recognized names."
    )
    parser.add_argument("video_path", help="Path to an MP4 (or other) video file")
    parser.add_argument("--every", type=int, default=3, help="Process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    args = parser.parse_args()

    identifier = _build_identifier()
    if not identifier.ready():
        print("FaceIdentifier is not ready. Check models and roster.json.")
        return

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {args.video_path}")
        return

    frame_idx = 0
    last_seen = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break
            if args.every > 1 and (frame_idx % args.every) != 0:
                continue

            matches = identifier.detect_and_identify(frame)
            now = time.time()
            for match in matches:
                if match.person_id is None or match.name is None:
                    continue
                last = last_seen.get(match.person_id, 0.0)
                if now - last < 0.5:
                    continue
                last_seen[match.person_id] = now
                print(
                    f"frame={frame_idx} person_id={match.person_id} "
                    f"name={match.name} score={match.score:.3f}"
                )
    finally:
        cap.release()


if __name__ == "__main__":
    main()
