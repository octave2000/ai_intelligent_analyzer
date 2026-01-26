# Classroom AI – Part 3 Status (What We Have Achieved)

## Summary (Plain English)
Part 3 is now implemented as a **gated perception layer**. It only runs when Part 2 says there is activity. It detects people, tracks them per camera, creates temporary global IDs across cameras, detects common objects, and emits neutral event facts (no conclusions).

## What Part 3 Does Today
- **Obeys the motion gate (Part 2)**
  - If a camera is `IDLE`, Part 3 does nothing.
  - If `ACTIVE`, Part 3 runs at a slow interval.
  - If `SPIKE`, Part 3 runs in short faster bursts.

- **Person detection and short‑term tracking (per camera)**
  - Uses a lightweight CPU detector (OpenCV HOG).
  - Tracks people across nearby frames with per‑camera track IDs.
  - Emits events: `person_detected`, `person_tracked`, `person_lost`.

- **Session‑only global IDs**
  - Per‑camera tracks can map to a temporary global person ID.
  - These IDs reset when the service restarts.
  - Fusion is conservative to avoid false merges.

- **Role classification (teacher vs student)**
  - Primary signal: uniform color range (configurable).
  - Secondary heuristic: taller/standing presence (conservative).
  - Emits `role_assigned` with confidence, otherwise stays `unknown`.

- **Object detection (lightweight, heuristic‑based)**
  - Detects objects in required categories:
    - Devices: phone, laptop, tablet
    - Academic: paper, notebook
    - Suspicious: knife‑like, concealed paper (exam mode)
    - Lab: beaker, test tube
    - Personal: backpack, pouch
  - Emits `object_detected` with category, risk level, and confidence.

- **Object–person association**
  - Associates objects to nearby persons (overlap/proximity only).
  - Emits `object_associated` (no intent statements).

- **Head orientation events (coarse)**
  - Emits `head_orientation_changed` (forward / left / right / down).

- **Proximity & group primitives**
  - Emits `proximity_event` when people are close for a sustained time.
  - Emits `group_formed` and `group_updated` for groups over time.

## What Part 3 Does NOT Do (By Design)
- No AI conclusions or labels like “cheating” or “threatening.”
- No face recognition or real‑world identity.
- No long‑term storage of frames or behavioral history.
- No heavy GPU inference (YOLO not used).

## How We View the Output
- **Gate status**: `GET /activity`
- **Perception events**: `GET /perception/events`

Each event includes:
- `timestamp`
- `room_id`
- `camera_id`
- `event_type`
- `confidence`
- (optional) `global_person_id`

## Where This Leaves Us
We now have a **full, working perception pipeline** that runs only when there is motion. It produces factual, low‑level signals that can later be interpreted by higher‑level logic or human review.

