# Classroom AI – Part 2 Detailed Implementation (Motion & Activity Gate)

## Goal (Plain English)
Part 2 adds a lightweight motion gate that watches each camera feed and decides if there is meaningful activity. It does not run AI inference. It just says: "nothing happening" vs "some movement" vs "sudden spike." This lets us avoid running heavy AI unless it is actually needed.

## High-Level Flow
1. Each camera feed is opened and read continuously.
2. The latest frame is cached in memory with a timestamp.
3. A motion gate samples frames on a fixed schedule (per camera).
4. It compares the current frame to the previous sampled frame and computes a motion score (0.0–1.0).
5. A short rolling window smooths noise.
6. The camera state is updated to one of: IDLE, ACTIVE, SPIKE.
7. Health and activity are exposed via API endpoints.

## Components

### 1) StreamIngestor (`app/stream_ingestor.py`)
- Opens the camera stream and keeps reading frames in a background thread.
- Stores the most recent frame + timestamp in memory.
- Tracks simple metrics: running, frames received, restarts, last error.
- Uses backoff and retry when a stream fails.
- Supports optional FFmpeg capture for RTSP stability.

### 2) StreamManager (`app/stream_manager.py`)
- Manages rooms and cameras.
- Starts/stops ingestors when cameras are added/removed.
- Loads/persists room + camera metadata to JSON storage.
- Provides `get_snapshot()` to access the latest frame.
- Exposes health summaries per room and overall.

### 3) MotionGateManager (`app/motion_gate.py`)
- Maintains one motion gate per camera.
- Samples frames at a fixed interval (configurable).
- Computes a normalized motion score by comparing current vs previous frame.
- Uses a rolling average window to smooth the score.
- Applies thresholds and hysteresis to set camera state:
  - IDLE: no meaningful motion
  - ACTIVE: sustained motion above threshold
  - SPIKE: sudden large motion spike

### 4) API (`app/api.py`)
- Adds endpoints to read activity state and motion score:
  - `GET /activity`
  - `GET /rooms/{room_id}/activity`
  - `GET /rooms/{room_id}/cameras/{camera_id}/activity`

## Motion Score Details
- Frame is converted to grayscale and downsampled.
- Normalized by mean/std to reduce lighting effects.
- Score is the mean absolute difference between current and previous frame.
- The result is clamped to 0.0–1.0.
- Rolling window average gives smoother behavior.

## State Transitions (Simplified)
- **IDLE → ACTIVE** when score >= `GATE_ACTIVE_ENTER`
- **ACTIVE → IDLE** when score < `GATE_ACTIVE_EXIT`
- **Any → SPIKE** when score >= `GATE_SPIKE_ENTER`
- **SPIKE → ACTIVE/IDLE** after cooldown when score falls

This gives stability (hysteresis) and prevents jitter.

## Configuration (Environment Variables)
- `GATE_SAMPLE_INTERVAL_SECONDS` (default `1.0`)
- `GATE_WINDOW_SIZE` (default `5`)
- `GATE_ACTIVE_ENTER` (default `0.08`)
- `GATE_ACTIVE_EXIT` (default `0.04`)
- `GATE_SPIKE_ENTER` (default `0.35`)
- `GATE_SPIKE_EXIT` (default `0.2`)
- `GATE_SPIKE_COOLDOWN_SECONDS` (default `3.0`)
- `GATE_STALE_SECONDS` (default `2.5`)
- `GATE_DOWNSAMPLE_WIDTH` (default `64`)
- `GATE_DOWNSAMPLE_HEIGHT` (default `36`)

Stream settings:
- `FRAME_WIDTH`, `FRAME_HEIGHT`, `TARGET_FPS`
- `USE_FFMPEG` (auto/force/disable)
- `READ_TIMEOUT_SECONDS`
- `BACKOFF_MIN_SECONDS`, `BACKOFF_MAX_SECONDS`

## What We Tested
- Added room + cameras (front/back) with RTSP URLs.
- Verified health endpoint reports healthy and frame counts.
- Verified activity endpoint returns IDLE with low motion scores.

## What This Enables Next
- A reliable switch that can tell us when to run AI inference.
- Lower compute costs by avoiding AI when rooms are idle.
- A stable foundation for occupancy / student detection in Part 3.

## Known Limits (By Design)
- No AI inference yet.
- No storage of historical frames or activity logs.
- Motion only depends on video changes, not semantic understanding.

## Quick Smoke Test
1. Start API:
   `uvicorn app.main:app --reload`
2. Register a room:
   `POST /rooms {"room_id":"room1"}`
3. Add cameras:
   `POST /rooms/room1/cameras {"camera_id":"front","url":"rtsp://...","role":"front"}`
   `POST /rooms/room1/cameras {"camera_id":"back","url":"rtsp://...","role":"back"}`
4. Check activity:
   `GET /rooms/room1/activity`

