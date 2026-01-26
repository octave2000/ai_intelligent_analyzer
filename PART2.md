# Classroom AI – Part 2 (Motion & Activity Gate)

## Scope
Part 2 adds a deterministic motion gate that decides whether downstream AI should run based on visual activity only.
It does not perform AI inference, detection, tracking, or storage.

## How it Works
- Samples frames at a fixed, configurable interval per camera.
- Computes a normalized motion score from the current frame vs. the immediately previous sampled frame.
- Aggregates scores over a short rolling window to reduce noise.
- Emits one of three states per camera: `IDLE`, `ACTIVE`, `SPIKE`, with hysteresis and cooldown.

## Key Rules Implemented
- Per‑camera isolation: each camera has its own sampling and state.
- Deterministic, CPU‑bounded processing.
- No frame history beyond the previous frame + rolling window.
- If frames are missing or stale, emits `IDLE` without blocking or retrying.

## Configuration
Environment variables:
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

## API Additions
- `GET /activity` → activity for all rooms/cameras
- `GET /rooms/{room_id}/activity` → activity for one room
- `GET /rooms/{room_id}/cameras/{camera_id}/activity` → activity for one camera

Each activity response includes:
- `camera_id`
- `camera_role`
- `activity_state` (`IDLE` | `ACTIVE` | `SPIKE`)
- `motion_score` (0.0–1.0)
- `timestamp`
- `seconds_since_state_change`
