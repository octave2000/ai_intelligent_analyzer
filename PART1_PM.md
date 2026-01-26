# Classroom AI – Part 1 Summary (Non‑Technical)

This phase delivers the video “plumbing” for classrooms. It does not do any AI analysis. It only connects to cameras, keeps the most recent image, and exposes simple health and snapshot endpoints.

## What we can do now
- **Register rooms and cameras**: We can create a room and add any number of cameras to it.
- **Fetch a current snapshot** from any camera on demand.
- **Check camera health** to see if a camera is healthy, degraded, or down.
- **Auto‑recover** if a stream disconnects (retries with backoff).
- **Survive restarts**: room/camera registrations are saved to a file and restored on startup.

## What is intentionally NOT included
- No AI or analytics (no detection, tracking, or recognition).
- No video recording or long‑term storage.
- No frame history (only the latest image is kept in memory).

## Why this is useful
This is the foundation for future AI features. It guarantees that live camera data is available quickly and reliably, while keeping memory usage constant and the system stable.

## How it works (high level)
- Each camera stream runs independently so one failure does not affect others.
- Only the latest frame is kept, so snapshots are always fast and memory stays flat.
- Health endpoints show if each camera is working and how fresh the last frame is.
- A room can be added or removed at any time via the API.

## Current API capabilities (plain language)
- **Create a room**
- **Add a camera to a room**
- **Remove a room**
- **Remove a camera**
- **List rooms and cameras**
- **Get a snapshot for a specific camera**
- **Check health for a room or the entire system**

## Example success criteria met
- Multiple rooms with multiple cameras can be handled at once.
- Snapshots return quickly without waiting for new frames.
- Streams reconnect automatically if they drop.

If you want, we can add authentication, dashboards, or analytics in the next phase.
