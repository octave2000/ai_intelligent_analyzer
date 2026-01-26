# Classroom AI – Part 1: Video Ingestion & System Architecture Rules

This document defines the **mandatory architecture, rules, and constraints** for the Classroom AI system.
It is intentionally **code-agnostic**. Implementation is left to the codex / engineering layer.

---

## 1. System Scope (Part 1)

Part 1 is responsible ONLY for:
- Reliable ingestion of two classroom camera streams (front + back)
- Low-latency frame access
- Stream health monitoring
- Snapshot delivery for downstream AI modules

Explicitly out of scope:
- AI inference
- Motion detection
- Object detection
- Pose estimation
- Tracking
- Storage of historical video

No AI logic is allowed in Part 1.

---

## 2. High-Level Architecture

### 2.1 Components
- **Stream Ingestor (per camera)**
  - Responsible for connecting to the camera stream
  - Decoding frames
  - Enforcing resolution and FPS caps
  - Maintaining the latest valid frame only

- **Stream Manager**
  - Owns exactly two ingestors:
    - `front_camera`
    - `back_camera`
  - Starts, stops, and monitors both streams
  - Exposes stream health metrics

- **API Layer**
  - Read-only access to:
    - Latest snapshot per camera
    - Stream health metrics
  - No business logic
  - No video processing beyond JPEG encoding

---

## 3. Camera Rules

### 3.1 Camera Count
- Exactly two cameras per classroom:
  - Front camera: teacher + board
  - Back camera: students

### 3.2 Stream Protocol
- Streams must be consumed as **live sources**
- Buffered playback is forbidden
- Latency must be minimized even at the cost of dropped frames

### 3.3 Frame Ownership
- The system must retain **only the most recent valid frame**
- Historical frames must not be queued or cached

---

## 4. Performance Constraints

### 4.1 Resolution
- All frames must be normalized to a fixed resolution
- Default target: low-to-mid resolution (e.g., 360p or equivalent)
- Resolution must be configurable at startup

### 4.2 FPS
- Ingest FPS must be capped
- Base ingest FPS should be low and stable
- FPS must be configurable at startup

### 4.3 CPU Priority
- Ingest must be CPU-safe
- Ingest must never block API requests
- No per-frame heavy computation is allowed

---

## 5. Reliability Rules

### 5.1 Auto-Reconnect
- If a stream disconnects, stalls, or fails:
  - The system must automatically attempt reconnection
  - Reconnection must use a backoff strategy
  - Failures must not crash the application

### 5.2 Fault Isolation
- Failure of one camera must not affect the other
- Each stream must be isolated in execution

### 5.3 Blocking Prevention
- Stream reading must never block:
  - API endpoints
  - Other streams
  - Shutdown signals

---

## 6. Frame Buffer Rules

### 6.1 Buffer Type
- Single-frame buffer only
- No queues
- No ring buffers

### 6.2 Atomic Access
- Reads and writes to the frame buffer must be atomic
- Partial frames must never be exposed

### 6.3 Timestamping
- Each frame must carry:
  - A capture timestamp
- Timestamp source must be system time

---

## 7. Health & Observability

### 7.1 Required Metrics (Per Camera)
- `is_running` (boolean)
- `last_frame_timestamp`
- `frame_age_seconds`
- `frames_received_total`
- `restart_count`
- `last_error_message`

### 7.2 Health Semantics
- A stream is considered **healthy** if:
  - `frame_age_seconds` is below a small threshold
- A stream is considered **degraded** if:
  - Frames are stale but reconnect attempts continue
- A stream is considered **down** if:
  - No frames and reconnects are failing

---

## 8. API Contract Rules

### 8.1 Snapshot Endpoints
- One snapshot endpoint per camera
- Always returns the **latest frame**
- Must not block waiting for a new frame

### 8.2 Health Endpoint
- Returns:
  - Overall system status
  - Per-camera health metrics
- Must respond even if streams are down

### 8.3 Statelessness
- API layer must remain stateless
- All state lives in the stream manager

---

## 9. Memory & Storage Rules

### 9.1 Memory
- Memory usage must be constant over time
- No unbounded growth
- Frame buffers must be overwritten in-place

### 9.2 Storage
- No disk storage of video or frames in Part 1
- No caching beyond in-memory latest frame

---

## 10. Shutdown & Lifecycle Rules

### 10.1 Startup
- Streams must start automatically on service startup
- Failure to connect initially must not crash the service

### 10.2 Shutdown
- Streams must terminate cleanly on shutdown
- External processes must be killed safely
- No hanging threads/processes

---

## 11. Security & Privacy Baseline

### 11.1 Data Minimization
- Only transient frame data allowed
- No identity inference
- No persistence

### 11.2 Access Control (Future-Proofing)
- Architecture must allow future auth without refactor
- No security assumptions in stream layer

---
