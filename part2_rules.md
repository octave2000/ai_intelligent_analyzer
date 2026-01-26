# Classroom AI – Part 2: Motion & Activity Gate
## Definitive Rules, Constraints, and Architecture Specification

This document defines **Part 2** of the Classroom AI system.
It is the **first analytical layer**, responsible for **activity gating**.
It does not perform AI inference or behavior analysis.

This document is authoritative.  
Any implementation MUST follow these rules.

---

## 1. Purpose and Responsibility

### 1.1 Core Responsibility
Part 2 answers exactly one question:

> “Is there enough visual activity right now to justify running downstream AI?”

Nothing more. Nothing less.

### 1.2 What Part 2 Enables
- CPU-first operation
- Event-driven AI execution
- Suppression of unnecessary compute
- Deterministic system behavior

### 1.3 What Part 2 Never Does
Part 2 must NEVER:
- Identify people
- Detect objects
- Track individuals
- Classify behavior
- Store or forward images
- Trigger alerts or decisions
- Persist long-term state

---

## 2. Position in the System Architecture

### 2.1 Layering Rule
Part 2 sits strictly between:

- **Part 1 (Video Ingestion)**
- **Part 3+ (AI Inference Layers)**

It consumes **frames** from Part 1  
It emits **signals** to Part 3+

There must be **no backward dependencies**.

---

## 3. Inputs

### 3.1 Frame Source
- Frames must come exclusively from Part 1
- Direct camera access is forbidden
- Frames are considered best-effort

### 3.2 Frame Freshness
- If a frame is stale, missing, or invalid:
  - The system must not block
  - The system must not retry
  - The system must emit `IDLE`

### 3.3 Frame Characteristics
Frames may vary in:
- Lighting
- Compression artifacts
- Minor jitter
- Noise

The gate must tolerate all of the above.

---

## 4. Frame Sampling Rules

### 4.1 Temporal Downsampling
- Frames must NOT be processed continuously
- A fixed sampling interval must be enforced
- Sampling rate must be configurable

Purpose:
- Bound CPU usage
- Ensure predictable load

### 4.2 Sampling Independence
- Sampling decisions must be independent per camera
- One camera’s activity must not affect another’s sampling

---

## 5. Motion Definition

### 5.1 What Motion Means
Motion is defined as:
> “A measurable difference between two sampled frames.”

Motion does NOT imply:
- Human movement
- Meaningful activity
- Suspicious behavior

It is a purely **visual delta**.

---

## 6. Motion Measurement Constraints

### 6.1 Comparison Scope
- Motion must be computed only between:
  - Current sampled frame
  - Immediately previous sampled frame

No multi-second or multi-minute history is allowed.

### 6.2 Spatial Scope
- Motion may be computed on:
  - Full frame
  - Reduced resolution
  - Reduced color space

But:
- The method must be consistent
- The output must be normalized

---

## 7. Motion Score

### 7.1 Scalar Requirement
Each evaluation produces a single scalar:

motion_score ∈ [0.0, 1.0]:

### 7.2 Normalization Rules
- Motion score must be independent of:
  - Resolution
  - Aspect ratio
  - Absolute brightness

### 7.3 Determinism
- Same frame pair → same motion score
- No randomness
- No learned parameters

---

## 8. Temporal Aggregation

### 8.1 Rolling Window
- Motion scores must be aggregated over a short window
- Window size must be:
  - Small
  - Fixed
  - Configurable

### 8.2 Purpose
Aggregation exists to:
- Reduce noise
- Avoid single-frame spikes
- Stabilize state transitions

---

## 9. Activity States

### 9.1 Allowed States
Exactly three states are allowed:

- `IDLE`
- `ACTIVE`
- `SPIKE`

No additional states are permitted.

---

## 10. State Semantics

### 10.1 IDLE
- Sustained low motion
- Scene is visually stable
- Downstream AI must not run

### 10.2 ACTIVE
- Sustained moderate motion
- Indicates ongoing activity
- Downstream AI may run at low frequency

### 10.3 SPIKE
- Sudden or unusually large motion
- Indicates abrupt change
- Downstream AI may run in burst mode

---

## 11. State Transition Rules

### 11.1 Hysteresis
- State transitions must include hysteresis
- Entry and exit thresholds must differ
- Rapid oscillation is forbidden

### 11.2 Cooldown
- After entering `SPIKE`, a cooldown period is mandatory
- Prevents repeated triggers from a single event

---

## 12. Per-Camera Isolation

### 12.1 Independence
- Each camera maintains its own:
  - Motion scores
  - Rolling window
  - Activity state

### 12.2 Failure Isolation
- Failure in one camera’s gate must not affect others

---

## 13. Camera Role Awareness

### 13.1 Role Metadata
Each camera has a role:
- `front`
- `back`
- `other`

### 13.2 Role Usage
- Motion computation is role-agnostic
- Role is included in outputs for downstream interpretation
- No role-specific logic inside Part 2

---

## 14. Outputs

### 14.1 Output Contract
For each camera, Part 2 must emit:

- `camera_id`
- `camera_role`
- `activity_state`
- `motion_score`
- `timestamp`

### 14.2 Output Frequency
- Outputs update whenever:
  - State changes
  - Motion score updates

---

## 15. Interaction With Downstream AI

### 15.1 Scheduling Contract
Downstream AI modules must obey:

- `IDLE` → AI forbidden
- `ACTIVE` → AI allowed (low rate)
- `SPIKE` → AI allowed (burst)

### 15.2 Authority Rule
Part 2 is authoritative.
Downstream AI must not override it.

---

## 16. Performance Constraints

### 16.1 CPU Budget
- Must complete faster than frame sampling interval
- Must never block:
  - Ingestion
  - API responses

### 16.2 Memory Budget
- Memory usage must be constant
- No unbounded buffers
- No frame history beyond rolling window

---

## 17. Failure & Degradation Handling

### 17.1 Missing Data
If frames are missing or invalid:
- Emit `IDLE`
- Do not throw errors
- Do not escalate

### 17.2 Silent Failure Rule
Part 2 must fail silently and safely.
The system must remain operational.

---

## 18. Observability & Debugging

### 18.1 Required Metrics
Expose:
- Current activity state per camera
- Last motion score
- Time since last state change

### 18.2 Explainability
Every state change must be explainable via:
- Motion score
- Aggregation window
- Thresholds

---

## 19. Security & Privacy

### 19.1 Data Minimization
- No images may leave Part 2
- No personal data inferred or stored

### 19.2 Auditability
- State transitions must be inspectable
- No hidden logic

---

## 20. Explicit Non-Goals

Part 2 must NEVER include:
- Object detection
- Pose estimation
- Face recognition
- Action recognition
- Alerts
- Notifications
- Decisions
- Storage

---

## 21. Completion Criteria

Part 2 is complete when:
- CPU usage remains stable over time
- AI compute is suppressed during idle scenes
- State transitions are predictable
- No memory growth occurs
- System remains debuggable

Only after this may Part 3 begin.

---

## 22. Governing Principle

**Compute is a resource.  
Silence is the default.  
AI must be earned by activity.**

The Motion & Activity Gate enforces this principle.

---
