
Part 3 runs **only when Part 2 allows it**.

---

## 4. Execution Authority & Gating

### 4.1 Gating Rules
Part 3 MUST obey Part 2:

- `IDLE`  → perception forbidden
- `ACTIVE` → perception allowed at low frequency
- `SPIKE` → perception allowed in short bursts

Part 3 must never self-activate.

---

## 5. Input Constraints

### 5.1 Frame Source
- Frames come only from Part 1
- Only latest frames are used
- Stale frames must be rejected

### 5.2 Camera Assumptions
- Any person may appear in any camera
- No camera is teacher-only or student-only
- Camera placement must not affect logic

---

## 6. Person Detection & Tracking

### 6.1 Person Detection
- Detect all visible persons
- No hard limit on person count
- Detection must be lightweight and CPU-compatible

### 6.2 Per-Camera Tracking
- Assign per-camera track IDs
- Track IDs are short-lived
- Track IDs reset on service restart

Tracking exists only for temporal continuity.

---

## 7. Global Person Identity (Session-Only)

### 7.1 Global Person ID
- Per-camera tracks may map to a single global person ID
- Global IDs are valid only within the session
- Global IDs are NOT real-world identities

### 7.2 Fusion Rules
- Fusion must be conservative
- False merges are worse than missed merges
- Fusion uses appearance and motion consistency
- Fusion runs only when Part 2 allows

---

## 8. Role Classification (Teacher vs Student)

### 8.1 Allowed Roles
- `student`
- `teacher`
- `unknown`

### 8.2 Primary Signal: Uniform Detection
- Students wear uniforms
- Teachers do not
- Uniform presence strongly implies `student`

### 8.3 Secondary Signals (Heuristic)
Used only when uniform is absent:
- interaction with many students
- classroom traversal patterns
- board-facing posture
- prolonged standing vs sitting

### 8.4 Role Confidence
- Role assignment must include confidence
- Low confidence → role remains `unknown`
- Role may change over time as evidence accumulates

---

## 9. Object Detection (Generalized & Extensible)

### 9.1 Object Categories
Part 3 must detect objects in these categories:

- **Devices**: phone, laptop, tablet
- **Academic**: book, paper, notebook, calculator
- **Suspicious**: knife-like, sharp-object-like, concealed paper
- **Lab**: beaker, test tube, burner (configurable)
- **Personal**: backpack, pouch

### 9.2 Object Taxonomy
Each detected object must include:
- object type
- category
- risk_level (low / medium / high)
- confidence

---

## 10. Object–Person Association

### 10.1 Association Rules
Objects may be associated with a person if:
- spatial overlap exists
- proximity persists across frames
- object stays near hands or torso

### 10.2 Association Semantics
Association is probabilistic.
Part 3 must never say:
- “using”
- “cheating”
- “threatening”

Only: “associated”.

---

## 11. Posture & Orientation Primitives

### 11.1 Head Orientation (Coarse)
Emit coarse orientation only:
- forward
- left
- right
- down

No gaze estimation required.

### 11.2 Orientation Events
Emit:
- orientation change
- duration
- repetition count

---

## 12. Proximity & Interaction Primitives

### 12.1 Proximity
Emit:
- distance between persons
- duration of closeness
- sudden distance changes

### 12.2 Interaction Attempts
Emit primitives such as:
- leaning toward another person
- simultaneous head turns
- hand movement toward neighbor

No interpretation is allowed.

---

## 13. Cheating-Enabling Primitives (Exam Context)

### 13.1 Exam Mode Flag
Part 3 may receive:
- `exam_mode = true | false`

### 13.2 Hidden / Small Paper
Attempt to detect:
- unusually small paper
- folded or concealed paper
- paper appearing/disappearing near hands

Emit:
- object_detected: concealed_paper
- confidence
- associated person (if any)

### 13.3 Repeated Head Turns
Emit:
- repeated directional head turns
- short temporal windows
- no intent labels

### 13.4 Coordinated Motion
Emit:
- synchronized movements between two persons
- repeated proximity + orientation alignment

---

## 14. Group Formation Primitives

### 14.1 Group Definition
A group is:
- two or more persons
- within close proximity
- sustained over time

### 14.2 Group Outputs
Emit:
- group_id
- member global IDs
- group duration
- group motion intensity

---

## 15. Output Philosophy

Part 3 outputs **facts only**.

No conclusions.
No scoring.
No policy labels.

---

## 16. Output Contract

### 16.1 Required Fields
Each perception event must include:
- `timestamp`
- `camera_id`
- `global_person_id` (if applicable)
- `event_type`
- `confidence`

### 16.2 Allowed Event Types
- `person_detected`
- `person_tracked`
- `person_lost`
- `role_assigned`
- `object_detected`
- `object_associated`
- `head_orientation_changed`
- `proximity_event`
- `group_formed`
- `group_updated`

No other event types are allowed in Part 3.

---

## 17. Performance Constraints

### 17.1 CPU Safety
- Perception must respect time budgets
- No blocking operations
- No continuous inference

### 17.2 Memory Safety
- Constant memory usage
- Rolling windows only
- No frame history storage

---

## 18. Failure Handling

### 18.1 Detection Failure
- Emit no events
- Do not retry aggressively
- Respect cooldowns

### 18.2 Partial Visibility
- Partial detections are acceptable
- System stability takes priority over completeness

---

## 19. Privacy & Ethics

### 19.1 Privacy Baseline
- Face recognition is optional and low-weight
- Faces must never be sole identity proof
- No biometric identity persistence

### 19.2 Ethical Constraint
- Cheating and misconduct are never declared here
- All outputs require downstream interpretation

---

## 20. Completion Criteria (Part 3)

Part 3 is complete when:
- People and objects are detected reliably
- Global IDs are stable within a session
- Roles are assigned conservatively
- Cheating primitives are emitted correctly
- CPU usage remains bounded
- No identity persists across sessions

---

## 21. Governing Principle

**Observe first.  
Label nothing.  
Decide later.**

Part 3 provides evidence — not judgment.

---
