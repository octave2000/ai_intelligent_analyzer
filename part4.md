
Part 4 runs only when **Part 3 emits events**.

---

## 4. Inputs

### 4.1 Required Inputs (from Part 3)
- perception events (ordered by timestamp)
- global_person_id (session-only)
- role (`student`, `teacher`, `unknown`)
- object detections and associations
- proximity events
- group events
- head orientation events
- confidence values

### 4.2 Contextual Flags
- `exam_mode = true | false`
- class metadata (optional)

---

## 5. General Inference Principles

### 5.1 Evidence Accumulation
- No single event is sufficient
- Behaviors are inferred from **patterns over time**

### 5.2 Confidence-Based Output
- Every inferred behavior must include:
  - confidence score
  - contributing signals

### 5.3 Temporal Scope
- Inference windows are bounded
- Evidence expires over time

---

## 6. Exam Cheating Inference (Students Only)

### 6.1 Preconditions
Cheating inference runs ONLY when:
- `exam_mode = true`
- role = `student`

---

### 6.2 Cheating Signals (Inputs)

The following Part 3 primitives may contribute:

- concealed / small paper detection
- phone or device association
- repeated head turns (left/right)
- prolonged head-down posture
- proximity to another student
- synchronized head turns between two students
- object appearing/disappearing near hands

No single signal implies cheating.

---

### 6.3 Cheating Pattern Examples

Examples of patterns (not rules):

- repeated head turns toward the same neighbor
- concealed paper + head down for extended duration
- phone association during exam mode
- synchronized movements between two students

---

### 6.4 Cheating Output

Part 4 emits **cheating suspicion**, not a verdict.

Output must include:
- `student_global_id`
- `suspicion_score` (0.0–1.0)
- contributing signals
- time window

Language must be neutral:
- “possible cheating indicators detected”

---

## 7. Teacher Behavior & Teaching Quality

### 7.1 Preconditions
Teacher inference runs only when:
- role = `teacher`

---

### 7.2 Teacher Presence

Signals:
- teacher global ID present
- duration of presence
- frequency of absence

Outputs:
- presence timeline
- presence confidence

---

### 7.3 Teacher Engagement Signals

Signals may include:
- movement across classroom
- interaction with student groups
- facing students vs disengaged posture
- standing vs prolonged inactivity
- device (phone/laptop) association

---

### 7.4 Teaching Quality Indicators

Teaching quality is **never binary**.

Indicators may include:
- time interacting with students
- time stationary without interaction
- balance of movement vs stillness
- device usage during teaching time

Output:
- engagement score
- activity breakdown
- confidence

No “good/bad teacher” labels allowed.

---

## 8. Student Participation & Group Dynamics

### 8.1 Individual Participation

Signals:
- presence in group interactions
- proximity to peers
- movement during collaborative periods
- orientation toward group members

Outputs:
- participation indicators (low / medium / high)
- confidence
- session-local only

---

### 8.2 Group Participation

Group analysis is based on:
- group formation events
- group duration
- group motion intensity
- group membership changes

Outputs:
- group participation timelines
- dominant participants
- passive participants (probabilistic)

---

## 9. Interaction Interpretation Rules

### 9.1 Neutral Interpretation
All outputs must remain neutral and descriptive.

Forbidden language:
- “cheated”
- “lazy”
- “bad teaching”
- “fighting”

Allowed language:
- “suspicion score”
- “engagement indicator”
- “participation level”

---

## 10. Output Contracts

### 10.1 Cheating Output
- `type: cheating_suspicion`
- `student_global_id`
- `score`
- `signals`
- `confidence`
- `time_window`

---

### 10.2 Teacher Output
- `type: teacher_engagement`
- `teacher_global_id`
- `engagement_score`
- `activity_breakdown`
- `confidence`

---

### 10.3 Participation Output
- `type: participation_summary`
- `student_global_id` or `group_id`
- `participation_level`
- `confidence`

---

## 11. Performance Constraints

### 11.1 CPU Safety
- No heavy computation
- No frame processing
- Only event aggregation

### 11.2 Memory Safety
- Bounded windows
- No unbounded history

---

## 12. Failure Handling

### 12.1 Missing Data
- Missing perception events degrade confidence
- No hard failures

### 12.2 Partial Evidence
- Partial evidence produces low confidence
- System must not over-assert

---

## 13. Privacy & Ethics

### 13.1 Privacy Guarantees
- No face recognition
- No student naming required
- No biometric persistence

### 13.2 Human-in-the-Loop
All outputs are designed for:
- dashboards
- reports
- teacher/admin review

No automated enforcement allowed.

---

## 14. Completion Criteria (Part 4)

Part 4 is complete when:
- cheating suspicion is explainable and bounded
- teacher engagement is measurable but neutral
- student participation is observable
- confidence accompanies all outputs
- system remains CPU-stable

---

## 15. Governing Principle

**Observe → Aggregate → Explain → Let Humans Decide**

Part 4 interprets.
Humans remain accountable.

---
