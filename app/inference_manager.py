import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from app.perception_manager import PerceptionManager

logger = logging.getLogger(__name__)


@dataclass
class PersonSignalWindow:
    events: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=500))
    last_cheating_output: float = 0.0
    last_teacher_output: float = 0.0
    last_participation_output: float = 0.0
    last_fight_output: float = 0.0
    last_interaction_output: float = 0.0
    last_absence_output: float = 0.0
    last_paper_output: float = 0.0


@dataclass
class GroupSignalWindow:
    events: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=200))
    last_output: float = 0.0


class InferenceManager:
    def __init__(
        self,
        perception: PerceptionManager,
        exam_mode: bool,
        cheating_window_seconds: float,
        cheating_emit_interval_seconds: float,
        teacher_window_seconds: float,
        teacher_emit_interval_seconds: float,
        participation_window_seconds: float,
        participation_emit_interval_seconds: float,
        sync_turn_window_seconds: float,
        fight_window_seconds: float,
        fight_emit_interval_seconds: float,
        fight_motion_threshold: float,
        fight_proximity_threshold: int,
    ) -> None:
        self.perception = perception
        self.exam_mode = exam_mode
        self.cheating_window_seconds = cheating_window_seconds
        self.cheating_emit_interval_seconds = cheating_emit_interval_seconds
        self.teacher_window_seconds = teacher_window_seconds
        self.teacher_emit_interval_seconds = teacher_emit_interval_seconds
        self.participation_window_seconds = participation_window_seconds
        self.participation_emit_interval_seconds = participation_emit_interval_seconds
        self.sync_turn_window_seconds = sync_turn_window_seconds
        self.fight_window_seconds = fight_window_seconds
        self.fight_emit_interval_seconds = fight_emit_interval_seconds
        self.fight_motion_threshold = fight_motion_threshold
        self.fight_proximity_threshold = fight_proximity_threshold

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_ts: float = 0.0
        self._people: Dict[int, PersonSignalWindow] = {}
        self._roles: Dict[int, str] = {}
        self._last_seen: Dict[int, float] = {}
        self._outputs: Deque[Dict[str, object]] = deque(maxlen=2000)
        self._group_windows: Dict[int, GroupSignalWindow] = {}
        self._lock = threading.Lock()
        self._teacher_absence_threshold_seconds = 30.0

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_outputs(self, limit: int = 200, since: Optional[float] = None) -> List[Dict[str, object]]:
        limit = max(1, min(1000, limit))
        results: List[Dict[str, object]] = []
        for output in reversed(self._outputs):
            if since is not None and _get_float(output, "timestamp") <= since:
                continue
            results.append(output)
            if len(results) >= limit:
                break
        results.reverse()
        return results

    def health(self) -> Dict[str, object]:
        return {
            "output_queue_size": len(self._outputs),
            "people_windows": len(self._people),
            "group_windows": len(self._group_windows),
        }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            events = self.perception.get_events(since=self._last_ts)
            if events:
                latest = max(_get_float(e, "timestamp") for e in events)
                self._last_ts = max(self._last_ts, latest)
                for event in events:
                    self._consume_event(event)
            self._evaluate()
            time.sleep(0.5)

    def _consume_event(self, event: Dict[str, object]) -> None:
        global_id = _get_int(event, "global_person_id")
        event_type = event.get("event_type")
        logger.debug(
            "inference.consume_event event_type=%s global_person_id=%s",
            event_type,
            global_id,
        )
        if global_id is not None and event_type in ("person_detected", "person_tracked"):
            self._last_seen[global_id] = _get_float(event, "timestamp") or time.time()
        if event_type == "role_assigned" and global_id is not None:
            role = event.get("role")
            if isinstance(role, str):
                self._roles[global_id] = role
        if global_id is not None:
            window = self._people.setdefault(global_id, PersonSignalWindow())
            window.events.append(event)

        if event_type in ("group_formed", "group_updated"):
            group_id = _get_int(event, "group_id")
            if group_id is not None:
                window = self._group_windows.setdefault(group_id, GroupSignalWindow())
                window.events.append(event)

    def _evaluate(self) -> None:
        now = time.time()
        for global_id, window in list(self._people.items()):
            role = self._roles.get(global_id, "unknown")
            if self.exam_mode and role == "student":
                self._maybe_emit_cheating(global_id, window, now)
                self._maybe_emit_paper_interaction(global_id, window, now)
            if role == "teacher":
                self._maybe_emit_teacher(global_id, window, now)
                self._maybe_emit_teacher_interaction(global_id, window, now)
                self._maybe_emit_teacher_absence(global_id, window, now)
            if role == "student":
                self._maybe_emit_participation(global_id, window, now)
            self._maybe_emit_fight(global_id, window, now)

        for group_id, window in list(self._group_windows.items()):
            self._maybe_emit_group_participation(group_id, window, now)

    def _maybe_emit_cheating(self, global_id: int, window: PersonSignalWindow, now: float) -> None:
        if now - window.last_cheating_output < self.cheating_emit_interval_seconds:
            return
        recent = _filter_recent(window.events, now - self.cheating_window_seconds)
        signals: List[str] = []
        score = 0.0

        concealed = _count_object(recent, "concealed_paper")
        if concealed > 0:
            signals.append("concealed_paper_detected")
            score += 0.35

        phone_assoc = _count_object_association(recent, {"phone", "laptop", "tablet"})
        if phone_assoc > 0:
            signals.append("device_associated")
            score += 0.3

        head_turns = _count_head_turns(recent)
        if head_turns >= 3:
            signals.append("repeated_head_turns")
            score += 0.2

        head_down = _count_head_down(recent)
        if head_down >= 3:
            signals.append("prolonged_head_down")
            score += 0.2

        proximity = _count_proximity_close(recent)
        if proximity > 0:
            signals.append("proximity_to_peer")
            score += 0.15

        sync_turns = _count_sync_turns(recent, self.sync_turn_window_seconds)
        if sync_turns > 0:
            signals.append("synchronized_head_turns")
            score += 0.2

        score = min(1.0, score)
        if score < 0.4:
            return

        confidence = min(1.0, score + 0.1)
        output = {
            "timestamp": now,
            "type": "cheating_suspicion",
            "student_global_id": global_id,
            "score": score,
            "signals": signals,
            "confidence": confidence,
            "time_window": self.cheating_window_seconds,
        }
        window.last_cheating_output = now
        self._outputs.append(output)
        logger.info(
            "inference.output type=%s global_person_id=%s score=%.3f",
            output.get("type"),
            global_id,
            score,
        )

    def _maybe_emit_teacher(self, global_id: int, window: PersonSignalWindow, now: float) -> None:
        if now - window.last_teacher_output < self.teacher_emit_interval_seconds:
            return
        recent = _filter_recent(window.events, now - self.teacher_window_seconds)
        if not recent:
            return

        movement = _movement_distance(recent)
        group_interactions = _count_group_membership(recent, global_id)
        device_assoc = _count_object_association(recent, {"phone", "laptop", "tablet"})
        down_events = _count_head_down(recent)

        movement_score = min(1.0, movement / 200.0)
        interaction_score = min(1.0, group_interactions / 3.0)
        device_penalty = min(0.4, device_assoc * 0.15)
        down_penalty = min(0.3, down_events * 0.05)

        engagement = max(0.0, min(1.0, 0.5 * movement_score + 0.5 * interaction_score - device_penalty - down_penalty))

        output = {
            "timestamp": now,
            "type": "teacher_engagement",
            "teacher_global_id": global_id,
            "engagement_score": engagement,
            "activity_breakdown": {
                "movement_score": movement_score,
                "interaction_score": interaction_score,
                "device_association_events": device_assoc,
                "head_down_events": down_events,
            },
            "confidence": min(1.0, 0.4 + 0.6 * (movement_score + interaction_score) / 2.0),
        }
        window.last_teacher_output = now
        self._outputs.append(output)
        logger.info(
            "inference.output type=%s teacher_global_id=%s score=%.3f",
            output.get("type"),
            global_id,
            engagement,
        )

    def _maybe_emit_participation(self, global_id: int, window: PersonSignalWindow, now: float) -> None:
        if now - window.last_participation_output < self.participation_emit_interval_seconds:
            return
        recent = _filter_recent(window.events, now - self.participation_window_seconds)
        if not recent:
            return

        group_interactions = _count_group_membership(recent, global_id)
        proximity = _count_proximity_close(recent)
        movement = _movement_distance(recent)

        score = 0.0
        score += min(0.5, group_interactions * 0.2)
        score += min(0.3, proximity * 0.15)
        score += min(0.2, movement / 250.0)
        score = min(1.0, score)

        if score < 0.25:
            level = "low"
        elif score < 0.6:
            level = "medium"
        else:
            level = "high"

        output = {
            "timestamp": now,
            "type": "participation_summary",
            "student_global_id": global_id,
            "participation_level": level,
            "confidence": min(1.0, 0.3 + score),
        }
        window.last_participation_output = now
        self._outputs.append(output)
        logger.info(
            "inference.output type=%s student_global_id=%s level=%s",
            output.get("type"),
            global_id,
            level,
        )

    def _maybe_emit_teacher_interaction(self, global_id: int, window: PersonSignalWindow, now: float) -> None:
        if now - window.last_interaction_output < self.teacher_emit_interval_seconds:
            return
        recent = _filter_recent(window.events, now - self.teacher_window_seconds)
        if not recent:
            return
        student_ids = set()
        best_duration = 0.0
        for e in recent:
            if e.get("event_type") != "proximity_event" or e.get("status") != "close":
                continue
            gids = e.get("global_ids")
            if not isinstance(gids, list) or global_id not in gids:
                continue
            for gid in gids:
                if gid == global_id:
                    continue
                if self._roles.get(gid) == "student":
                    student_ids.add(gid)
            best_duration = max(best_duration, _get_float(e, "duration_seconds"))
        if not student_ids:
            return
        confidence = min(1.0, 0.4 + min(0.6, best_duration / 10.0))
        output = {
            "timestamp": now,
            "type": "teacher_student_interaction",
            "teacher_global_id": global_id,
            "student_global_ids": sorted(student_ids),
            "duration_seconds": best_duration,
            "confidence": confidence,
            "time_window": self.teacher_window_seconds,
        }
        window.last_interaction_output = now
        self._outputs.append(output)
        logger.info(
            "inference.output type=%s teacher_global_id=%s students=%d",
            output.get("type"),
            global_id,
            len(student_ids),
        )

    def _maybe_emit_teacher_absence(self, global_id: int, window: PersonSignalWindow, now: float) -> None:
        if now - window.last_absence_output < self.teacher_emit_interval_seconds:
            return
        last_seen = self._last_seen.get(global_id)
        if last_seen is None:
            return
        absence = now - last_seen
        if absence < self._teacher_absence_threshold_seconds:
            return
        output = {
            "timestamp": now,
            "type": "teacher_absence",
            "teacher_global_id": global_id,
            "absence_seconds": absence,
            "confidence": min(1.0, 0.4 + min(0.6, absence / 120.0)),
            "time_window": self.teacher_window_seconds,
        }
        window.last_absence_output = now
        self._outputs.append(output)
        logger.info(
            "inference.output type=%s teacher_global_id=%s absence=%.1fs",
            output.get("type"),
            global_id,
            absence,
        )

    def _maybe_emit_paper_interaction(self, global_id: int, window: PersonSignalWindow, now: float) -> None:
        if now - window.last_paper_output < self.cheating_emit_interval_seconds:
            return
        recent = _filter_recent(window.events, now - self.cheating_window_seconds)
        if not recent:
            return
        has_paper = any(
            e.get("event_type") == "object_associated"
            and e.get("object_type") in ("paper", "notebook", "book", "concealed_paper")
            for e in recent
        )
        if not has_paper:
            return
        related = _extract_related_ids(recent, global_id)
        if not related:
            return
        output = {
            "timestamp": now,
            "type": "paper_interaction",
            "student_global_id": global_id,
            "related_global_ids": related,
            "confidence": 0.5,
            "time_window": self.cheating_window_seconds,
        }
        window.last_paper_output = now
        self._outputs.append(output)
        logger.info(
            "inference.output type=%s student_global_id=%s related=%d",
            output.get("type"),
            global_id,
            len(related),
        )

    def _maybe_emit_fight(self, global_id: int, window: PersonSignalWindow, now: float) -> None:
        if now - window.last_fight_output < self.fight_emit_interval_seconds:
            return
        recent = _filter_recent(window.events, now - self.fight_window_seconds)
        if not recent:
            return
        movement = _movement_distance(recent)
        proximity = _count_proximity_close(recent)
        if movement < self.fight_motion_threshold or proximity < self.fight_proximity_threshold:
            return
        involved = _extract_related_ids(recent, global_id)
        signals = ["rapid_motion", "close_proximity"]
        confidence = min(1.0, 0.4 + (movement / (self.fight_motion_threshold * 2.0)) * 0.3 + proximity * 0.1)
        output = {
            "timestamp": now,
            "type": "safety_suspicion",
            "category": "fight",
            "global_person_id": global_id,
            "related_global_ids": involved,
            "signals": signals,
            "confidence": min(1.0, confidence),
            "time_window": self.fight_window_seconds,
        }
        window.last_fight_output = now
        self._outputs.append(output)
        logger.info(
            "inference.output type=%s global_person_id=%s confidence=%.3f",
            output.get("type"),
            global_id,
            output.get("confidence"),
        )

    def _maybe_emit_group_participation(self, group_id: int, window: GroupSignalWindow, now: float) -> None:
        if now - window.last_output < self.participation_emit_interval_seconds:
            return
        recent = _filter_recent(window.events, now - self.participation_window_seconds)
        if not recent:
            return
        duration = _group_duration(recent)
        motion_intensity = _group_motion_intensity(recent)
        members = _group_members(recent)
        student_members = [gid for gid in members if self._roles.get(gid) == "student"]
        teacher_members = [gid for gid in members if self._roles.get(gid) == "teacher"]

        if duration < 5.0:
            level = "low"
        elif duration < 20.0:
            level = "medium"
        else:
            level = "high"

        output = {
            "timestamp": now,
            "type": "participation_summary",
            "group_id": group_id,
            "participation_level": level,
            "confidence": min(1.0, 0.3 + min(1.0, duration / 30.0) + motion_intensity * 0.3),
        }
        window.last_output = now
        self._outputs.append(output)
        logger.info(
            "inference.output type=%s group_id=%s level=%s",
            output.get("type"),
            group_id,
            level,
        )

        if len(student_members) >= 2 and not teacher_members:
            collab_output = {
                "timestamp": now,
                "type": "group_collaboration",
                "group_id": group_id,
                "student_global_ids": sorted(student_members),
                "duration_seconds": duration,
                "confidence": min(1.0, 0.4 + min(0.6, duration / 30.0) + motion_intensity * 0.2),
            }
            self._outputs.append(collab_output)
            logger.info(
                "inference.output type=%s group_id=%s students=%d",
                collab_output.get("type"),
                group_id,
                len(student_members),
            )

            student_output = {
                "timestamp": now,
                "type": "student_group_interaction",
                "group_id": group_id,
                "student_global_ids": sorted(student_members),
                "duration_seconds": duration,
                "confidence": min(1.0, 0.3 + min(0.6, duration / 30.0)),
            }
            self._outputs.append(student_output)
            logger.info(
                "inference.output type=%s group_id=%s students=%d",
                student_output.get("type"),
                group_id,
                len(student_members),
            )


def _filter_recent(events: Deque[Dict[str, object]], since: float) -> List[Dict[str, object]]:
    return [e for e in events if _get_float(e, "timestamp") >= since]


def _count_object(events: List[Dict[str, object]], object_type: str) -> int:
    return sum(
        1
        for e in events
        if e.get("event_type") == "object_detected" and e.get("object_type") == object_type
    )


def _count_object_association(events: List[Dict[str, object]], types: set) -> int:
    return sum(
        1
        for e in events
        if e.get("event_type") == "object_associated" and e.get("object_type") in types
    )


def _count_head_turns(events: List[Dict[str, object]]) -> int:
    return sum(
        1
        for e in events
        if e.get("event_type") == "head_orientation_changed"
        and e.get("orientation") in ("left", "right")
    )


def _count_head_down(events: List[Dict[str, object]]) -> int:
    return sum(
        1
        for e in events
        if e.get("event_type") == "head_orientation_changed"
        and e.get("orientation") == "down"
    )


def _count_proximity_close(events: List[Dict[str, object]]) -> int:
    return sum(
        1
        for e in events
        if e.get("event_type") == "proximity_event"
        and e.get("status") == "close"
        and _get_float(e, "duration_seconds") >= 2.0
    )


def _count_sync_turns(events: List[Dict[str, object]], window_seconds: float) -> int:
    head_events = [
        e
        for e in events
        if e.get("event_type") == "head_orientation_changed"
        and e.get("orientation") in ("left", "right")
    ]
    head_events.sort(key=lambda e: _get_float(e, "timestamp"))
    count = 0
    for i, ev in enumerate(head_events):
        for other in head_events[i + 1 :]:
            if _get_float(other, "timestamp") - _get_float(ev, "timestamp") > window_seconds:
                break
            if ev.get("orientation") == other.get("orientation"):
                count += 1
                break
    return count


def _movement_distance(events: List[Dict[str, object]]) -> float:
    positions: List[Tuple[float, float]] = []
    for e in events:
        if e.get("event_type") in ("person_detected", "person_tracked"):
            bbox = e.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                x1 = _to_float(bbox[0])
                y1 = _to_float(bbox[1])
                x2 = _to_float(bbox[2])
                y2 = _to_float(bbox[3])
                positions.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    distance = 0.0
    for i in range(1, len(positions)):
        x1, y1 = positions[i - 1]
        x2, y2 = positions[i]
        distance += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance


def _count_group_membership(events: List[Dict[str, object]], global_id: int) -> int:
    count = 0
    for e in events:
        if e.get("event_type") in ("group_formed", "group_updated"):
            members = e.get("members")
            if isinstance(members, list) and global_id in members:
                count += 1
    return count


def _group_duration(events: List[Dict[str, object]]) -> float:
    durations = [
        float(_get_float(e, "duration_seconds"))
        for e in events
        if e.get("event_type") in ("group_formed", "group_updated")
    ]
    return max(durations) if durations else 0.0


def _group_members(events: List[Dict[str, object]]) -> List[int]:
    for e in reversed(events):
        if e.get("event_type") in ("group_formed", "group_updated"):
            members = e.get("members")
            if isinstance(members, list):
                return [m for m in members if isinstance(m, int)]
    return []


def _group_motion_intensity(events: List[Dict[str, object]]) -> float:
    counts = len([e for e in events if e.get("event_type") == "group_updated"])
    return min(1.0, counts / 5.0)


def _extract_related_ids(events: List[Dict[str, object]], global_id: int) -> List[int]:
    related = set()
    for e in events:
        if e.get("event_type") != "proximity_event":
            continue
        gids = e.get("global_ids")
        if not isinstance(gids, list):
            continue
        if global_id in gids:
            for gid in gids:
                if isinstance(gid, int) and gid != global_id:
                    related.add(gid)
    return sorted(related)


def _get_float(event: Dict[str, object], key: str) -> float:
    value = event.get(key, 0.0)
    return _to_float(value)


def _get_int(event: Dict[str, object], key: str) -> Optional[int]:
    value = event.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0
