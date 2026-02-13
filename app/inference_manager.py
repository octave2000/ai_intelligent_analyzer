import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from app.perception_manager import PerceptionManager
from app.schedule_manager import ScheduleManager

logger = logging.getLogger(__name__)


@dataclass
class PersonSignalWindow:
    events: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=800))
    last_cheating_output: float = 0.0
    last_teacher_output: float = 0.0
    last_participation_output: float = 0.0
    last_fight_output: float = 0.0
    last_interaction_output: float = 0.0
    last_absence_output: float = 0.0
    last_paper_output: float = 0.0
    last_attention_output: float = 0.0
    last_offtask_output: float = 0.0
    last_sleep_output: float = 0.0
    last_student_device_output: float = 0.0
    last_teacher_device_output: float = 0.0
    last_behavior_output: float = 0.0


@dataclass
class GroupSignalWindow:
    events: Deque[Dict[str, object]] = field(default_factory=lambda: deque(maxlen=300))
    room_id: Optional[str] = None
    last_output: float = 0.0


class InferenceManager:
    def __init__(
        self,
        perception: PerceptionManager,
        schedule: ScheduleManager,
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
        teacher_absence_threshold_seconds: float,
        behavior_room_events_maxlen: int,
        behavior_lesson_window_seconds: float,
        behavior_lesson_emit_interval_seconds: float,
        behavior_sleep_emit_interval_seconds: float,
        behavior_device_emit_interval_seconds: float,
        behavior_emit_interval_seconds: float,
        behavior_cheating_min_score: float,
        behavior_offtask_min_movement: float,
        behavior_sleep_min_duration_seconds: float,
        behavior_sleep_min_head_down_events: int,
        behavior_sleep_min_bowing_events: int,
        behavior_sleep_min_risk_score: float,
        behavior_lesson_min_event_count: int,
        behavior_lesson_teacher_focus_strength_threshold: float,
        behavior_lesson_participation_strength_threshold: float,
        behavior_lesson_teacher_activity_strength_threshold: float,
        behavior_lesson_student_phone_concern_threshold: int,
        behavior_lesson_teacher_phone_concern_threshold: int,
        behavior_lesson_sleep_concern_threshold: int,
        behavior_level_low_max: float,
        behavior_level_medium_max: float,
    ) -> None:
        self.perception = perception
        self.schedule = schedule
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
        self._teacher_absence_threshold_seconds = max(
            1.0, teacher_absence_threshold_seconds
        )
        self._person_ids: Dict[int, str] = {}
        self._person_roles: Dict[str, str] = {}
        self._person_rooms: Dict[int, Tuple[str, str]] = {}

        self._room_events_maxlen = max(500, behavior_room_events_maxlen)
        self._room_events: Dict[str, Deque[Dict[str, object]]] = {}
        self._room_last_lesson_output: Dict[str, float] = {}
        self._lesson_window_seconds = max(30.0, behavior_lesson_window_seconds)
        self._lesson_emit_interval_seconds = max(
            5.0, behavior_lesson_emit_interval_seconds
        )
        self._sleep_emit_interval_seconds = max(0.0, behavior_sleep_emit_interval_seconds)
        self._device_emit_interval_seconds = max(
            0.0, behavior_device_emit_interval_seconds
        )
        if behavior_emit_interval_seconds > 0.0:
            self._behavior_emit_interval_seconds = behavior_emit_interval_seconds
        else:
            self._behavior_emit_interval_seconds = max(
                8.0, self.participation_emit_interval_seconds
            )
        self._cheating_min_score = _clamp01(behavior_cheating_min_score)
        self._offtask_min_movement = max(0.0, behavior_offtask_min_movement)
        self._sleep_min_duration_seconds = max(
            0.0, behavior_sleep_min_duration_seconds
        )
        self._sleep_min_head_down_events = max(0, behavior_sleep_min_head_down_events)
        self._sleep_min_bowing_events = max(0, behavior_sleep_min_bowing_events)
        self._sleep_min_risk_score = _clamp01(behavior_sleep_min_risk_score)
        self._lesson_min_event_count = max(1, behavior_lesson_min_event_count)
        self._lesson_teacher_focus_strength_threshold = _clamp01(
            behavior_lesson_teacher_focus_strength_threshold
        )
        self._lesson_participation_strength_threshold = _clamp01(
            behavior_lesson_participation_strength_threshold
        )
        self._lesson_teacher_activity_strength_threshold = _clamp01(
            behavior_lesson_teacher_activity_strength_threshold
        )
        self._lesson_student_phone_concern_threshold = max(
            1, behavior_lesson_student_phone_concern_threshold
        )
        self._lesson_teacher_phone_concern_threshold = max(
            1, behavior_lesson_teacher_phone_concern_threshold
        )
        self._lesson_sleep_concern_threshold = max(
            1, behavior_lesson_sleep_concern_threshold
        )
        low_max = _clamp01(behavior_level_low_max)
        medium_max = _clamp01(behavior_level_medium_max)
        if medium_max <= low_max:
            medium_max = min(1.0, low_max + 0.01)
        self._level_low_max = low_max
        self._level_medium_max = medium_max

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

    def get_outputs(
        self, limit: int = 200, since: Optional[float] = None
    ) -> List[Dict[str, object]]:
        limit = max(1, min(1000, limit))
        results: List[Dict[str, object]] = []
        with self._lock:
            outputs = list(self._outputs)
        for output in reversed(outputs):
            if since is not None and _output_cursor_ts(output) <= since:
                continue
            results.append(output)
            if len(results) >= limit:
                break
        results.reverse()
        return results

    def health(self) -> Dict[str, object]:
        with self._lock:
            output_count = len(self._outputs)
        return {
            "output_queue_size": output_count,
            "people_windows": len(self._people),
            "group_windows": len(self._group_windows),
            "rooms_tracked": len(self._room_events),
        }

    def _append_output(self, output: Dict[str, object], emitted_at: float) -> None:
        if "emitted_at" not in output:
            output["emitted_at"] = emitted_at
        ts = _get_float(output, "timestamp")
        if ts <= 0.0:
            output["timestamp"] = emitted_at
            ts = emitted_at
        output["lag_seconds"] = max(0.0, emitted_at - ts)
        with self._lock:
            self._outputs.append(output)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            events = self.perception.get_events(since=self._last_ts)
            if events:
                latest = max(_event_cursor_ts(e) for e in events)
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
        self._update_person_identity(event, global_id)

        room_id = event.get("room_id")
        if isinstance(room_id, str):
            room_window = self._room_events.setdefault(
                room_id, deque(maxlen=self._room_events_maxlen)
            )
            room_window.append(event)

        if global_id is not None:
            if event_type in ("person_detected", "person_tracked"):
                self._last_seen[global_id] = _event_cursor_ts(event) or time.time()
            camera_id = event.get("camera_id")
            if isinstance(room_id, str) and isinstance(camera_id, str):
                self._person_rooms[global_id] = (room_id, camera_id)
            role = _event_role(event, self._roles, global_id)
            if role in ("teacher", "student"):
                self._roles[global_id] = role
            window = self._people.setdefault(global_id, PersonSignalWindow())
            window.events.append(event)

        if event_type in ("group_formed", "group_updated"):
            group_id = _get_int(event, "group_id")
            if group_id is not None:
                window = self._group_windows.setdefault(group_id, GroupSignalWindow())
                if isinstance(room_id, str):
                    window.room_id = room_id
                window.events.append(event)

    def _update_person_identity(
        self, event: Dict[str, object], global_id: Optional[int]
    ) -> None:
        person_id = event.get("person_id")
        if isinstance(person_id, str) and global_id is not None:
            self._person_ids[global_id] = person_id

        role = event.get("person_role")
        if not isinstance(role, str) and event.get("event_type") == "role_assigned":
            role = event.get("role")
        if isinstance(role, str) and role in ("teacher", "student"):
            if global_id is not None:
                self._roles[global_id] = role
            if isinstance(person_id, str):
                self._person_roles[person_id] = role

        if event.get("event_type") == "identity_resolved":
            previous_id = event.get("previous_person_id")
            if (
                isinstance(previous_id, str)
                and isinstance(person_id, str)
                and isinstance(role, str)
            ):
                self._person_roles[person_id] = role

    def _person_id_for_global(self, global_id: int) -> str:
        return self._person_ids.get(global_id, f"unknown:{global_id}")

    def _attach_location(self, output: Dict[str, object], global_id: int) -> None:
        room = self._person_rooms.get(global_id)
        if room is None:
            return
        output["room_id"] = room[0]
        output["camera_id"] = room[1]

    @staticmethod
    def _should_emit(last_emit: float, now: float, min_interval_seconds: float) -> bool:
        if min_interval_seconds <= 0.0:
            return True
        return (now - last_emit) >= min_interval_seconds

    def _level_from_score(self, score: float) -> str:
        if score < self._level_low_max:
            return "low"
        if score < self._level_medium_max:
            return "medium"
        return "high"

    def _evaluate(self) -> None:
        now = time.time()
        for global_id, window in list(self._people.items()):
            role = self._roles.get(global_id, "unknown")
            if role == "teacher":
                self._maybe_emit_teacher(global_id, window, now)
                self._maybe_emit_teacher_device_usage(global_id, window, now)
                self._maybe_emit_teacher_interaction(global_id, window, now)
                self._maybe_emit_teacher_absence(global_id, window, now)
            if role == "student":
                self._maybe_emit_participation(global_id, window, now)
                self._maybe_emit_attention(global_id, window, now)
                self._maybe_emit_offtask(global_id, window, now)
                self._maybe_emit_sleeping(global_id, window, now)
                self._maybe_emit_student_device_distraction(global_id, window, now)
                self._maybe_emit_student_behavior_summary(global_id, window, now)
                if self.exam_mode:
                    self._maybe_emit_cheating(global_id, window, now)
                    self._maybe_emit_paper_interaction(global_id, window, now)
            self._maybe_emit_fight(global_id, window, now)

        for group_id, window in list(self._group_windows.items()):
            self._maybe_emit_group_participation(group_id, window, now)

        for room_id in list(self._room_events.keys()):
            self._maybe_emit_lesson_summary(room_id, now)

    def _maybe_emit_cheating(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_cheating_output, now, self.cheating_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.cheating_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)
        signals: List[str] = []
        score = 0.0

        concealed = _count_object(recent, "concealed_paper")
        if concealed > 0:
            signals.append("concealed_paper_detected")
            score += min(0.35, concealed * 0.15)

        phone_assoc = _count_object_association(
            recent, {"phone", "laptop", "tablet", "device"}
        )
        if phone_assoc > 0:
            signals.append("device_associated")
            score += min(0.35, phone_assoc * 0.08)

        head_turns = _count_head_turns(recent)
        if head_turns >= 3:
            signals.append("repeated_head_turns")
            score += min(0.2, head_turns * 0.03)

        head_down = _count_head_down(recent)
        if head_down >= 3:
            signals.append("prolonged_head_down")
            score += min(0.2, head_down * 0.025)

        proximity = _count_proximity_close(recent)
        if proximity > 0:
            signals.append("proximity_to_peer")
            score += min(0.2, proximity * 0.06)

        sync_turns = _count_sync_turns(recent, self.sync_turn_window_seconds)
        if sync_turns > 0:
            signals.append("synchronized_head_turns")
            score += min(0.2, sync_turns * 0.06)

        sleep_events = _count_event_types(recent, {"sleeping_suspected"})
        if sleep_events > 0:
            signals.append("fatigue_behavior")
            score += min(0.2, sleep_events * 0.08)

        score = _clamp01(score)
        if score < self._cheating_min_score:
            return

        output = {
            "timestamp": event_ts,
            "type": "cheating_suspicion",
            "student_person_id": self._person_id_for_global(global_id),
            "student_global_id": global_id,
            "score": score,
            "signals": signals,
            "confidence": min(1.0, score + 0.12),
            "time_window": self.cheating_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_cheating_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s global_person_id=%s score=%.3f",
            output.get("type"),
            global_id,
            score,
        )

    def _maybe_emit_teacher(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_teacher_output, now, self.teacher_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.teacher_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)

        movement = _movement_distance(recent)
        group_interactions = _count_group_membership(
            recent, self._person_id_for_global(global_id)
        )
        device_assoc = _count_object_association(
            recent, {"phone", "laptop", "tablet", "device"}
        )
        teacher_phone = _count_event_types(recent, {"teacher_phone_usage"})
        down_events = _count_head_down(recent)
        sleep_events = _count_event_types(recent, {"sleeping_suspected"})

        room_id, _camera_id = self._person_rooms.get(global_id, (None, None))
        student_focus_ratio = 0.0
        if isinstance(room_id, str):
            room_events = self._room_events.get(room_id, deque())
            room_recent = _filter_recent(room_events, now - self.teacher_window_seconds)
            student_focus_ratio = _room_teacher_focus_ratio(room_recent, self._roles)

        movement_score = min(1.0, movement / 220.0)
        interaction_score = min(1.0, group_interactions / 4.0)
        posture_score = max(
            0.0, 1.0 - min(1.0, (down_events + (sleep_events * 2)) / 8.0)
        )
        base_engagement = _clamp01(
            (0.45 * movement_score) + (0.35 * interaction_score) + (0.2 * posture_score)
        )
        penalty = min(0.6, (device_assoc * 0.08) + (teacher_phone * 0.18))
        engagement = _clamp01((0.75 * base_engagement) + (0.25 * student_focus_ratio) - penalty)

        output = {
            "timestamp": event_ts,
            "type": "teacher_engagement",
            "teacher_person_id": self._person_id_for_global(global_id),
            "teacher_global_id": global_id,
            "engagement_score": engagement,
            "activity_breakdown": {
                "movement_score": movement_score,
                "interaction_score": interaction_score,
                "posture_score": posture_score,
                "student_focus_ratio": student_focus_ratio,
                "device_association_events": device_assoc,
                "teacher_phone_events": teacher_phone,
            },
            "confidence": min(1.0, 0.4 + (0.6 * engagement)),
            "time_window": self.teacher_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_teacher_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s teacher_global_id=%s score=%.3f",
            output.get("type"),
            global_id,
            engagement,
        )

    def _maybe_emit_participation(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_participation_output,
            now,
            self.participation_emit_interval_seconds,
        ):
            return
        recent = _filter_recent(window.events, now - self.participation_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)

        group_interactions = _count_group_membership(
            recent, self._person_id_for_global(global_id)
        )
        proximity = _count_proximity_close(recent)
        movement = _movement_distance(recent)
        teacher_ratio, notebook_ratio, _unknown_ratio = _focus_ratios(recent)

        score = 0.0
        score += min(0.4, group_interactions * 0.15)
        score += min(0.25, proximity * 0.08)
        score += min(0.2, movement / 280.0)
        score += min(0.15, (teacher_ratio + notebook_ratio) * 0.2)
        score = _clamp01(score)

        output = {
            "timestamp": event_ts,
            "type": "participation_summary",
            "student_person_id": self._person_id_for_global(global_id),
            "student_global_id": global_id,
            "participation_level": self._level_from_score(score),
            "participation_score": score,
            "confidence": min(1.0, 0.35 + score),
            "teacher_present": self._teacher_present(global_id, event_ts),
            "time_window": self.participation_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_participation_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s student_global_id=%s level=%s",
            output.get("type"),
            global_id,
            output.get("participation_level"),
        )

    def _maybe_emit_teacher_interaction(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_interaction_output, now, self.teacher_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.teacher_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)
        student_ids = set()
        student_global_ids = set()
        best_duration = 0.0
        for event in recent:
            if event.get("event_type") != "proximity_event" or event.get("status") != "close":
                continue
            gids = event.get("global_ids")
            if not isinstance(gids, list) or global_id not in gids:
                continue
            for gid in gids:
                if not isinstance(gid, int) or gid == global_id:
                    continue
                if self._roles.get(gid) == "student":
                    student_ids.add(self._person_id_for_global(gid))
                    student_global_ids.add(gid)
            best_duration = max(best_duration, _get_float(event, "duration_seconds"))
        if not student_ids:
            return
        confidence = min(1.0, 0.45 + min(0.55, best_duration / 12.0))
        output = {
            "timestamp": event_ts,
            "type": "teacher_student_interaction",
            "teacher_person_id": self._person_id_for_global(global_id),
            "teacher_global_id": global_id,
            "student_person_ids": sorted(student_ids),
            "student_global_ids": sorted(student_global_ids),
            "duration_seconds": best_duration,
            "confidence": confidence,
            "time_window": self.teacher_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_interaction_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s teacher_global_id=%s students=%d",
            output.get("type"),
            global_id,
            len(student_ids),
        )

    def _maybe_emit_teacher_absence(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_absence_output, now, self.teacher_emit_interval_seconds
        ):
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
            "teacher_person_id": self._person_id_for_global(global_id),
            "teacher_global_id": global_id,
            "absence_seconds": absence,
            "confidence": min(1.0, 0.45 + min(0.55, absence / 120.0)),
            "time_window": self.teacher_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_absence_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s teacher_global_id=%s absence=%.1fs",
            output.get("type"),
            global_id,
            absence,
        )

    def _maybe_emit_paper_interaction(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_paper_output, now, self.cheating_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.cheating_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)
        related = _extract_related_person_ids(recent, self._person_id_for_global(global_id))
        concealed = _count_object(recent, "concealed_paper")
        if concealed <= 0 and not related:
            return
        output = {
            "timestamp": event_ts,
            "type": "paper_interaction",
            "student_person_id": self._person_id_for_global(global_id),
            "student_global_id": global_id,
            "related_person_ids": related,
            "related_global_ids": _extract_related_ids(recent, global_id),
            "concealed_paper_count": concealed,
            "confidence": min(1.0, 0.45 + min(0.4, concealed * 0.1)),
            "time_window": self.cheating_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_paper_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s student_global_id=%s related=%d concealed=%d",
            output.get("type"),
            global_id,
            len(related),
            concealed,
        )

    def _maybe_emit_fight(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_fight_output, now, self.fight_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.fight_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)
        movement = _movement_distance(recent)
        proximity = _count_proximity_close(recent)
        if (
            movement < self.fight_motion_threshold
            and proximity < self.fight_proximity_threshold
        ):
            return
        involved = _extract_related_person_ids(recent, self._person_id_for_global(global_id))
        signals: List[str] = []
        if movement >= self.fight_motion_threshold:
            signals.append("rapid_motion")
        if proximity >= self.fight_proximity_threshold:
            signals.append("close_proximity")
        confidence = _clamp01(
            0.4
            + min(0.4, movement / max(1.0, self.fight_motion_threshold * 2.0))
            + min(0.3, proximity * 0.08)
        )
        output = {
            "timestamp": event_ts,
            "type": "safety_suspicion",
            "category": "fight",
            "person_id": self._person_id_for_global(global_id),
            "global_person_id": global_id,
            "related_person_ids": involved,
            "related_global_ids": _extract_related_ids(recent, global_id),
            "signals": signals,
            "confidence": confidence,
            "time_window": self.fight_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_fight_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s global_person_id=%s confidence=%.3f",
            output.get("type"),
            global_id,
            output.get("confidence"),
        )

    def _maybe_emit_attention(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_attention_output, now, self.participation_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.participation_window_seconds)
        if not recent:
            return
        teacher_ratio, notebook_ratio, unknown_ratio = _focus_ratios(recent)
        total_focus = teacher_ratio + notebook_ratio + unknown_ratio
        if total_focus <= 0.0:
            return
        event_ts = _latest_event_ts(recent, now)
        device_penalty = min(
            0.5,
            (
                _count_event_types(recent, {"student_phone_usage"}) * 0.12
                + _count_object_association(
                    recent, {"phone", "laptop", "tablet", "device"}
                )
                * 0.08
            ),
        )
        score = _clamp01(
            (teacher_ratio * 0.65)
            + (notebook_ratio * 0.25)
            + ((1.0 - unknown_ratio) * 0.1)
            - device_penalty
        )

        if teacher_ratio >= notebook_ratio and teacher_ratio >= unknown_ratio:
            focus_mode = "teacher"
        elif notebook_ratio >= unknown_ratio:
            focus_mode = "notebook"
        else:
            focus_mode = "unknown"

        output = {
            "timestamp": event_ts,
            "type": "attention_summary",
            "student_person_id": self._person_id_for_global(global_id),
            "student_global_id": global_id,
            "attention_level": self._level_from_score(score),
            "attention_score": score,
            "focus_mode": focus_mode,
            "teacher_present": self._teacher_present(global_id, event_ts),
            "signals": {
                "teacher_ratio": teacher_ratio,
                "notebook_ratio": notebook_ratio,
                "unknown_ratio": unknown_ratio,
                "device_penalty": device_penalty,
            },
            "time_window": self.participation_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_attention_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s student_global_id=%s level=%s",
            output.get("type"),
            global_id,
            output.get("attention_level"),
        )

    def _maybe_emit_offtask(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_offtask_output, now, self.participation_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.participation_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)
        if self._teacher_present(global_id, event_ts):
            return
        movement = _movement_distance(recent)
        device_assoc = _count_object_association(
            recent, {"phone", "laptop", "tablet", "device"}
        )
        phone_events = _count_event_types(recent, {"student_phone_usage"})
        if (
            movement < self._offtask_min_movement
            and device_assoc == 0
            and phone_events == 0
        ):
            return
        output = {
            "timestamp": event_ts,
            "type": "offtask_movement",
            "student_person_id": self._person_id_for_global(global_id),
            "student_global_id": global_id,
            "movement_score": movement,
            "device_events": device_assoc + phone_events,
            "confidence": _clamp01(
                0.4 + min(0.4, movement / 450.0) + min(0.2, (device_assoc + phone_events) * 0.08)
            ),
            "teacher_present": False,
            "time_window": self.participation_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_offtask_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s student_global_id=%s movement=%.1f",
            output.get("type"),
            global_id,
            movement,
        )

    def _maybe_emit_sleeping(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_sleep_output, now, self._sleep_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.participation_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)
        sleep_events = [
            event
            for event in recent
            if event.get("event_type") == "sleeping_suspected"
        ]
        bowing_events = sum(
            1
            for event in recent
            if event.get("event_type") == "posture_changed"
            and event.get("posture") == "bowing"
        )
        head_down = _count_head_down(recent)
        longest_sleep = 0.0
        for event in sleep_events:
            longest_sleep = max(longest_sleep, _get_float(event, "sleep_duration_seconds"))
        if (
            longest_sleep < self._sleep_min_duration_seconds
            and head_down < self._sleep_min_head_down_events
            and bowing_events < self._sleep_min_bowing_events
        ):
            return
        risk_score = _clamp01(
            (0.35 if sleep_events else 0.0)
            + min(0.45, longest_sleep / 18.0)
            + min(0.25, head_down / 8.0)
            + min(0.2, bowing_events / 5.0)
        )
        if risk_score < self._sleep_min_risk_score:
            return
        output = {
            "timestamp": event_ts,
            "type": "student_sleep_risk",
            "student_person_id": self._person_id_for_global(global_id),
            "student_global_id": global_id,
            "sleep_risk_level": self._level_from_score(risk_score),
            "sleep_risk_score": risk_score,
            "signals": {
                "sleep_events": len(sleep_events),
                "longest_sleep_duration_seconds": longest_sleep,
                "bowing_events": bowing_events,
                "head_down_events": head_down,
            },
            "time_window": self.participation_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_sleep_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s student_global_id=%s score=%.3f",
            output.get("type"),
            global_id,
            risk_score,
        )

    def _maybe_emit_student_device_distraction(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_student_device_output, now, self._device_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.participation_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)
        phone_events = _count_event_types(recent, {"student_phone_usage"})
        device_assoc = _count_object_association(
            recent, {"phone", "laptop", "tablet", "device"}
        )
        visible_device = _count_event_types(recent, {"device_usage_detected"})
        total = phone_events + device_assoc + visible_device
        if total <= 0:
            return
        score = _clamp01(
            (phone_events * 0.32) + (device_assoc * 0.08) + (visible_device * 0.06)
        )
        output = {
            "timestamp": event_ts,
            "type": "student_device_distraction",
            "student_person_id": self._person_id_for_global(global_id),
            "student_global_id": global_id,
            "distraction_level": self._level_from_score(score),
            "distraction_score": score,
            "signals": {
                "student_phone_events": phone_events,
                "device_association_events": device_assoc,
                "device_visibility_events": visible_device,
            },
            "time_window": self.participation_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_student_device_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s student_global_id=%s score=%.3f",
            output.get("type"),
            global_id,
            score,
        )

    def _maybe_emit_teacher_device_usage(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_teacher_device_output, now, self._device_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.teacher_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)
        teacher_phone = _count_event_types(recent, {"teacher_phone_usage"})
        teacher_device = _count_event_type_by_role(
            recent, "device_usage_detected", "teacher", self._roles
        )
        if teacher_phone <= 0 and teacher_device <= 0:
            return
        score = _clamp01((teacher_phone * 0.35) + (teacher_device * 0.1))
        output = {
            "timestamp": event_ts,
            "type": "teacher_device_usage",
            "teacher_person_id": self._person_id_for_global(global_id),
            "teacher_global_id": global_id,
            "usage_level": self._level_from_score(score),
            "usage_score": score,
            "signals": {
                "teacher_phone_events": teacher_phone,
                "teacher_device_events": teacher_device,
            },
            "time_window": self.teacher_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_teacher_device_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s teacher_global_id=%s score=%.3f",
            output.get("type"),
            global_id,
            score,
        )

    def _maybe_emit_student_behavior_summary(
        self, global_id: int, window: PersonSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_behavior_output, now, self._behavior_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.participation_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)
        teacher_ratio, notebook_ratio, unknown_ratio = _focus_ratios(recent)
        attention_score = _clamp01(
            (teacher_ratio * 0.7) + (notebook_ratio * 0.3) - (unknown_ratio * 0.2)
        )
        movement_score = min(1.0, _movement_distance(recent) / 260.0)
        collaboration_score = min(
            1.0,
            (
                _count_group_membership(recent, self._person_id_for_global(global_id))
                + _count_proximity_close(recent)
            )
            / 6.0,
        )
        device_penalty = min(
            0.6,
            (
                _count_event_types(recent, {"student_phone_usage"}) * 0.12
                + _count_object_association(
                    recent, {"phone", "laptop", "tablet", "device"}
                )
                * 0.08
            ),
        )
        sleep_penalty = min(
            0.6,
            (
                _count_event_types(recent, {"sleeping_suspected"}) * 0.22
                + _count_head_down(recent) * 0.03
            ),
        )
        teacher_bonus = 0.1 if self._teacher_present(global_id, event_ts) else 0.0
        score = _clamp01(
            (0.35 * attention_score)
            + (0.25 * movement_score)
            + (0.3 * collaboration_score)
            + teacher_bonus
            - device_penalty
            - sleep_penalty
        )
        output = {
            "timestamp": event_ts,
            "type": "student_behavior_summary",
            "student_person_id": self._person_id_for_global(global_id),
            "student_global_id": global_id,
            "behavior_level": self._level_from_score(score),
            "behavior_score": score,
            "teacher_present": self._teacher_present(global_id, event_ts),
            "signals": {
                "attention_score": attention_score,
                "movement_score": movement_score,
                "collaboration_score": collaboration_score,
                "device_penalty": device_penalty,
                "sleep_penalty": sleep_penalty,
            },
            "time_window": self.participation_window_seconds,
        }
        self._attach_location(output, global_id)
        window.last_behavior_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s student_global_id=%s score=%.3f",
            output.get("type"),
            global_id,
            score,
        )

    def _teacher_present(self, global_id: int, now: float) -> bool:
        room = self._person_rooms.get(global_id)
        if room is None:
            return False
        room_id, _camera_id = room
        return self.schedule.is_teacher_present(room_id, now)

    def _maybe_emit_group_participation(
        self, group_id: int, window: GroupSignalWindow, now: float
    ) -> None:
        if not self._should_emit(
            window.last_output, now, self.participation_emit_interval_seconds
        ):
            return
        recent = _filter_recent(window.events, now - self.participation_window_seconds)
        if not recent:
            return
        event_ts = _latest_event_ts(recent, now)
        duration = _group_duration(recent)
        if duration <= 0.0:
            return
        motion_intensity = _group_motion_intensity(recent)
        members = _group_members(recent)
        student_members = [pid for pid in members if self._person_roles.get(pid) == "student"]

        score = _clamp01(min(1.0, duration / 30.0) + (motion_intensity * 0.25))
        level = self._level_from_score(score)

        output = {
            "timestamp": event_ts,
            "type": "group_participation_summary",
            "group_id": group_id,
            "participation_level": level,
            "participation_score": score,
            "duration_seconds": duration,
            "student_person_ids": sorted(student_members),
            "confidence": min(1.0, 0.35 + score),
            "time_window": self.participation_window_seconds,
        }
        if isinstance(window.room_id, str):
            output["room_id"] = window.room_id
        window.last_output = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s group_id=%s level=%s",
            output.get("type"),
            group_id,
            level,
        )

        collab_output = {
            "timestamp": event_ts,
            "type": "group_collaboration",
            "group_id": group_id,
            "student_person_ids": sorted(student_members),
            "student_global_ids": [
                gid for gid, pid in self._person_ids.items() if pid in student_members
            ],
            "duration_seconds": duration,
            "confidence": min(1.0, 0.45 + min(0.55, duration / 35.0)),
            "time_window": self.participation_window_seconds,
        }
        if isinstance(window.room_id, str):
            collab_output["room_id"] = window.room_id
        self._append_output(collab_output, now)
        logger.info(
            "inference.output type=%s group_id=%s students=%d",
            collab_output.get("type"),
            group_id,
            len(student_members),
        )

    def _maybe_emit_lesson_summary(self, room_id: str, now: float) -> None:
        last_emit = self._room_last_lesson_output.get(room_id, 0.0)
        if not self._should_emit(last_emit, now, self._lesson_emit_interval_seconds):
            return
        room_events = self._room_events.get(room_id)
        if not room_events:
            return
        recent = _filter_recent(room_events, now - self._lesson_window_seconds)
        if len(recent) < self._lesson_min_event_count:
            return

        teacher_focus_ratio = _room_teacher_focus_ratio(recent, self._roles)
        teacher_movement = _count_event_by_role(
            recent, "body_movement", "teacher", self._roles
        )
        student_movement = _count_event_by_role(
            recent, "body_movement", "student", self._roles
        )
        group_events = _count_event_types(recent, {"group_formed", "group_updated"})
        sleep_events = _count_event_types(recent, {"sleeping_suspected"})
        student_phone_events = _count_event_types(recent, {"student_phone_usage"})
        teacher_phone_events = _count_event_types(recent, {"teacher_phone_usage"})
        proximity_close = _count_proximity_close(recent)

        participation_signal = _clamp01(
            (student_movement / 25.0) + (group_events / 10.0) + (proximity_close / 20.0)
        )
        teacher_activity_signal = _clamp01(teacher_movement / 10.0)
        disruption_penalty = min(
            0.9,
            (student_phone_events * 0.06)
            + (teacher_phone_events * 0.15)
            + (sleep_events * 0.12),
        )
        quality_score = _clamp01(
            (0.4 * teacher_focus_ratio)
            + (0.3 * teacher_activity_signal)
            + (0.3 * participation_signal)
            - disruption_penalty
        )

        strengths: List[str] = []
        concerns: List[str] = []
        if teacher_focus_ratio >= self._lesson_teacher_focus_strength_threshold:
            strengths.append("students often focused toward teacher")
        if participation_signal >= self._lesson_participation_strength_threshold:
            strengths.append("strong participation and collaboration activity")
        if teacher_activity_signal >= self._lesson_teacher_activity_strength_threshold:
            strengths.append("teacher remained active in classroom")
        if student_phone_events >= self._lesson_student_phone_concern_threshold:
            concerns.append("frequent student phone activity")
        if teacher_phone_events >= self._lesson_teacher_phone_concern_threshold:
            concerns.append("teacher phone usage may reduce engagement")
        if sleep_events >= self._lesson_sleep_concern_threshold:
            concerns.append("repeated sleeping/fatigue behavior detected")
        if not strengths:
            strengths.append("baseline classroom activity present")
        if not concerns:
            concerns.append("no major behavior risk detected in this window")

        output = {
            "timestamp": now,
            "type": "lesson_comprehensive_summary",
            "room_id": room_id,
            "window_seconds": self._lesson_window_seconds,
            "quality_level": self._level_from_score(quality_score),
            "quality_score": quality_score,
            "metrics": {
                "teacher_focus_ratio": teacher_focus_ratio,
                "teacher_activity_signal": teacher_activity_signal,
                "participation_signal": participation_signal,
                "student_phone_events": student_phone_events,
                "teacher_phone_events": teacher_phone_events,
                "sleep_events": sleep_events,
                "group_events": group_events,
            },
            "strengths": strengths,
            "concerns": concerns,
        }
        self._room_last_lesson_output[room_id] = now
        self._append_output(output, now)
        logger.info(
            "inference.output type=%s room_id=%s score=%.3f",
            output.get("type"),
            room_id,
            quality_score,
        )


def _filter_recent(
    events: Deque[Dict[str, object]] | List[Dict[str, object]],
    since: float,
) -> List[Dict[str, object]]:
    return [event for event in events if _get_float(event, "timestamp") >= since]


def _event_cursor_ts(event: Dict[str, object]) -> float:
    emitted = _get_float(event, "emitted_at")
    event_ts = _get_float(event, "timestamp")
    if emitted > 0.0 and event_ts > 0.0:
        return max(emitted, event_ts)
    if emitted > 0.0:
        return emitted
    return event_ts


def _output_cursor_ts(output: Dict[str, object]) -> float:
    emitted = _get_float(output, "emitted_at")
    output_ts = _get_float(output, "timestamp")
    if emitted > 0.0 and output_ts > 0.0:
        return max(emitted, output_ts)
    if emitted > 0.0:
        return emitted
    return output_ts


def _latest_event_ts(events: List[Dict[str, object]], fallback: float) -> float:
    latest = 0.0
    for event in events:
        ts = _get_float(event, "timestamp")
        if ts > latest:
            latest = ts
    if latest > 0.0:
        return latest
    return fallback


def _count_event_types(events: List[Dict[str, object]], event_types: Set[str]) -> int:
    return sum(1 for event in events if event.get("event_type") in event_types)


def _count_event_type_by_role(
    events: List[Dict[str, object]],
    event_type: str,
    role: str,
    roles_by_global: Dict[int, str],
) -> int:
    count = 0
    for event in events:
        if event.get("event_type") != event_type:
            continue
        if _event_role(event, roles_by_global) == role:
            count += 1
    return count


def _count_event_by_role(
    events: List[Dict[str, object]],
    event_type: str,
    role: str,
    roles_by_global: Dict[int, str],
) -> int:
    count = 0
    for event in events:
        if event.get("event_type") != event_type:
            continue
        if _event_role(event, roles_by_global) == role:
            count += 1
    return count


def _count_object(events: List[Dict[str, object]], object_type: str) -> int:
    return sum(
        1
        for event in events
        if event.get("event_type") == "object_detected"
        and event.get("object_type") == object_type
    )


def _count_object_association(events: List[Dict[str, object]], types: Set[str]) -> int:
    return sum(
        1
        for event in events
        if event.get("event_type") == "object_associated"
        and event.get("object_type") in types
    )


def _count_head_turns(events: List[Dict[str, object]]) -> int:
    return sum(
        1
        for event in events
        if event.get("event_type") == "head_orientation_changed"
        and event.get("orientation") in ("left", "right")
    )


def _count_head_down(events: List[Dict[str, object]]) -> int:
    return sum(
        1
        for event in events
        if event.get("event_type") == "head_orientation_changed"
        and event.get("orientation") == "down"
    )


def _count_proximity_close(events: List[Dict[str, object]]) -> int:
    return sum(
        1
        for event in events
        if event.get("event_type") == "proximity_event"
        and event.get("status") == "close"
        and _get_float(event, "duration_seconds") >= 2.0
    )


def _count_sync_turns(events: List[Dict[str, object]], window_seconds: float) -> int:
    head_events = [
        event
        for event in events
        if event.get("event_type") == "head_orientation_changed"
        and event.get("orientation") in ("left", "right")
    ]
    head_events.sort(key=lambda event: _get_float(event, "timestamp"))
    count = 0
    for i, event in enumerate(head_events):
        for other in head_events[i + 1 :]:
            if (
                _get_float(other, "timestamp") - _get_float(event, "timestamp")
                > window_seconds
            ):
                break
            if event.get("orientation") == other.get("orientation"):
                count += 1
                break
    return count


def _movement_distance(events: List[Dict[str, object]]) -> float:
    positions: List[Tuple[float, float]] = []
    for event in events:
        if event.get("event_type") in ("person_detected", "person_tracked"):
            bbox = event.get("bbox")
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


def _count_group_membership(events: List[Dict[str, object]], person_id: str) -> int:
    count = 0
    for event in events:
        if event.get("event_type") in ("group_formed", "group_updated"):
            members = event.get("person_ids")
            if isinstance(members, list) and person_id in members:
                count += 1
    return count


def _group_duration(events: List[Dict[str, object]]) -> float:
    durations = [
        _get_float(event, "duration_seconds")
        for event in events
        if event.get("event_type") in ("group_formed", "group_updated")
    ]
    return max(durations) if durations else 0.0


def _group_members(events: List[Dict[str, object]]) -> List[str]:
    for event in reversed(events):
        if event.get("event_type") in ("group_formed", "group_updated"):
            members = event.get("person_ids")
            if isinstance(members, list):
                return [member for member in members if isinstance(member, str)]
    return []


def _group_motion_intensity(events: List[Dict[str, object]]) -> float:
    updates = len(
        [event for event in events if event.get("event_type") == "group_updated"]
    )
    return min(1.0, updates / 5.0)


def _extract_related_person_ids(
    events: List[Dict[str, object]], person_id: str
) -> List[str]:
    related = set()
    for event in events:
        if event.get("event_type") != "proximity_event":
            continue
        members = event.get("person_ids")
        if isinstance(members, list):
            if person_id in members:
                for pid in members:
                    if isinstance(pid, str) and pid != person_id:
                        related.add(pid)
            continue
        gids = event.get("global_ids")
        if isinstance(gids, list):
            if any(isinstance(gid, int) and f"unknown:{gid}" == person_id for gid in gids):
                for gid in gids:
                    if isinstance(gid, int):
                        related.add(f"unknown:{gid}")
    return sorted(related)


def _extract_related_ids(events: List[Dict[str, object]], global_id: int) -> List[int]:
    related = set()
    for event in events:
        if event.get("event_type") != "proximity_event":
            continue
        gids = event.get("global_ids")
        if not isinstance(gids, list):
            continue
        if global_id in gids:
            for gid in gids:
                if isinstance(gid, int) and gid != global_id:
                    related.add(gid)
    return sorted(related)


def _focus_ratios(events: List[Dict[str, object]]) -> Tuple[float, float, float]:
    focus_counts = {"teacher": 0, "notebook": 0, "unknown": 0}
    for event in events:
        if event.get("event_type") != "attention_observation":
            continue
        mode = event.get("focus_mode")
        if mode not in focus_counts:
            mode = "unknown"
        focus_counts[mode] += 1
    total = sum(focus_counts.values())
    if total <= 0:
        return 0.0, 0.0, 0.0
    return (
        focus_counts["teacher"] / float(total),
        focus_counts["notebook"] / float(total),
        focus_counts["unknown"] / float(total),
    )


def _room_teacher_focus_ratio(
    events: List[Dict[str, object]],
    roles_by_global: Dict[int, str],
) -> float:
    teacher_focus = 0
    total_student_focus = 0
    for event in events:
        if event.get("event_type") != "attention_observation":
            continue
        if _event_role(event, roles_by_global) != "student":
            continue
        mode = event.get("focus_mode")
        total_student_focus += 1
        if mode == "teacher":
            teacher_focus += 1
    if total_student_focus <= 0:
        return 0.0
    return teacher_focus / float(total_student_focus)


def _event_role(
    event: Dict[str, object],
    roles_by_global: Dict[int, str],
    global_id: Optional[int] = None,
) -> str:
    role = event.get("person_role")
    if isinstance(role, str) and role:
        return role
    role = event.get("role")
    if isinstance(role, str) and role:
        return role
    gid = global_id if global_id is not None else _get_int(event, "global_person_id")
    if gid is not None:
        mapped = roles_by_global.get(gid)
        if isinstance(mapped, str):
            return mapped
    return "unknown"

def _clamp01(value: float) -> float:
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


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
