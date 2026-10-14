from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .errors import IntegrationError


def parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise IntegrationError("INVALID_TIMESTAMP", "timestamp is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise IntegrationError("INVALID_TIMESTAMP", "timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise IntegrationError("INVALID_TIMESTAMP", "timestamp must include a timezone")
    return parsed


def validate_event(event: dict, known_activity_ids: set[str] | None = None, allow_retry: bool = False) -> dict:
    if not isinstance(event, dict):
        raise IntegrationError("MALFORMED_REQUEST", "event must be an object")
    learner_id = str(event.get("learner_id", "")).strip()
    if not learner_id:
        raise IntegrationError("MISSING_LEARNER_ID", "learner_id is required")
    activity_id = str(event.get("activity_id", "")).strip()
    if not activity_id:
        raise IntegrationError("MISSING_ACTIVITY_ID", "activity_id is required")
    if known_activity_ids is not None and activity_id not in known_activity_ids:
        raise IntegrationError("UNKNOWN_ACTIVITY", f"unknown activity_id: {activity_id}")
    if event.get("correct") not in (0, 1, False, True):
        raise IntegrationError("INVALID_CORRECTNESS", "correct must be 0 or 1")
    if "attempt_count" not in event:
        raise IntegrationError("MISSING_ATTEMPT_COUNT", "attempt_count is required for the first-attempt contract")
    if isinstance(event["attempt_count"], bool) or not isinstance(event["attempt_count"], int) or event["attempt_count"] < 1:
        raise IntegrationError("INVALID_ATTEMPT_COUNT", "attempt_count must be an integer >= 1")
    if event["attempt_count"] > 1 and not allow_retry:
        raise IntegrationError("RETRY_EVENT_NOT_ALLOWED", "retry outcomes cannot enter first-attempt learner history")
    parsed = parse_timestamp(event.get("timestamp"))
    clean = dict(event)
    clean["learner_id"] = learner_id
    clean["activity_id"] = activity_id
    clean["correct"] = int(event["correct"])
    clean["timestamp"] = parsed.isoformat()
    clean.setdefault("event_id", f"{learner_id}|{activity_id}|{clean['timestamp']}|{event['attempt_count']}")
    return clean


class JsonEventStore:
    """Small local store. Append validation prevents duplicates and time reversal."""

    def __init__(self, path: str | Path, known_activity_ids: set[str] | None = None):
        self.path = Path(path)
        self.known_activity_ids = known_activity_ids

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise IntegrationError("MISSING_HISTORY", "history store is unreadable") from exc
        if not isinstance(value, list):
            raise IntegrationError("MISSING_HISTORY", "history store must contain a list")
        return value

    def get_history(self, learner_id: str) -> list[dict]:
        if not str(learner_id).strip():
            raise IntegrationError("MISSING_LEARNER_ID", "learner_id is required")
        events = [e for e in self._read() if e.get("learner_id") == learner_id]
        return sorted(events, key=lambda e: parse_timestamp(e["timestamp"]))

    def append_event(self, event: dict) -> dict:
        clean = validate_event(event, self.known_activity_ids)
        all_events = self._read()
        if any(e.get("event_id") == clean["event_id"] for e in all_events):
            return {"status": "duplicate_ignored", "error_code": "DUPLICATE_EVENT", "event_id": clean["event_id"]}
        learner_events = [e for e in all_events if e.get("learner_id") == clean["learner_id"]]
        if learner_events:
            last = max(parse_timestamp(e["timestamp"]) for e in learner_events)
            if parse_timestamp(clean["timestamp"]) < last:
                raise IntegrationError("NON_CHRONOLOGICAL_EVENT", "event predates stored learner history")
        all_events.append(clean)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(all_events, indent=2, ensure_ascii=False), encoding="utf-8")
        return {"status": "appended", "event_id": clean["event_id"]}

    def build_state(self, learner_id: str, grade: int, current_skill: str | None = None):
        from .state_builder import build_state_from_history
        return build_state_from_history(learner_id, grade, self.get_history(learner_id), current_skill)
