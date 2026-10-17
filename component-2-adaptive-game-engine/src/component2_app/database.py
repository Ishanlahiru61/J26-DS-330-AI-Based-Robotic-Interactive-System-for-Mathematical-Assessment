from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from integration_ready.errors import IntegrationError
from integration_ready.event_store import parse_timestamp, validate_event
from .coverage import GAME_ORDER, GAME_TO_AREA
from .paths import DEFAULT_DB_PATH, MIGRATION_PATH


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path = DEFAULT_DB_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(MIGRATION_PATH.read_text(encoding="utf-8"))

    def upsert_learner(self, learner_id: str, display_name: str, grade: int, synthetic: bool = False) -> None:
        if not isinstance(learner_id, str) or not learner_id.strip() or type(grade) is not int or not 1 <= grade <= 5:
            raise IntegrationError("INVALID_LEARNER", "learner id and grade 1-5 are required")
        with self.connect() as db:
            previous = db.execute('SELECT grade FROM learners WHERE learner_id=?', (learner_id,)).fetchone()
            if previous and previous['grade'] != grade and db.execute("SELECT 1 FROM sessions WHERE learner_id=? AND status IN ('ACTIVE','PAUSED')", (learner_id,)).fetchone():
                raise IntegrationError('ACTIVE_SESSION_GRADE_CHANGE', 'End the current session before changing grade')
            db.execute("INSERT INTO learners VALUES(?,?,?,?,?) ON CONFLICT(learner_id) DO UPDATE SET display_name=excluded.display_name,grade=excluded.grade", (learner_id, display_name, grade, int(synthetic), now_iso()))
            for game in GAME_ORDER:
                db.execute("INSERT OR IGNORE INTO skill_profiles VALUES(?,?,?,?,?,?,?,?)", (learner_id, game, GAME_TO_AREA[game], 0, 0, None, None, now_iso()))

    def learners(self) -> list[dict]:
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM learners ORDER BY learner_id")]

    def learner(self, learner_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM learners WHERE learner_id=?", (learner_id,)).fetchone()
        if not row: raise IntegrationError("MISSING_LEARNER_ID", "learner does not exist")
        return dict(row)

    def create_or_resume_session(self, learner_id: str) -> dict:
        self.learner(learner_id)
        with self.connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE learner_id=? AND status IN ('ACTIVE','PAUSED') ORDER BY started_at DESC LIMIT 1", (learner_id,)).fetchone()
            stamp = now_iso()
            if row:
                db.execute("UPDATE sessions SET status='ACTIVE',updated_at=? WHERE session_id=?", (stamp, row["session_id"]))
                return {**dict(row), "status": "ACTIVE", "resumed": True}
            sid = str(uuid.uuid4())
            db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,NULL)", (sid, learner_id, "ACTIVE", stamp, stamp))
            return {"session_id": sid, "learner_id": learner_id, "status": "ACTIVE", "started_at": stamp, "resumed": False}

    def session(self, session_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
        if not row: raise IntegrationError("MISSING_SESSION", "session does not exist")
        return dict(row)

    def set_session_status(self, session_id: str, status: str) -> None:
        if status not in {"ACTIVE", "PAUSED", "ENDED"}: raise ValueError(status)
        if status == 'ENDED':
            active = self.active_question(session_id)
            if active:
                self.record_non_scoring(session_id, active['question_id'], 'UNANSWERED')
        stamp = now_iso()
        with self.connect() as db:
            db.execute("UPDATE sessions SET status=?,updated_at=?,ended_at=CASE WHEN ?='ENDED' THEN ? ELSE ended_at END WHERE session_id=?", (status, stamp, status, stamp, session_id))
            if status == "ENDED": db.execute("UPDATE question_instances SET status='CANCELLED' WHERE session_id=? AND status='ACTIVE'", (session_id,))

    def history(self, learner_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute("SELECT i.*,q.game_type,q.tier FROM interactions i JOIN question_instances q USING(question_id) WHERE i.learner_id=? AND i.validation_status='VALID' AND i.attempt_count=1 AND i.correct IN (0,1)", (learner_id,)).fetchall()
        cutoff = datetime.now(timezone.utc)
        safe = []
        for row in rows:
            try:
                stamp = parse_timestamp(row['timestamp'])
            except IntegrationError:
                continue
            if stamp <= cutoff:
                safe.append(dict(row))
        return sorted(safe, key=lambda r: (parse_timestamp(r['timestamp']), r['event_id']))

    def recent_game_history(self, learner_id: str) -> list[dict]:
        return [{k: row[k] for k in ('game_type', 'correct', 'timestamp')} for row in self.history(learner_id)]

    def profiles(self, learner_id: str) -> list[dict]:
        history = self.history(learner_id)
        result = []
        for game in GAME_ORDER:
            rows = [r for r in history if r['game_type'] == game]
            n, correct = len(rows), sum(r['correct'] for r in rows)
            result.append({'learner_id': learner_id, 'game_type': game, 'area': GAME_TO_AREA[game],
                           'valid_interactions': n, 'correct_interactions': correct,
                           'skill_index': 100 * correct / n if n else None,
                           'last_assessed_at': rows[-1]['timestamp'] if n else None})
        return result

    def recommendation_history(self, learner_id: str, session_id: str | None = None) -> list[dict]:
        """Return stored public decision metadata; answer keys are never selected."""
        query = "SELECT recommendation_id,session_id,activity_id,game_type,area,tier,mode,reason_code,fallback_rule,policy_version,created_at,payload_json FROM recommendations WHERE learner_id=?"
        params: list[str] = [learner_id]
        if session_id is not None:
            query += " AND session_id=?"
            params.append(session_id)
        query += " ORDER BY created_at,recommendation_id"
        with self.connect() as db:
            rows = db.execute(query, params).fetchall()
        result = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            result.append({
                "recommendation_id": row["recommendation_id"], "session_id": row["session_id"],
                "activity_id": row["activity_id"], "game_type": row["game_type"],
                "area": row["area"], "tier": row["tier"], "mode": row["mode"],
                "decision_type": payload.get("decision_type"), "reason_code": row["reason_code"],
                "fallback_reason": payload.get("abstention_reason"),
                "predicted_probability": payload.get("predicted_probability"),
                "model_version": payload.get("model_version"), "created_at": row["created_at"],
            })
        return result

    def active_question(self, session_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT q.*,r.payload_json AS recommendation_payload FROM question_instances q JOIN recommendations r USING(recommendation_id) WHERE q.session_id=? AND q.status='ACTIVE'", (session_id,)).fetchone()
        return dict(row) if row else None

    def save_task(self, session_id: str, recommendation: dict, question, timestamp: str | None = None) -> dict:
        rid, stamp = str(uuid.uuid4()), timestamp or now_iso()
        with self.connect() as db:
            db.execute("INSERT INTO recommendations VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (rid, session_id, recommendation["learner_id"], recommendation["recommended_activity_id"], recommendation["game_type"], recommendation["area"], recommendation["tier"], recommendation["recommendation_mode"], recommendation["reason_code"], recommendation.get("fallback_level"), recommendation["policy_version"], stamp, json.dumps(recommendation, sort_keys=True)))
            db.execute("INSERT INTO question_instances VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (question.question_id, rid, session_id, recommendation["recommended_activity_id"], question.game_type, question.tier, question.prompt, json.dumps(question.public_robot_task(), sort_keys=True), question.expected_answer, "ACTIVE", stamp, None))
        return {"recommendation_id": rid, "question_id": question.question_id, "robot_task": question.public_robot_task(), "recommendation": recommendation}

    def record_answer(self, session_id: str, question_id: str, answer: int | None, event_id: str | None = None, response_ms: int | None = None, timestamp: str | None = None, *, validated_correctness: int | None = None, c1_context: dict | None = None) -> dict:
        stamp, eid = timestamp or now_iso(), event_id or str(uuid.uuid4())
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            existing = db.execute("SELECT payload_json FROM interactions WHERE event_id=?", (eid,)).fetchone()
            if existing:
                payload = json.loads(existing[0])
                if payload.get("session_id") == session_id and payload.get("question_id") == question_id and payload.get("answer") == answer and (validated_correctness is None or payload['correct'] == validated_correctness) and payload.get('c1_context') == c1_context:
                    return {"status":"duplicate_ignored","event_id":eid,"correct":payload["correct"]}
                raise IntegrationError("DUPLICATE_EVENT", "event_id already exists with a different payload")
            session = db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            if not session or session["status"] != "ACTIVE": raise IntegrationError("INVALID_SESSION", "session is not active")
            q = db.execute("SELECT * FROM question_instances WHERE question_id=? AND session_id=?", (question_id, session_id)).fetchone()
            if not q or q["status"] != "ACTIVE": raise IntegrationError("INVALID_QUESTION", "question is not active")
            if validated_correctness is None:
                if type(answer) is not int or not 0 <= answer <= 25: raise IntegrationError("INVALID_ANSWER", "detected answer-cube count must be an integer from 0 to 25")
                correct = int(answer == q['expected_answer'])
            else:
                if type(validated_correctness) is not int or validated_correctness not in (0, 1):
                    raise IntegrationError('INVALID_CORRECTNESS', 'correctness must be integer 0 or 1')
                correct = validated_correctness
                if not c1_context or c1_context.get('learner_id') != session['learner_id'] or c1_context.get('activity_id') != q['activity_id']:
                    raise IntegrationError('C1_CONTEXT_MISMATCH', 'learner and activity must match the issued question')
            if response_ms is not None and (type(response_ms) is not int or response_ms < 0):
                raise IntegrationError('INVALID_RESPONSE_TIME', 'optional response_ms must be a nonnegative integer')
            instant = parse_timestamp(stamp)
            if instant > datetime.now(timezone.utc):
                raise IntegrationError('FUTURE_EVENT', 'response timestamp cannot be in the future')
            if instant < parse_timestamp(q['created_at']):
                raise IntegrationError('PRE_QUESTION_EVENT', 'response predates question presentation')
            prior = db.execute('SELECT timestamp FROM interactions WHERE learner_id=?', (session['learner_id'],)).fetchall()
            if any(instant < parse_timestamp(r['timestamp']) for r in prior):
                raise IntegrationError('NON_CHRONOLOGICAL_EVENT', 'response predates learner history')
            stamp = instant.astimezone(timezone.utc).isoformat()
            event = validate_event({"event_id":eid,"learner_id":session["learner_id"],"activity_id":q["activity_id"],"correct":correct,"attempt_count":1,"timestamp":stamp}, allow_retry=False)
            learner_grade = db.execute('SELECT grade FROM learners WHERE learner_id=?', (session['learner_id'],)).fetchone()[0]
            payload = {**event,"session_id":session_id,"question_id":question_id,"answer":answer,"response_ms":response_ms,
                       'grade': learner_grade, 'tier': q['tier'], 'game_type': q['game_type'], 'c1_context': c1_context}
            db.execute("INSERT INTO interactions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (eid, session_id, question_id, event["learner_id"], event["activity_id"], answer, correct, 1, event["timestamp"], response_ms, "VALID", json.dumps(payload, sort_keys=True)))
            db.execute("UPDATE question_instances SET status='ANSWERED',answered_at=? WHERE question_id=?", (event["timestamp"], question_id))
            db.execute("UPDATE skill_profiles SET valid_interactions=valid_interactions+1,correct_interactions=correct_interactions+?,skill_index=100.0*(correct_interactions+?)/(valid_interactions+1),last_assessed_at=?,updated_at=? WHERE learner_id=? AND game_type=?", (correct, correct, event["timestamp"], event["timestamp"], event["learner_id"], q["game_type"]))
        return {"status":"recorded","event_id":eid,"correct":correct,"expected_answer":q["expected_answer"]}

    def record_non_scoring(self, session_id: str, question_id: str, status: str, event_id: str | None = None) -> dict:
        if status not in {"INVALID", "UNANSWERED", "TIMEOUT"}: raise ValueError(status)
        eid, stamp = event_id or str(uuid.uuid4()), now_iso()
        with self.connect() as db:
            session = db.execute("SELECT * FROM sessions WHERE session_id=?", (session_id,)).fetchone()
            q = db.execute("SELECT * FROM question_instances WHERE question_id=? AND session_id=?", (question_id, session_id)).fetchone()
            if not session or not q or q["status"] != "ACTIVE": raise IntegrationError("INVALID_QUESTION", "question is not active")
            payload = {"event_id":eid,"session_id":session_id,"question_id":question_id,"learner_id":session["learner_id"],"activity_id":q["activity_id"],"answer":None,"correct":None,"attempt_count":1,"timestamp":stamp,"validation_status":status}
            db.execute("INSERT INTO interactions VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (eid,session_id,question_id,session["learner_id"],q["activity_id"],None,None,1,stamp,None,status,json.dumps(payload,sort_keys=True)))
            db.execute("UPDATE question_instances SET status='CANCELLED',answered_at=? WHERE question_id=?", (stamp,question_id))
        return {"status":status,"event_id":eid,"counted":False}

    def reset_synthetic(self) -> None:
        with self.connect() as db:
            ids = [r[0] for r in db.execute("SELECT learner_id FROM learners WHERE synthetic=1")]
            for learner_id in ids:
                db.execute("DELETE FROM profile_exports WHERE learner_id=?", (learner_id,))
                db.execute("DELETE FROM interactions WHERE learner_id=?", (learner_id,))
                db.execute("DELETE FROM question_instances WHERE session_id IN (SELECT session_id FROM sessions WHERE learner_id=?)", (learner_id,))
                db.execute("DELETE FROM recommendations WHERE learner_id=?", (learner_id,))
                db.execute("DELETE FROM sessions WHERE learner_id=?", (learner_id,))
                db.execute("DELETE FROM skill_profiles WHERE learner_id=?", (learner_id,))
                db.execute("DELETE FROM learners WHERE learner_id=?", (learner_id,))

    def seed_synthetic(self) -> None:
        self.reset_synthetic()
        for lid, name, grade in (("DEMO_001","Cold start",3),("DEMO_002","Strong history",3),("DEMO_003","Needs reinforcement",3),("DEMO_004","Uneven profile",4)):
            self.upsert_learner(lid, name, grade, True)
        patterns = {
            "DEMO_002": [("COUNTING", 1)] * 3,
            "DEMO_003": [("COUNTING", 1)] * 3 + [("COMPARISON", 0)] * 3,
            "DEMO_004": [(game, int(game in {"COUNTING","PLACE_VALUE","ADDITION","MULTIPLICATION"})) for game in GAME_ORDER for _ in range(3)],
        }
        base = datetime.now(timezone.utc) - timedelta(days=1)
        for lid, entries in patterns.items():
            session = self.create_or_resume_session(lid)
            for index, (game, outcome) in enumerate(entries):
                activity = {"COUNTING":"CNT_A_01","COMPARISON":"CMP_A_01","ORDERING":"ORD_A_01","PLACE_VALUE":"PV_A_01","ADDITION":"ADD_A_02","SUBTRACTION":"SUB_A_01","MULTIPLICATION":"MUL_A_01","DIVISION":"DIV_A_01"}[game]
                recommendation = {"learner_id":lid,"recommended_activity_id":activity,"game_type":game,"area":GAME_TO_AREA[game],"tier":"Tier A","recommendation_mode":"SEED","reason_code":"SYNTHETIC_FIXTURE","fallback_level":None,"policy_version":"fixture-v1"}
                from .questions import QuestionFactory
                q = QuestionFactory.generate(game, "Tier A", f"{lid}-{index}")
                self.save_task(session["session_id"], recommendation, q, timestamp=(base + timedelta(minutes=index)).isoformat())
                answer = q.expected_answer if outcome else (q.expected_answer % 25) + 1
                self.record_answer(session["session_id"], q.question_id, answer, timestamp=(base + timedelta(minutes=index)).isoformat())
            self.set_session_status(session["session_id"], "ENDED")
