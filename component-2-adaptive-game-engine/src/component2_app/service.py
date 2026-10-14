from __future__ import annotations

import csv
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from recommendation_engine.curriculum_filter import generate_candidates
from recommendation_engine.learner_state import make_learner_state
from recommendation_engine.recommender import RecommendationEngine
from .coverage import GAME_TO_AREA, GAME_TO_SKILL, choose_game
from .database import Database, now_iso
from .paths import CATALOG_PATH, DEFAULT_DB_PATH
from .questions import QuestionFactory


class Component2Service:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db = Database(db_path)
        self.engine = RecommendationEngine(CATALOG_PATH)
        with CATALOG_PATH.open(encoding="utf-8-sig", newline="") as handle:
            self.catalog = list(csv.DictReader(handle))
        self.lookup = {row["activity_id"]: row for row in self.catalog}

    def seed_demo_learners(self) -> None:
        self.db.seed_synthetic()

    def start_or_resume(self, learner_id: str) -> dict:
        return self.db.create_or_resume_session(learner_id)

    def pause(self, session_id: str) -> None:
        self.db.set_session_status(session_id, "PAUSED")

    def end(self, session_id: str) -> None:
        self.db.set_session_status(session_id, "ENDED")

    def _state(self, learner_id: str, grade: int, current_skill: str):
        history = self.db.history(learner_id)
        by_skill: dict[str, list[int]] = {}
        exposure, correct = Counter(), Counter()
        successes = []
        for event in history:
            skill = self.lookup[event["activity_id"]]["project_skill"]
            exposure[skill] += 1; correct[skill] += event["correct"]
            by_skill.setdefault(skill, []).append(event["correct"])
            if event["correct"]: successes.append(event["activity_id"])
        return make_learner_state(
            learner_id=learner_id, current_grade=grade, current_skill_context=current_skill,
            prior_interaction_count=len(history), recent_outcomes=tuple(e["correct"] for e in history[-3:]),
            recent_skill_outcomes=tuple(by_skill.get(current_skill, [])[-3:]),
            recent_skill_outcomes_by_skill={k:tuple(v[-3:]) for k,v in by_skill.items()},
            prior_skill_exposures=dict(exposure), prior_skill_correct=dict(correct),
            skill_history_availability=exposure[current_skill] > 0,
            successfully_attempted_activities=tuple(dict.fromkeys(successes)),
            last_activity=history[-1]["activity_id"] if history else None,
            last_tier=self.lookup[history[-1]["activity_id"]]["tier"] if history else None,
        )

    def next_task(self, session_id: str, seed: int | str | None = None) -> dict:
        session = self.db.session(session_id)
        if session["status"] != "ACTIVE": raise ValueError("Session is not active")
        active = self.db.active_question(session_id)
        if active:
            return {"question_id":active["question_id"],"robot_task":json.loads(active["robot_task_json"]),"recommendation":json.loads(active["recommendation_payload"]),"resumed":True}
        learner = self.db.learner(session["learner_id"])
        base = self._state(learner["learner_id"], learner["grade"], "Counting / subitizing")
        eligible, _ = generate_candidates(self.engine.catalog, base)
        eligible_games = {c["game_type"] for c in eligible}
        game, coverage_reason = choose_game(self.db.profiles(learner["learner_id"]), eligible_games, self.db.recent_game_history(learner["learner_id"]))
        state = self._state(learner["learner_id"], learner["grade"], GAME_TO_SKILL[game])
        raw = self.engine.recommend_next_activity(state)
        selected = self.lookup.get(raw["candidate_activity_id"] or "")
        if not selected or selected["game_type"] != game:
            candidates, _ = generate_candidates(self.engine.catalog, state)
            safe = sorted((c for c in candidates if c["game_type"] == game and c["tier"] == "Tier A"), key=lambda c:c["activity_id"])
            if not safe: raise ValueError("No curriculum-safe activity for selected game")
            selected = safe[0]
        recommendation = {
            "learner_id":learner["learner_id"], "recommended_activity_id":selected["activity_id"],
            "activity_name":selected["activity_name"], "game_type":game, "area":GAME_TO_AREA[game], "project_skill":selected["project_skill"],
            "tier":selected["tier"], "recommendation_mode":"RULE_BASED_FALLBACK", "reason_code":coverage_reason,
            "policy_transition_reason":raw["reason_code"], "fallback_level":raw["fallback_rule"], "abstained":True,
            "abstention_reason":raw["abstention_reason"] or "LEARNED_RANKING_UNAVAILABLE", "predicted_probability":None,
            "model_support_status":"separate_research_path", "history_count":raw["history_count"],
            "policy_version":"STEP4_POLICY_v1.0.0+COVERAGE_v1", "catalog_version":"c2_eight_game_catalog_v1.0.0",
            "created_at":now_iso(),
        }
        question = QuestionFactory.generate(game, selected["tier"], seed or str(uuid.uuid4()))
        return self.db.save_task(session_id, recommendation, question)

    def submit(self, session_id: str, question_id: str, selected_cube_count: int, event_id: str | None = None, response_ms: int | None = None) -> dict:
        return self.db.record_answer(session_id, question_id, selected_cube_count, event_id, response_ms)

    def profile(self, learner_id: str) -> dict:
        skills = []
        for row in self.db.profiles(learner_id):
            n = row["valid_interactions"]
            confidence = "UNASSESSED" if n == 0 else ("LIMITED_EVIDENCE" if n < 3 else "POLICY_MINIMUM_REACHED")
            with self.db.connect() as db:
                recent = db.execute("SELECT AVG(correct) AS recent_accuracy,AVG(response_ms) AS mean_response_ms FROM (SELECT i.correct,i.response_ms FROM interactions i JOIN question_instances q USING(question_id) WHERE i.learner_id=? AND i.validation_status='VALID' AND i.attempt_count=1 AND q.game_type=? ORDER BY i.timestamp DESC LIMIT 5)", (learner_id,row["game_type"])).fetchone()
                tiers = db.execute("SELECT SUM(CASE WHEN q.tier='Tier A' THEN 1 ELSE 0 END) AS tier_a_exposure,SUM(CASE WHEN q.tier='Tier B' THEN 1 ELSE 0 END) AS tier_b_exposure FROM interactions i JOIN question_instances q USING(question_id) WHERE i.learner_id=? AND i.validation_status='VALID' AND i.attempt_count=1 AND q.game_type=?", (learner_id,row["game_type"])).fetchone()
            skills.append({**row, "confidence":confidence, **dict(recent), **dict(tiers)})
        areas = []
        for area in sorted({s["area"] for s in skills}):
            values = [s["skill_index"] for s in skills if s["area"] == area]
            complete = len([v for v in values if v is not None]) == 2
            areas.append({"area":area,"area_index":sum(values)/len(values) if complete else None,"status":"COMPLETE" if complete else "PARTIAL"})
        return {"learner":self.db.learner(learner_id),"skills":skills,"areas":areas,"profile_policy":"100 * first_attempt_correct / valid_first_attempt_interactions; no evidence is null"}

    def export_c4(self, learner_id: str, session_id: str | None = None) -> dict:
        payload = {"schema_version":"C2_TO_C4_v1.0.0","scoring_policy_version":"C2_DESCRIPTIVE_INDEX_v1.0.0","exported_at":now_iso(),"learner_id":learner_id,"session_id":session_id,"source_component":"C2","profile":self.profile(learner_id),"semantics":{"null_index":"not assessed","confidence":"evidence-count category","not_a_diagnosis":True}}
        with self.db.connect() as db:
            db.execute("INSERT INTO profile_exports VALUES(?,?,?,?,?,?)", (str(uuid.uuid4()),learner_id,session_id,payload["exported_at"],payload["schema_version"],json.dumps(payload,sort_keys=True)))
        return payload

    def record_non_scoring(self, session_id: str, question_id: str, status: str) -> dict:
        return self.db.record_non_scoring(session_id, question_id, status)
