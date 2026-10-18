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
from .coverage import GAME_ORDER, GAME_TO_AREA, GAME_TO_SKILL, AREA_IDS, choose_adaptive_focus
from recommendation_engine.curriculum import grade_support, curriculum_details
from recommendation_engine.local_state_model import load_adaptive_settings
from recommendation_engine.predictor import PredictorRouter
from integration_ready.errors import IntegrationError
from integration_ready.event_store import validate_event
from .database import Database, now_iso
from .paths import CATALOG_PATH, DEFAULT_DB_PATH
from .questions import QuestionFactory


class Component2Service:
    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db = Database(db_path)
        self.engine = RecommendationEngine(CATALOG_PATH)
        self.predictor = PredictorRouter()
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

    def assessment_status(self, learner_id: str) -> dict:
        learner = self.db.learner(learner_id)
        grade = learner["grade"]
        settings = load_adaptive_settings()
        applicable = sorted({c["game_type"] for c in self.engine.catalog
                             if c["active"] and c["grade_min"] <= grade <= c["grade_max"]
                             and c["tier"] == "Tier A" and grade_support(c["game_type"], c["tier"], grade)})
        profiles = {p["game_type"]: p for p in self.db.profiles(learner_id)}
        minimum = int(settings["minimum_skill_observations"])
        skills = [{"game_type": game,
                   "valid_interactions": int(profiles.get(game, {}).get("valid_interactions", 0)),
                   "minimum_required": minimum,
                   "evidence_status": "SUFFICIENT" if int(profiles.get(game, {}).get("valid_interactions", 0)) >= minimum else "INSUFFICIENT"}
                  for game in applicable]
        total = len(self.db.history(learner_id))
        complete = bool(skills) and all(row["evidence_status"] == "SUFFICIENT" for row in skills)
        stopped = total >= int(settings["maximum_total_interactions"])
        return {
            "status": "COMPLETE" if complete else ("STOPPED_MAX_INTERACTIONS" if stopped else "IN_PROGRESS"),
            "minimum_observations_per_skill": minimum,
            "maximum_total_interactions": int(settings["maximum_total_interactions"]),
            "total_valid_interactions": total,
            "applicable_games": applicable,
            "skills": skills,
            "areas_requiring_further_assessment": sorted({GAME_TO_AREA[row["game_type"]] for row in skills if row["evidence_status"] == "INSUFFICIENT"}),
            "stopping_rule": "all grade-applicable games meet minimum evidence, or the maximum interaction safeguard is reached",
        }

    def _terminal_task(self, learner: dict, session_id: str, status: dict, reason: str) -> dict:
        return {"question_id": None, "robot_task": None, "recommendation": {
            "learner_id": learner["learner_id"], "session_id": session_id,
            "recommended_activity_id": None, "recommendation_mode": "NO_ELIGIBLE_ACTIVITY" if status["status"] == "IN_PROGRESS" else ("ASSESSMENT_COMPLETE" if status["status"] == "COMPLETE" else "ASSESSMENT_STOPPED"),
            "decision_type": "COMPLETE" if status["status"] == "COMPLETE" else "STOP",
            "reason_code": reason, "grade": learner["grade"], "game_type": None,
            "area": None, "area_id": None, "project_skill": None, "tier": None,
            "fallback_level": None, "abstained": True, "abstention_reason": reason,
            "history_count": status["total_valid_interactions"], "created_at": now_iso(),
            "predicted_probability": None, "curriculum_eligible": False,
            "assessment_status": status,
            "decision_trace": {"eligible_candidates": [], "excluded_candidates": [],
                               "decision_type": "COMPLETE" if status["status"] == "COMPLETE" else "STOP",
                               "pathway": "ASSESSMENT_COMPLETE", "fallback_reason": reason}}}

    def _state(self, learner_id: str, grade: int, current_skill: str):
        history = self.db.history(learner_id)
        by_skill: dict[str, list[int]] = {}
        exposure, correct = Counter(), Counter()
        tier_a_exposure, tier_a_correct = Counter(), Counter()
        successes = []
        for event in history:
            skill = self.lookup[event["activity_id"]]["project_skill"]
            exposure[skill] += 1; correct[skill] += event["correct"]
            if event['tier'] == 'Tier A':
                tier_a_exposure[skill] += 1
                tier_a_correct[skill] += event['correct']
            by_skill.setdefault(skill, []).append(event["correct"])
            if event["correct"]: successes.append(event["activity_id"])
        return make_learner_state(
            learner_id=learner_id, current_grade=grade, current_skill_context=current_skill,
            prior_interaction_count=len(history), prior_correct_count=sum(e["correct"] for e in history),
            recent_outcomes=tuple(e["correct"] for e in history[-5:]),
            recent_skill_outcomes=tuple(by_skill.get(current_skill, [])[-3:]),
            recent_skill_outcomes_by_skill={k:tuple(v[-3:]) for k,v in by_skill.items()},
            prior_skill_exposures=dict(exposure), prior_skill_correct=dict(correct),
            tier_a_exposures=dict(tier_a_exposure), tier_a_correct=dict(tier_a_correct),
            skill_history_availability=exposure[current_skill] > 0,
            successfully_attempted_activities=tuple(dict.fromkeys(successes)),
            last_activity=history[-1]["activity_id"] if history else None,
            last_tier=self.lookup[history[-1]["activity_id"]]["tier"] if history else None,
            interactions_since_last_correct=(
                len(history) - 1 - max(i for i, e in enumerate(history) if e["correct"])
                if any(e["correct"] for e in history) else None
            ),
        )

    def next_task(self, session_id: str, seed: int | str | None = None) -> dict:
        session = self.db.session(session_id)
        if session["status"] != "ACTIVE": raise ValueError("Session is not active")
        active = self.db.active_question(session_id)
        if active:
            return {"question_id":active["question_id"],"robot_task":json.loads(active["robot_task_json"]),"recommendation":json.loads(active["recommendation_payload"]),"resumed":True}
        learner = self.db.learner(session["learner_id"])
        status = self.assessment_status(learner["learner_id"])
        if status["status"] != "IN_PROGRESS":
            self.db.set_session_status(session_id, "ENDED")
            return self._terminal_task(learner, session_id, status, status["status"])
        base = self._state(learner["learner_id"], learner["grade"], "Counting / subitizing")
        eligible, filtered = generate_candidates(self.engine.catalog, base)
        eligible_games = {c["game_type"] for c in eligible}
        if not eligible_games:
            return self._terminal_task(learner, session_id, status, "NO_ELIGIBLE_CANDIDATE")
        recent = self.db.recent_game_history(learner["learner_id"])
        predictions = self.predictor.predict_many(base, [c["project_skill"] for c in eligible])
        adaptive = choose_adaptive_focus(self.db.profiles(learner["learner_id"]), eligible, recent, base, predictions)
        game = adaptive["game_type"]
        if not game:
            updated = self.assessment_status(learner["learner_id"])
            if adaptive["decision_type"] == "COMPLETE":
                updated["status"] = "COMPLETE"
            self.db.set_session_status(session_id, "ENDED")
            return self._terminal_task(learner, session_id, updated, adaptive["reason_code"])
        state = self._state(learner["learner_id"], learner["grade"], GAME_TO_SKILL[game])
        candidates, blocked = generate_candidates(self.engine.catalog, state)
        focused = [c for c in candidates if c["game_type"] == game]
        coverage_blocked = [{**c, "filter_reason": "ADAPTIVE_FOCUS_OTHER_GAME"} for c in candidates if c["game_type"] != game]
        adaptive["model_output"] = adaptive.get("model_output") or None
        raw = self.engine.recommend_next_activity(state, eligible_candidates=focused,
            filtered_candidates=blocked + coverage_blocked, adaptive_decision=adaptive)
        selected = self.lookup.get(raw["candidate_activity_id"] or "")
        if not selected or selected["game_type"] != game:
            raise ValueError("Adaptive engine did not return a curriculum-safe focused activity")
        recommendation = {
            "learner_id":learner["learner_id"], "session_id": session_id, "recommended_activity_id":selected["activity_id"],
            "activity_name":selected["activity_name"], "game_type":game, "area":GAME_TO_AREA[game], "project_skill":selected["project_skill"],
            "tier":selected["tier"], "recommendation_mode":raw["recommendation_mode"], "decision_type":raw["decision_type"], "reason_code":raw["reason_code"],
            "policy_transition_reason":raw["reason_code"], "fallback_level":raw["fallback_rule"], "abstained":raw["abstained"],
            "abstention_reason":raw["abstention_reason"], "predicted_probability":raw["predicted_probability"],
            "model_support_status":raw["model_support_status"], "model_version": raw["model_version"],
            "research_model_version": raw["research_model_version"], "model_provenance": raw["model_provenance"],
            "history_count":raw["history_count"],
            "policy_version":"C2_ADAPTIVE_ASSESSMENT_v1", "catalog_version":"c2_eight_game_catalog_v2",
            "grade": learner["grade"], "area_id": AREA_IDS[GAME_TO_AREA[game]],
            "curriculum_eligible": True, "curriculum": curriculum_details(game, selected['tier'], learner['grade']),
            "eligible_candidate_ids": raw['eligible_candidate_ids'], "filtered_candidates": raw['filtered_candidates'],
            "decision_trace": raw["decision_trace"], "assessment_status": status,
            "created_at":now_iso(),
        }
        question = QuestionFactory.generate(game, selected["tier"], seed if seed is not None else str(uuid.uuid4()), grade=learner["grade"])
        return self.db.save_task(session_id, recommendation, question)

    def submit(self, session_id: str, question_id: str, selected_cube_count: int, event_id: str | None = None, response_ms: int | None = None) -> dict:
        return self.db.record_answer(session_id, question_id, selected_cube_count, event_id, response_ms)

    def ingest_c1(self, payload: dict) -> dict:
        """C1 supplies validated correctness; both adapters share the atomic event writer."""
        required = {'event_id', 'learner_id', 'activity_id', 'session_id', 'question_id', 'correct', 'attempt_count', 'timestamp'}
        if not isinstance(payload, dict) or not required.issubset(payload):
            raise IntegrationError('MALFORMED_REQUEST', 'missing required C1 fields')
        if any(not isinstance(payload[k], str) or not payload[k].strip() for k in ('event_id','learner_id','activity_id','session_id','question_id')):
            raise IntegrationError('MALFORMED_REQUEST', 'identifiers must be nonempty strings')
        if type(payload['correct']) is not int or payload['correct'] not in (0, 1):
            raise IntegrationError('INVALID_CORRECTNESS', 'correct must be integer 0 or 1')
        event = validate_event(payload, set(self.lookup))
        return self.db.record_answer(event['session_id'], event['question_id'], None,
            event['event_id'], event.get('response_ms'), event['timestamp'],
            validated_correctness=event['correct'],
            c1_context={'learner_id': event['learner_id'], 'activity_id': event['activity_id']})

    def profile(self, learner_id: str) -> dict:
        """Aggregate the same chronological valid events used by recommendation."""
        learner = self.db.learner(learner_id)
        history = self.db.history(learner_id)
        grade = learner['grade']
        applicable = {c['game_type'] for c in self.engine.catalog if c['active']
                      and c['grade_min'] <= grade <= c['grade_max']
                      and grade_support(c['game_type'], c['tier'], grade)
                      and c['robot_suitability'].lower() == 'yes'
                      and c['physical_cube_suitability'].lower() == 'yes'}
        state = self._state(learner_id, grade, 'Counting / subitizing')
        eligible, filtered = generate_candidates(self.engine.catalog, state)

        def summarize(rows):
            n = len(rows)
            correct = sum(r['correct'] for r in rows)
            recent = rows[-5:]
            return {
                'attempts': n, 'correct': correct, 'incorrect': n-correct,
                'accuracy_percent': 100 * correct / n if n else None,
                'recent_attempts': len(recent),
                'recent_accuracy_percent': 100 * sum(r['correct'] for r in recent) / len(recent) if recent else None,
                'confidence': 'UNASSESSED' if not n else ('LIMITED_EVIDENCE' if n < 3 else 'DESCRIPTIVE_ONLY'),
                'tier_breakdown': {tier: {
                    'attempts': sum(r['tier'] == tier for r in rows),
                    'correct': sum(r['correct'] for r in rows if r['tier'] == tier)}
                    for tier in ('Tier A', 'Tier B')},
                'distinct_activities_attempted': len({r['activity_id'] for r in rows}),
                'last_activity_id': rows[-1]['activity_id'] if rows else None,
                'last_assessed_at': rows[-1]['timestamp'] if rows else None,
            }

        skills = []
        for game in GAME_ORDER:
            rows = [r for r in history if r['game_type'] == game]
            stats = summarize(rows)
            skills.append({
                'game_type': game, 'skill_id': game, 'skill_name': GAME_TO_SKILL[game],
                'area': GAME_TO_AREA[game], 'area_id': AREA_IDS[GAME_TO_AREA[game]],
                'applicable_to_grade': game in applicable,
                'status': 'ASSESSED' if rows else 'NOT_ASSESSED',
                'valid_interactions': stats['attempts'], 'correct_interactions': stats['correct'],
                'skill_index': stats['accuracy_percent'],
                'tier_a_exposure': stats['tier_breakdown']['Tier A']['attempts'],
                'tier_b_exposure': stats['tier_breakdown']['Tier B']['attempts'],
                **stats,
            })
        areas = []
        for area, area_id in AREA_IDS.items():
            games = [g for g in GAME_ORDER if GAME_TO_AREA[g] == area]
            required = [g for g in games if g in applicable]
            rows = [r for r in history if r['game_type'] in games]
            assessed = sorted({r['game_type'] for r in rows})
            covered = [g for g in required if g in assessed]
            status = 'NOT_APPLICABLE' if not required else ('COMPLETE' if len(covered) == len(required) else ('PARTIAL' if covered else 'NOT_ASSESSED'))
            stats = summarize(rows)
            areas.append({
                'area_id': area_id, 'area': area, 'included_skills': games,
                'applicable_skills': required, 'assessed_skills': assessed,
                'status': 'ASSESSED' if rows else 'NOT_ASSESSED',
                'coverage_status': status, 'required_skill_count': len(required),
                'assessed_required_skill_count': len(covered),
                'area_index': stats['accuracy_percent'], **stats,
            })
        with self.db.connect() as db:
            quality = {r['validation_status']: r['n'] for r in db.execute(
                'SELECT validation_status,COUNT(*) AS n FROM interactions WHERE learner_id=? GROUP BY validation_status',
                (learner_id,))}
        return {
            'learner': learner, 'grade': grade, 'generated_at': now_iso(),
            'skills': skills, 'areas': areas,
            'coverage': {'applicable_games': sorted(applicable),
                         'eligible_activity_ids': [c['activity_id'] for c in eligible],
                         'filtered_activities': [{'activity_id': c['activity_id'], 'reason': c['filter_reason']} for c in filtered]},
            'data_quality': {'recorded_status_counts': quality,
                             'excluded_valid_events': quality.get('VALID', 0)-len(history),
                             'response_time_used_for_adaptation': False},
            'area_mapping_version': 'C2_OPERATIONAL_AREAS_v2',
            'assessment': self.assessment_status(learner_id),
            'recommendation_history': self.db.recommendation_history(learner_id),
            'profile_policy': '100 * correct / valid first-attempt count; area accuracy pooled by attempts; unassessed is null; counts are not clinical confidence',
        }

    def export_c4(self, learner_id: str, session_id: str | None = None) -> dict:
        if session_id and self.db.session(session_id)['learner_id'] != learner_id:
            raise IntegrationError('SESSION_LEARNER_MISMATCH', 'export session belongs to another learner')
        profile = self.profile(learner_id)
        if session_id:
            profile['assessment']['session_status'] = self.db.session(session_id)['status']
        payload = {"schema_version":"C2_TO_C4_v2.0.0","scoring_policy_version":"C2_DESCRIPTIVE_INDEX_v2.0.0","exported_at":now_iso(),"learner_id":learner_id,"session_id":session_id,"source_component":"C2","profile":profile,"semantics":{"null_index":"not assessed","confidence":"evidence counts only; no clinical cutoffs","not_a_diagnosis":True,"trained_model":"TRANSFER_H2_GB_RT_OFF_v1 has held-out ASSISTments evidence only and is not validated for local children","fallback":"LOCAL_BETA_BERNOULLI_v1 is a local-history baseline, not a separately trained supervised model"}}
        with self.db.connect() as db:
            db.execute("INSERT INTO profile_exports VALUES(?,?,?,?,?,?)", (str(uuid.uuid4()),learner_id,session_id,payload["exported_at"],payload["schema_version"],json.dumps(payload,sort_keys=True)))
        return payload

    def record_non_scoring(self, session_id: str, question_id: str, status: str) -> dict:
        return self.db.record_non_scoring(session_id, question_id, status)
