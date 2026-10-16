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
from .coverage import GAME_ORDER, GAME_TO_AREA, GAME_TO_SKILL, AREA_IDS, choose_game
from recommendation_engine.curriculum import grade_support, curriculum_details
from integration_ready.errors import IntegrationError
from integration_ready.event_store import validate_event
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
            prior_interaction_count=len(history), recent_outcomes=tuple(e["correct"] for e in history[-3:]),
            recent_skill_outcomes=tuple(by_skill.get(current_skill, [])[-3:]),
            recent_skill_outcomes_by_skill={k:tuple(v[-3:]) for k,v in by_skill.items()},
            prior_skill_exposures=dict(exposure), prior_skill_correct=dict(correct),
            tier_a_exposures=dict(tier_a_exposure), tier_a_correct=dict(tier_a_correct),
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
        if not eligible_games:
            return {'question_id': None, 'robot_task': None, 'recommendation': {
                'learner_id': learner['learner_id'], 'recommended_activity_id': None,
                'recommendation_mode': 'HUMAN_REVIEW_REQUIRED', 'reason_code': 'NO_ELIGIBLE_CANDIDATE',
                'grade': learner['grade'], 'game_type': None, 'area': None, 'area_id': None,
                'project_skill': None, 'tier': None, 'fallback_level': 'HUMAN_SELECTION_REQUIRED',
                'abstained': True, 'abstention_reason': 'NO_ELIGIBLE_CANDIDATE',
                'history_count': base.prior_interaction_count, 'created_at': now_iso(),
                'predicted_probability': None, 'curriculum_eligible': False}}
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
            "policy_version":"C2_COVERAGE_PROGRESS_v2", "catalog_version":"c2_eight_game_catalog_v2",
            "grade": learner["grade"], "area_id": AREA_IDS[GAME_TO_AREA[game]],
            "curriculum_eligible": True, "curriculum": curriculum_details(game, selected['tier'], learner['grade']),
            "eligible_candidate_ids": raw['eligible_candidate_ids'], "filtered_candidates": raw['filtered_candidates'],
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
            'profile_policy': '100 * correct / valid first-attempt count; area accuracy pooled by attempts; unassessed is null; counts are not clinical confidence',
        }

    def export_c4(self, learner_id: str, session_id: str | None = None) -> dict:
        if session_id and self.db.session(session_id)['learner_id'] != learner_id:
            raise IntegrationError('SESSION_LEARNER_MISMATCH', 'export session belongs to another learner')
        payload = {"schema_version":"C2_TO_C4_v2.0.0","scoring_policy_version":"C2_DESCRIPTIVE_INDEX_v2.0.0","exported_at":now_iso(),"learner_id":learner_id,"session_id":session_id,"source_component":"C2","profile":self.profile(learner_id),"semantics":{"null_index":"not assessed","confidence":"evidence counts only; no clinical cutoffs","not_a_diagnosis":True}}
        with self.db.connect() as db:
            db.execute("INSERT INTO profile_exports VALUES(?,?,?,?,?,?)", (str(uuid.uuid4()),learner_id,session_id,payload["exported_at"],payload["schema_version"],json.dumps(payload,sort_keys=True)))
        return payload

    def record_non_scoring(self, session_id: str, question_id: str, status: str) -> dict:
        return self.db.record_non_scoring(session_id, question_id, status)
