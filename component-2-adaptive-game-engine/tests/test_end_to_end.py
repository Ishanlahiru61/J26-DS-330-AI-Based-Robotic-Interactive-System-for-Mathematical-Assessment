import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from component2_app.service import Component2Service
from component2_app.questions import QuestionFactory
from component2_app.coverage import GAME_ORDER, choose_game
from component2_app.database import now_iso
from integration_ready.errors import IntegrationError
from recommendation_engine.curriculum import grade_support
from recommendation_engine.curriculum_filter import generate_candidates, curriculum_eligible
from recommendation_engine.feature_adapter import candidate_model_support, build_model_input
from recommendation_engine.progression_rules import tier_b_evidence

ROOT = Path(__file__).resolve().parents[1]


class IntegratedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'test.db'
        self.service = Component2Service(self.path)
        self.service.db.upsert_learner('L', 'Learner', 1)
        self.sid = self.service.start_or_resume('L')['session_id']

    def tearDown(self):
        self.tmp.cleanup()

    def expected(self, task):
        with self.service.db.connect() as db:
            return db.execute('SELECT expected_answer FROM question_instances WHERE question_id=?', (task['question_id'],)).fetchone()[0]

    def answer(self, correct=True):
        task = self.service.next_task(self.sid, 1)
        self.service.submit(self.sid, task['question_id'], self.expected(task) if correct else 0)
        return task

    def c1(self, task, **changes):
        return {'event_id': 'C1', 'learner_id': 'L', 'session_id': self.sid,
                'question_id': task['question_id'], 'activity_id': task['recommendation']['recommended_activity_id'],
                'correct': 1, 'attempt_count': 1, 'timestamp': now_iso(), **changes}

    def test_all_five_grades_continuous_flow_curriculum_and_coverage(self):
        for grade in range(1, 6):
            with self.subTest(grade=grade):
                lid = f'G{grade}'
                self.service.db.upsert_learner(lid, lid, grade)
                sid = self.service.start_or_resume(lid)['session_id']
                seen = set()
                for i in range(48):
                    task = self.service.next_task(sid, i)
                    if task['question_id'] is None:
                        self.assertEqual(task['recommendation']['recommendation_mode'],'ASSESSMENT_COMPLETE')
                        break
                    rec = task['recommendation']
                    if i == 0:
                        self.assertEqual((rec['game_type'], rec['tier']), ('COUNTING','Tier A'))
                    self.assertTrue(grade_support(rec['game_type'], rec['tier'], grade))
                    self.assertTrue(rec['curriculum_eligible'])
                    self.assertEqual(rec['history_count'], i)
                    seen.add(rec['game_type'])
                    self.service.submit(sid, task['question_id'], self.expected(task))
                else:
                    self.fail('assessment did not terminate')
                expected = set(GAME_ORDER) - ({'PLACE_VALUE','MULTIPLICATION','DIVISION'} if grade == 1 else {'MULTIPLICATION','DIVISION'} if grade == 2 else set())
                self.assertEqual(seen, expected)
                profile = self.service.profile(lid)
                self.assertEqual(sum(s['attempts'] for s in profile['skills']), i)
                self.assertEqual(profile['assessment']['status'],'COMPLETE')
                self.assertTrue(all(a['coverage_status'] in {'COMPLETE','NOT_APPLICABLE'} for a in profile['areas']))

    def test_grade_one_addition_never_exceeds_six(self):
        for seed in range(250):
            q = QuestionFactory.generate('ADDITION', 'Tier A', seed, grade=1)
            self.assertLessEqual(q.expected_answer, 6)

    def test_question_level_rejects_unsupported_advanced_pairs(self):
        for game, tier, grade in [('PLACE_VALUE','Tier A',1),('ORDERING','Tier B',1),
                                  ('MULTIPLICATION','Tier A',2),('MULTIPLICATION','Tier B',4),
                                  ('DIVISION','Tier B',3),('ADDITION','Tier B',1)]:
            with self.subTest(game=game,tier=tier,grade=grade), self.assertRaises(ValueError):
                QuestionFactory.generate(game,tier,0,grade)

    def test_grade_boundaries_and_hardware_for_every_allowed_variant(self):
        for grade in range(1,6):
            for game in GAME_ORDER:
                for tier in ('Tier A','Tier B'):
                    if grade_support(game,tier,grade):
                        for seed in range(20):
                            q=QuestionFactory.generate(game,tier,seed,grade)
                            self.assertLessEqual(len(q.numbered_cubes),2)
                            self.assertTrue(all(0<=x<=9 for x in q.numbered_cubes))
                            self.assertTrue(1<=q.expected_answer<=25)

    def test_inactive_cube_robot_and_prerequisite_exclusions(self):
        state=self.service._state('L',1,'Counting / subitizing')
        candidate=dict(self.service.engine.catalog[0])
        for field,value,reason in [('active',False,'INACTIVE'),('robot_suitability','no','ROBOT_UNSUITABLE'),('physical_cube_suitability','no','CUBE_UNSUITABLE')]:
            self.assertEqual(curriculum_eligible({**candidate,field:value},state),(False,reason))
        _,blocked=generate_candidates(self.service.engine.catalog,state)
        self.assertIn('PREREQUISITE_REQUIRED',{x['filter_reason'] for x in blocked})

    def test_tier_b_requires_tier_a_evidence_and_recent_success(self):
        self.answer(); self.answer()
        state=self.service._state('L',1,'Counting / subitizing')
        self.assertFalse(tier_b_evidence(state,'Counting / subitizing'))
        self.answer()
        state=self.service._state('L',1,'Counting / subitizing')
        self.assertTrue(tier_b_evidence(state,'Counting / subitizing'))
        from recommendation_engine.learner_state import make_learner_state
        base=dict(learner_id='T',current_grade=3,current_skill_context='Addition',
                  prior_skill_exposures={'Addition':10},prior_skill_correct={'Addition':10},
                  recent_skill_outcomes_by_skill={'Addition':(1,1,1)})
        self.assertFalse(tier_b_evidence(make_learner_state(**base),'Addition'))
        self.assertTrue(tier_b_evidence(make_learner_state(**base,tier_a_exposures={'Addition':3},tier_a_correct={'Addition':3}),'Addition'))

    def test_repeated_errors_do_not_starve_other_games(self):
        seen=set()
        for _ in range(40):
            task=self.service.next_task(self.sid)
            if task['question_id'] is None: break
            seen.add(task['recommendation']['game_type'])
            self.service.submit(self.sid,task['question_id'],0)
        self.assertEqual(seen,{'COUNTING','COMPARISON','ORDERING','ADDITION','SUBTRACTION'})
        profiles=[{'game_type':g,'valid_interactions':5,'last_assessed_at':'2020'} for g in GAME_ORDER]
        recent=[{'game_type':'COUNTING','correct':0}]*12
        self.assertNotEqual(choose_game(profiles,set(GAME_ORDER),recent)[0],'COUNTING')

    def test_no_eligible_candidate_has_no_question_or_probability(self):
        self.service.engine.catalog=[]
        task=self.service.next_task(self.sid)
        self.assertIsNone(task['robot_task'])
        self.assertEqual(task['recommendation']['recommendation_mode'],'NO_ELIGIBLE_ACTIVITY')

    def test_external_c1_and_ui_share_history_writer(self):
        task=self.service.next_task(self.sid)
        event=self.c1(task)
        self.assertEqual(self.service.ingest_c1(event)['correct'],1)
        self.assertEqual(self.service.ingest_c1(event)['status'],'duplicate_ignored')
        self.answer(False)
        history=self.service.db.history('L')
        self.assertEqual([e['correct'] for e in history],[1,0])
        self.assertEqual([e['attempt_count'] for e in history],[1,1])
        self.assertEqual(json.loads(history[0]['payload_json'])['grade'],1)

    def test_second_response_for_question_rejected(self):
        task=self.answer()
        with self.assertRaises(IntegrationError):
            self.service.submit(self.sid,task['question_id'],0,'second')

    def test_invalid_c1_events_do_not_change_history(self):
        task=self.service.next_task(self.sid)
        cases=[{'correct':2},{'correct':True},{'attempt_count':2},{'timestamp':'bad'},
               {'timestamp':'2099-01-01T00:00:00Z'},{'learner_id':'other'},
               {'activity_id':'PV_A_01'},{'response_ms':-1}, {'timestamp':'2000-01-01T00:00:00Z'}]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(IntegrationError):
                self.service.ingest_c1(self.c1(task,**case))
        self.assertEqual(self.service.db.history('L'),[])

    def test_restart_resume_and_learner_isolation(self):
        self.answer()
        pending=self.service.next_task(self.sid)
        restarted=Component2Service(self.path)
        self.assertEqual(restarted.next_task(self.sid)['question_id'],pending['question_id'])
        self.assertEqual(len(restarted.db.history('L')),1)
        restarted.db.upsert_learner('OTHER','Other',5)
        self.assertEqual(restarted.db.history('OTHER'),[])

    def test_ending_active_question_records_non_scoring(self):
        self.service.next_task(self.sid)
        self.service.end(self.sid)
        self.assertEqual(self.service.profile('L')['data_quality']['recorded_status_counts'],{'UNANSWERED':1})
        self.assertEqual(self.service.db.history('L'),[])

    def test_profile_counts_weighted_areas_tiers_and_schema(self):
        for i in range(12):
            self.answer(correct=i%3!=0)
        p=self.service.profile('L')
        history=self.service.db.history('L')
        for area in p['areas']:
            events=[e for e in history if e['game_type'] in area['included_skills']]
            self.assertEqual(area['attempts'],len(events))
            self.assertEqual(area['correct'],sum(e['correct'] for e in events))
            if events:
                self.assertAlmostEqual(area['accuracy_percent'],100*area['correct']/len(events))
            else:
                self.assertEqual(area['status'],'NOT_ASSESSED')
                self.assertIsNone(area['accuracy_percent'])
        self.assertEqual(next(a for a in p['areas'] if a['area_id']=='A2')['coverage_status'],'NOT_APPLICABLE')
        import jsonschema
        jsonschema.validate(self.service.export_c4('L'),json.loads((ROOT/'contracts/c2_to_c4_profile.schema.json').read_text()))

    def test_export_cannot_reference_other_learners_session(self):
        self.service.db.upsert_learner('OTHER','Other',2)
        sid=self.service.start_or_resume('OTHER')['session_id']
        with self.assertRaises(IntegrationError): self.service.export_c4('L',sid)

    def test_fake_assistments_ids_cannot_enable_local_scoring(self):
        candidate={**self.service.engine.catalog[0],'assistments_skill_id':'1','assistments_template_id':'2'}
        state=self.service._state('L',1,'Counting / subitizing')
        self.assertNotEqual(candidate_model_support(candidate,state)[0],'supported')
        with self.assertRaises(ValueError): build_model_input(candidate,state)

    def test_no_current_question_outcome_enters_recommendation(self):
        task=self.service.next_task(self.sid)
        self.assertEqual(task['recommendation']['history_count'],0)
        self.assertEqual(sum(s['attempts'] for s in self.service.profile('L')['skills']),0)
        self.assertNotIn('expected_answer',task['robot_task'])

    def test_active_grade_change_rejected(self):
        with self.assertRaises(IntegrationError): self.service.db.upsert_learner('L','Learner',5)

    def test_response_time_does_not_affect_recommendation(self):
        task=self.service.next_task(self.sid,0)
        self.service.submit(self.sid,task['question_id'],self.expected(task),response_ms=999999)
        self.assertFalse(self.service.profile('L')['data_quality']['response_time_used_for_adaptation'])


if __name__=='__main__': unittest.main()
