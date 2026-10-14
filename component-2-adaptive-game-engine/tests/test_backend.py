import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from component2_app.coverage import GAME_ORDER, choose_game
from component2_app.mocks import MockC1
from component2_app.service import Component2Service
from integration_ready.errors import IntegrationError


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed_dir = tempfile.TemporaryDirectory()
        cls.seed_path = Path(cls.seed_dir.name) / "seed.db"
        Component2Service(cls.seed_path).seed_demo_learners()

    @classmethod
    def tearDownClass(cls):
        cls.seed_dir.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        target = Path(self.tmp.name) / "component2.db"
        shutil.copy2(self.seed_path, target)
        self.service = Component2Service(target)

    def tearDown(self):
        self.tmp.cleanup()

    def test_fixture_ids_and_isolation(self):
        self.assertEqual([x["learner_id"] for x in self.service.db.learners()], ["DEMO_001","DEMO_002","DEMO_003","DEMO_004"])
        self.assertEqual(len(self.service.db.history("DEMO_001")), 0)
        self.assertGreater(len(self.service.db.history("DEMO_002")), 0)

    def test_cold_start_is_safe_counting(self):
        session = self.service.start_or_resume("DEMO_001")
        task = self.service.next_task(session["session_id"], 1)
        self.assertEqual(task["recommendation"]["game_type"], "COUNTING")
        self.assertEqual(task["recommendation"]["tier"], "Tier A")
        self.assertIsNone(task["recommendation"]["predicted_probability"])

    def test_strong_fixture_demonstrates_tier_b(self):
        session = self.service.start_or_resume("DEMO_002")
        task = self.service.next_task(session["session_id"], 2)
        self.assertEqual(task["recommendation"]["tier"], "Tier B")

    def test_error_fixture_demonstrates_reinforcement(self):
        session = self.service.start_or_resume("DEMO_003")
        task = self.service.next_task(session["session_id"], 3)
        self.assertEqual(task["recommendation"]["game_type"], "COMPARISON")
        self.assertEqual(task["recommendation"]["tier"], "Tier A")
        self.assertEqual(task["recommendation"]["reason_code"], "REPEATED_ERROR_REINFORCEMENT")

    def test_active_question_is_resumed_not_duplicated(self):
        session = self.service.start_or_resume("DEMO_001")
        first = self.service.next_task(session["session_id"], 1)
        second = self.service.next_task(session["session_id"], 2)
        self.assertEqual(first["question_id"], second["question_id"])

    def test_answer_updates_profile_and_next_task_uses_history(self):
        session = self.service.start_or_resume("DEMO_001")
        task = self.service.next_task(session["session_id"], 1)
        expected = self._expected(task["question_id"])
        result = self.service.submit(session["session_id"], task["question_id"], expected)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(self.service.profile("DEMO_001")["skills"][0]["skill_index"], 100.0)
        nxt = self.service.next_task(session["session_id"], 2)
        self.assertNotEqual(nxt["question_id"], task["question_id"])

    def test_zero_cube_answer_is_valid_but_wrong(self):
        session = self.service.start_or_resume("DEMO_001")
        task = self.service.next_task(session["session_id"], 1)
        result = self.service.submit(session["session_id"], task["question_id"], 0)
        self.assertEqual(result["correct"], 0)

    def test_duplicate_response_not_counted_twice(self):
        session = self.service.start_or_resume("DEMO_001")
        task = self.service.next_task(session["session_id"], 1); eid = "fixed-event"
        expected = self._expected(task["question_id"])
        self.service.submit(session["session_id"], task["question_id"], expected, eid)
        duplicate = self.service.submit(session["session_id"], task["question_id"], expected, eid)
        self.assertEqual(duplicate["status"], "duplicate_ignored")
        self.assertEqual(self.service.profile("DEMO_001")["skills"][0]["valid_interactions"], 1)

    def test_conflicting_duplicate_is_rejected(self):
        session = self.service.start_or_resume("DEMO_001")
        task = self.service.next_task(session["session_id"], 1)
        self.service.submit(session["session_id"], task["question_id"], 0, "fixed")
        with self.assertRaises(IntegrationError): self.service.submit(session["session_id"], task["question_id"], 1, "fixed")

    def test_non_scoring_events_do_not_change_profile(self):
        for status in ("INVALID", "UNANSWERED", "TIMEOUT"):
            session = self.service.start_or_resume("DEMO_001")
            task = self.service.next_task(session["session_id"], status)
            self.service.record_non_scoring(session["session_id"], task["question_id"], status)
        self.assertTrue(all(row["valid_interactions"] == 0 for row in self.service.profile("DEMO_001")["skills"]))

    def test_switching_learners_cannot_transfer_question(self):
        a = self.service.start_or_resume("DEMO_001"); task = self.service.next_task(a["session_id"], 1)
        b = self.service.start_or_resume("DEMO_002")
        with self.assertRaises(IntegrationError): self.service.submit(b["session_id"], task["question_id"], 1)

    def test_pause_resume_and_end(self):
        session = self.service.start_or_resume("DEMO_001")
        self.service.pause(session["session_id"])
        resumed = self.service.start_or_resume("DEMO_001")
        self.assertEqual(resumed["session_id"], session["session_id"])
        self.service.end(session["session_id"])
        self.assertEqual(self.service.db.session(session["session_id"])["status"], "ENDED")

    def test_zero_evidence_is_null(self):
        profile = self.service.profile("DEMO_001")
        self.assertTrue(all(x["skill_index"] is None and x["confidence"] == "UNASSESSED" for x in profile["skills"]))
        self.assertTrue(all(x["area_index"] is None and x["status"] == "PARTIAL" for x in profile["areas"]))

    def test_area_summary_requires_both_games(self):
        session = self.service.start_or_resume("DEMO_001")
        task = self.service.next_task(session["session_id"], 1)
        self.service.submit(session["session_id"], task["question_id"], self._expected(task["question_id"]))
        area = next(x for x in self.service.profile("DEMO_001")["areas"] if x["area"] == "Number sense")
        self.assertEqual(area["status"], "PARTIAL"); self.assertIsNone(area["area_index"])

    def test_c4_export_is_versioned_and_nonclinical(self):
        payload = self.service.export_c4("DEMO_001")
        self.assertEqual(payload["schema_version"], "C2_TO_C4_v1.0.0")
        self.assertTrue(payload["semantics"]["not_a_diagnosis"])
        self.assertEqual(len(payload["profile"]["skills"]), 8)
        self.assertEqual(len(payload["profile"]["areas"]), 4)

    def test_coverage_tie_break_is_deterministic(self):
        profiles = [{"game_type":g,"valid_interactions":0,"last_assessed_at":None} for g in GAME_ORDER]
        self.assertEqual(choose_game(profiles, set(GAME_ORDER), [])[0], "COUNTING")

    def test_coverage_reaches_all_eight_games_without_bypassing_prerequisites(self):
        self.service.db.upsert_learner("COVERAGE", "Coverage learner", 3, True)
        session = self.service.start_or_resume("COVERAGE"); seen = set()
        for index in range(32):
            task = self.service.next_task(session["session_id"], index)
            seen.add(task["recommendation"]["game_type"])
            self.service.submit(session["session_id"], task["question_id"], self._expected(task["question_id"]))
        self.assertEqual(seen, set(GAME_ORDER))

    def test_catalog_has_two_tiers_for_eight_games_and_four_two_game_areas(self):
        games = {}
        for row in self.service.catalog: games.setdefault(row["game_type"], set()).add(row["tier"])
        self.assertEqual(set(games), set(GAME_ORDER))
        self.assertTrue(all(tiers == {"Tier A","Tier B"} for tiers in games.values()))
        area_games = {}
        for row in self.service.catalog: area_games.setdefault(row["area"], set()).add(row["game_type"])
        self.assertEqual(len(area_games), 4)
        self.assertTrue(all(len(values) == 2 for values in area_games.values()))

    def test_mock_c1_returns_versioned_validated_event(self):
        session = self.service.start_or_resume("DEMO_001")
        task = self.service.next_task(session["session_id"], 1)
        result = MockC1(self.service).validate_and_submit("DEMO_001", session["session_id"], task["question_id"], self._expected(task["question_id"]))
        self.assertEqual(result["contract_version"], "C1_VALIDATED_RESPONSE_v1.0.0")
        self.assertEqual(result["validation_status"], "VALID")
        self.assertEqual(result["validated_correctness"], 1)

    def test_hardware_unsafe_catalog_candidate_is_filtered(self):
        state = self.service._state("DEMO_002", 3, "Counting / subitizing")
        unsafe = dict(self.service.engine.catalog[0]); unsafe["activity_id"] = "UNSAFE"; unsafe["robot_suitability"] = "no"
        from recommendation_engine.curriculum_filter import curriculum_eligible
        self.assertEqual(curriculum_eligible(unsafe, state), (False, "ROBOT_UNSUITABLE"))

    def test_transaction_rolls_back_on_profile_failure(self):
        session = self.service.start_or_resume("DEMO_001"); task = self.service.next_task(session["session_id"], 1)
        with self.service.db.connect() as db:
            db.execute("DELETE FROM skill_profiles WHERE learner_id='DEMO_001' AND game_type='COUNTING'")
            db.execute("CREATE TRIGGER fail_profile BEFORE UPDATE ON skill_profiles BEGIN SELECT RAISE(ABORT,'forced'); END")
        # Recreate row so the update fires.
        self.service.db.upsert_learner("DEMO_001", "Cold start", 3, True)
        with self.assertRaises(sqlite3.IntegrityError): self.service.submit(session["session_id"], task["question_id"], 1)
        self.assertEqual(len(self.service.db.history("DEMO_001")), 0)

    def test_public_task_never_contains_answer(self):
        session = self.service.start_or_resume("DEMO_001")
        task = self.service.next_task(session["session_id"], 1)
        self.assertNotIn("expected_answer", json.dumps(task["robot_task"]))

    def _expected(self, question_id):
        with self.service.db.connect() as db:
            return db.execute("SELECT expected_answer FROM question_instances WHERE question_id=?", (question_id,)).fetchone()[0]


if __name__ == "__main__": unittest.main()
