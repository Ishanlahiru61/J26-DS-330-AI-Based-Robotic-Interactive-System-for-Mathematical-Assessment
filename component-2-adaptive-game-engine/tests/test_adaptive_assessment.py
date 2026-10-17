import tempfile
import unittest
from pathlib import Path

from component2_app.service import Component2Service
from recommendation_engine.local_state_model import estimate_skill_success
from recommendation_engine.learner_state import make_learner_state


class LocalStateModelTests(unittest.TestCase):
    def test_strict_prior_estimate_abstains_then_separates_patterns(self):
        empty = make_learner_state(learner_id="L", current_grade=3,
                                   prior_skill_exposures={"Addition": 1}, prior_skill_correct={"Addition": 1})
        self.assertIsNone(estimate_skill_success(empty, "Addition")["success_probability"])
        strong = make_learner_state(learner_id="L", current_grade=3,
                                    prior_skill_exposures={"Addition": 3}, prior_skill_correct={"Addition": 3})
        weak = make_learner_state(learner_id="L", current_grade=3,
                                  prior_skill_exposures={"Addition": 3}, prior_skill_correct={"Addition": 0})
        self.assertEqual(estimate_skill_success(strong, "Addition")["success_probability"], .8)
        self.assertEqual(estimate_skill_success(weak, "Addition")["success_probability"], .2)

    def test_invalid_counts_are_rejected(self):
        state = make_learner_state(learner_id="L", current_grade=3,
                                   prior_skill_exposures={"Addition": 2}, prior_skill_correct={"Addition": 3})
        with self.assertRaises(ValueError):
            estimate_skill_success(state, "Addition")


class LiveAdaptiveAssessmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = Component2Service(Path(self.tmp.name) / "live.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _new(self, learner_id):
        self.service.db.upsert_learner(learner_id, learner_id, 3)
        return self.service.start_or_resume(learner_id)["session_id"]

    def _answer(self, sid, task, correct):
        with self.service.db.connect() as db:
            expected = db.execute("SELECT expected_answer FROM question_instances WHERE question_id=?", (task["question_id"],)).fetchone()[0]
        return self.service.submit(sid, task["question_id"], expected if correct else 0)

    def test_stronger_history_advances_and_weak_history_reinforces(self):
        strong = self._new("STRONG")
        strong_tasks = []
        for index in range(4):
            task = self.service.next_task(strong, index); strong_tasks.append(task)
            self._answer(strong, task, True)
        self.assertEqual(strong_tasks[3]["recommendation"]["decision_type"], "ADVANCE")
        self.assertEqual(strong_tasks[3]["recommendation"]["tier"], "Tier B")
        self.assertEqual(strong_tasks[3]["recommendation"]["recommendation_mode"], "EXPERIMENTAL_MODEL_ASSISTED")
        self.assertEqual(strong_tasks[3]["recommendation"]["model_version"], "LOCAL_BETA_BERNOULLI_v1")
        self.assertNotEqual(strong_tasks[3]["recommendation"]["model_version"], "H2_GB_RT_OFF_v1")

        weak = self._new("WEAK")
        weak_tasks = []
        for index in range(4):
            task = self.service.next_task(weak, index); weak_tasks.append(task)
            self._answer(weak, task, False)
        self.assertEqual(weak_tasks[2]["recommendation"]["decision_type"], "REINFORCE")
        self.assertEqual(weak_tasks[3]["recommendation"]["decision_type"], "REINFORCE")
        self.assertEqual(weak_tasks[3]["recommendation"]["tier"], "Tier A")
        self.assertLess(weak_tasks[3]["recommendation"]["predicted_probability"], .4)

    def test_two_complete_patterns_cover_all_grade_three_games_and_summary(self):
        for learner_id, correct in (("A", True), ("B", False)):
            with self.subTest(learner_id=learner_id):
                sid = self._new(learner_id)
                decisions = []
                for index in range(48):
                    task = self.service.next_task(sid, index)
                    if task["question_id"] is None:
                        self.assertEqual(task["recommendation"]["recommendation_mode"], "ASSESSMENT_COMPLETE")
                        break
                    decisions.append(task["recommendation"])
                    self._answer(sid, task, correct)
                else:
                    self.fail("assessment did not terminate")
                summary = self.service.export_c4(learner_id, sid)["profile"]
                self.assertEqual(summary["assessment"]["status"], "COMPLETE")
                self.assertEqual(set(summary["assessment"]["applicable_games"]), {s["game_type"] for s in summary["skills"] if s["applicable_to_grade"]})
                self.assertTrue(all(s["evidence_status"] == "SUFFICIENT" for s in summary["assessment"]["skills"]))
                self.assertEqual(len(summary["recommendation_history"]), len(decisions))
                self.assertIn("SWITCH", {d["decision_type"] for d in decisions})
                self.assertTrue(any(d["recommendation_mode"] == "EXPERIMENTAL_MODEL_ASSISTED" for d in decisions))
                self.assertIn("ADVANCE" if correct else "REINFORCE", {d["decision_type"] for d in decisions})

    def test_sessions_and_histories_remain_isolated(self):
        a, b = self._new("ISO_A"), self._new("ISO_B")
        task = self.service.next_task(a, 1); self._answer(a, task, True)
        self.assertEqual(len(self.service.db.history("ISO_A")), 1)
        self.assertEqual(len(self.service.db.history("ISO_B")), 0)
        self.assertEqual(self.service.assessment_status("ISO_B")["total_valid_interactions"], 0)


if __name__ == "__main__":
    unittest.main()
