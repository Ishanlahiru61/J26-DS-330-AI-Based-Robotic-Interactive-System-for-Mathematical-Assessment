import tempfile
import unittest
from pathlib import Path

from backend.app.modules.adaptive_questions import Component2Service


class PolicySimulationTests(unittest.TestCase):
    def run_pattern(self, learner_id, outcome):
        with tempfile.TemporaryDirectory() as folder:
            service = Component2Service(Path(folder) / "simulation.sqlite3")
            service.register_learner(learner_id, learner_id, 3)
            session_id = service.start_or_resume(learner_id)["session_id"]
            occurrences = {}
            for index in range(25):
                task = service.next_task(session_id, f"{learner_id}-{index}")
                if task["question_id"] is None:
                    status = task["recommendation"]["assessment_status"]
                    return index, status
                game = task["recommendation"]["game_type"]
                occurrences[game] = occurrences.get(game, 0) + 1
                question = service.db.question(session_id, task["question_id"])
                correct = outcome(game, occurrences[game])
                answer = question["expected_answer"] if correct else (question["expected_answer"] % 25) + 1
                service.submit(session_id, task["question_id"], answer)
        self.fail("assessment did not terminate by the hard cap")

    def test_strong_grade_three_completes_at_19(self):
        count, status = self.run_pattern("strong", lambda game, occurrence: True)
        self.assertEqual((count, status["status"]), (19, "COMPLETE"))

    def test_struggling_grade_three_completes_at_19_in_current_version(self):
        count, status = self.run_pattern("struggling", lambda game, occurrence: False)
        self.assertEqual((count, status["status"]), (19, "COMPLETE"))

    def test_inconsistent_grade_three_stops_incomplete_at_24(self):
        count, status = self.run_pattern("inconsistent", lambda game, occurrence: occurrence % 2 == 1)
        self.assertEqual(count, 24)
        self.assertEqual(status["status"], "STOPPED_INCOMPLETE_EVIDENCE")
        self.assertEqual(status["stopping_reason"], "HARD_LIMIT_24_VALID_RESPONSES")
