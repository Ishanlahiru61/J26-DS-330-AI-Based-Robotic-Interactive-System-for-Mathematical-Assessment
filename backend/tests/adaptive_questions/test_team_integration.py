import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.modules.adaptive_questions import Component2Service
from backend.app.modules.adaptive_questions.infrastructure.errors import IntegrationError
from backend.app.modules.adaptive_questions.service import StatefulAdaptiveQuestionService
from backend.app.shared.contracts import AssessmentEvent, CubeAssessmentResult


class TeamContractIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.core = Component2Service(Path(self.temp.name) / "c2.sqlite3")
        self.service = StatefulAdaptiveQuestionService(self.core)

    def tearDown(self):
        self.temp.cleanup()

    async def _issued(self):
        await self.service.register_learner("child-1", 3)
        session = await self.service.start_or_resume("child-1")
        task = await self.service.recommend_for_session(session["session_id"])
        return session, task

    @staticmethod
    def _contracts(session, task, *, status="valid", attempt_count=1, event_id="event-1"):
        now = datetime.now(timezone.utc) + timedelta(milliseconds=1)
        event = AssessmentEvent(
            event_id=event_id,
            pseudonymous_child_id="child-1",
            session_id=session["session_id"],
            task_id=task.task_id,
            task_type=task.robot_task.game_type,
            difficulty_level=task.robot_task.tier,
            started_at=now - timedelta(seconds=1),
            completed_at=now,
            model_version="C1-test",
        )
        expected = task.robot_task.metadata["target_number"]
        result = CubeAssessmentResult(
            detected_cube_count=expected,
            cube_colours=[],
            cube_positions=[],
            detected_answer=float(expected),
            expected_answer=float(expected),
            is_correct=True,
            error_value=0,
            response_time_ms=1000,
            attempt_count=attempt_count,
            hint_count=0,
            detection_confidence=0.99,
            validation_status=status,
        )
        return event, result

    async def test_cold_start_is_explicit_abstention_not_fabricated_probability(self):
        _, task = await self._issued()
        self.assertIsNone(task.recommendation.predicted_success_probability)
        self.assertEqual(task.recommendation.decision_status.value, "abstained")
        self.assertIsNone(task.recommendation.model_version)

    async def test_valid_c1_event_is_scored_once_and_correlation_ids_are_preserved(self):
        session, task = await self._issued()
        event, result = self._contracts(session, task)
        recorded = await self.service.ingest_validated_response(event, result)
        self.assertEqual(recorded["status"], "recorded")
        self.assertEqual(len(self.core.db.history("child-1", session["session_id"])), 1)
        duplicate = await self.service.ingest_validated_response(event, result)
        self.assertEqual(duplicate["status"], "duplicate_ignored")
        conflicting = result.model_copy(update={"is_correct": False})
        with self.assertRaises(IntegrationError):
            await self.service.ingest_validated_response(event, conflicting)

    async def test_uncertain_and_retry_results_do_not_enter_scored_history(self):
        session, task = await self._issued()
        event, result = self._contracts(session, task, status="uncertain")
        outcome = await self.service.ingest_validated_response(event, result)
        self.assertFalse(outcome["counted"])
        self.assertEqual(self.core.db.history("child-1", session["session_id"]), [])

        next_task = await self.service.recommend_for_session(session["session_id"])
        event2, result2 = self._contracts(session, next_task, attempt_count=2, event_id="event-2")
        outcome2 = await self.service.ingest_validated_response(event2, result2)
        self.assertFalse(outcome2["counted"])
        self.assertEqual(self.core.db.history("child-1", session["session_id"]), [])

    async def test_c4_export_keeps_descriptive_evidence_and_no_risk_inference(self):
        session, task = await self._issued()
        event, result = self._contracts(session, task)
        await self.service.ingest_validated_response(event, result)
        evidence = await self.service.export_mathematical_evidence("child-1", session["session_id"])
        self.assertEqual(evidence["source_component"], "C2")
        self.assertIn("skills", evidence["profile"])
        self.assertIn("areas", evidence["profile"])
        self.assertIn("assessment", evidence["profile"])
        self.assertEqual(evidence["profile"]["response_evidence"][0]["task_id"], task.task_id)
        self.assertEqual(evidence["profile"]["response_evidence"][0]["hint_count"], 0)
        self.assertNotIn("screening_risk_level", evidence)


class TeamRuntimeTests(unittest.TestCase):
    def test_fastapi_application_registers_stateful_routes(self):
        client = TestClient(app)
        self.assertEqual(client.get("/health").status_code, 200)
        paths = set(client.get("/openapi.json").json()["paths"])
        self.assertIn("/adaptive-questions/sessions/{session_id}/next", paths)
        self.assertIn("/adaptive-questions/validated-responses", paths)

    def test_core_package_copies_to_a_minimal_application_without_backend_imports(self):
        package = Path(__file__).resolve().parents[2] / "app" / "modules" / "adaptive_questions"
        with tempfile.TemporaryDirectory() as folder:
            copied = Path(folder) / "adaptive_questions"
            shutil.copytree(package, copied)
            database = Path(folder) / "portable.sqlite3"
            code = (
                "from adaptive_questions import Component2Service; "
                f"s=Component2Service(r'{database}'); "
                "s.register_learner('portable','portable',3); "
                "sid=s.start_or_resume('portable')['session_id']; "
                "exec(\"for i in range(3):\\n t=s.next_task(sid,i)\\n q=s.db.question(sid,t['question_id'])\\n s.submit(sid,t['question_id'],q['expected_answer'])\"); "
                "t=s.next_task(sid,4); "
                "assert t['robot_task'] and t['recommendation']['model_version']=='TRANSFER_H2_GB_RT_OFF_v1'"
            )
            completed = subprocess.run(
                [sys.executable, "-c", code],
                cwd=folder,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
