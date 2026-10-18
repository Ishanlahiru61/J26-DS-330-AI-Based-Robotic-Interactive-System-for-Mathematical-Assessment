from __future__ import annotations

import uuid
from datetime import datetime, timezone


class MockRobot:
    hardware_profile_version = "ROBOT_TWO_NUMBER_CUBES_v1"

    def present(self, robot_task: dict) -> dict:
        if "expected_answer" in robot_task:
            raise ValueError("Public robot task must not expose the answer key")
        if len(robot_task.get("numbered_cubes", [])) > 2:
            raise ValueError("Robot hardware limit exceeded")
        return {**robot_task, "hardware_profile_version": self.hardware_profile_version, "presentation_status":"PRESENTED"}


class MockC1:
    contract_version = "C1_VALIDATED_RESPONSE_v1.0.0"

    def __init__(self, service):
        self.service = service

    def validate_and_submit(self, learner_id: str, session_id: str, question_id: str, detected_answer_cube_count: int, event_id: str | None = None, response_ms: int | None = None) -> dict:
        if self.service.db.session(session_id)['learner_id'] != learner_id:
            raise ValueError('C1 learner/session mismatch')
        result = self.service.submit(session_id, question_id, detected_answer_cube_count, event_id or str(uuid.uuid4()), response_ms)
        return {"contract_version":self.contract_version,"learner_id":learner_id,"session_id":session_id,"question_instance_id":question_id,"event_id":result["event_id"],"detected_answer_cube_count":detected_answer_cube_count,"validated_correctness":result["correct"],"attempt_count":1,"validated_at":datetime.now(timezone.utc).isoformat(),"validation_status":"VALID" if result["status"] == "recorded" else "DUPLICATE_IGNORED","result":result}
