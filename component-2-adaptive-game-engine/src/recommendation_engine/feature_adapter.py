from __future__ import annotations


ASSISTMENTS_ONLY_FIELDS = ("skill_id", "problem_type", "template_id", "position", "tutor_mode", "original")


def candidate_model_support(candidate: dict, state) -> tuple[str, list[str]]:
    """Local catalog has conceptual skill/history transfer but no validated public-ID identity map."""
    if not candidate.get("curriculum_trace_id"):
        return "unsupported", ["missing_curriculum_trace"]
    if candidate.get("assistments_skill_id") and candidate.get("assistments_template_id"):
        return "supported", []
    return "partially supported", ["no_validated_local_to_ASSISTments_candidate_encoding"]


def build_model_input(candidate: dict, state) -> dict:
    status, reasons = candidate_model_support(candidate, state)
    if status != "supported":
        raise ValueError("MODEL_UNSUPPORTED: " + ";".join(reasons))
    raise NotImplementedError("Pilot-time local adapter has not been validated")
