# C2 Team Integration Guide

## Required workflow

1. Register one pseudonymous child ID with Grade 1–5.
2. Start or resume the child's C2 session.
3. Request the next task for that `session_id`.
4. Send the returned `robot_task` to the robot controller. Preserve its
   `question_id` as the team `AssessmentEvent.task_id`.
5. C1 returns `AssessmentEvent` plus `CubeAssessmentResult` to
   `/adaptive-questions/validated-responses`.
6. Request the next task again until `assessment_status` is terminal.
7. Export `/learners/{id}/evidence?session_id=...` for C4's mathematical input.

The async `StatefulAdaptiveQuestionService` wraps the synchronous portable core
with `asyncio.to_thread`. SQLite owns the C2 projection: issued questions,
recommendation provenance, scored first-attempt events, non-scoring quality
events, and session state. A team-wide event ledger may remain canonical, but
C2 ingestion is idempotent by `event_id` and rejects a conflicting duplicate.

## Contract rules

- `pseudonymous_child_id` maps directly to C2 `learner_id`.
- `session_id` must be the ID returned by C2.
- `task_id` must equal the issued C2 `question_id`.
- `task_type` is the C2 game ID, such as `COUNTING` or `ADDITION`.
- `difficulty_level` is `Tier A` or `Tier B`.
- Only `validation_status == "valid"`, `attempt_count == 1`, and a completed
  event update adaptive history.
- Uncertain, invisible, missing-hand, invalid, and retry results are recorded as
  non-scoring evidence and cannot become ordinary incorrect answers.
- Learner, session, task, game, tier, and expected-answer mismatches are rejected.

`QuestionRecommendation.predicted_success_probability` and `model_version` are
optional. `decision_status=abstained` is used at cold start or when no valid
probability exists; C2 never inserts 0.5, 0.0, or another synthetic probability.
The fallback still chooses a safe curriculum-valid task and its provenance is
returned in the stateful response.

The old `AdaptiveQuestionService.recommend_next(history, age_group, curriculum)`
protocol remains available for source compatibility, but it is insufficient for
cold start and durable assessment because it has no learner/session identity.
New orchestration must use `StatefulAdaptiveQuestionService`.

## C4 boundary

The C2 evidence export is not a `ScreeningProfile`. It retains question-level
history summaries, skill and area accuracy, coverage, stopping reason, data
quality, and recommendation provenance. C4 may combine that mathematical
evidence with C3 evidence downstream; C2 never imports C3 or C4 model code.
