# C2 Team Integration Guide

This guide describes the implemented C2 software interface. C2 emits a
declarative robot task, but it does not execute motors, capture camera data, or
validate physical cube placement. Its evidence is descriptive mathematical
assessment evidence, not a diagnosis or clinically validated screening result.

## HTTP workflow

Start the application from the repository root:

```powershell
python -m pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
```

The following PowerShell sequence is executable against the local server.

### 1. Register a learner

```powershell
$baseUri = "http://127.0.0.1:8000"
$learnerId = "PP1-C2-001"

$learner = Invoke-RestMethod `
  -Method Post `
  -Uri "$baseUri/adaptive-questions/learners" `
  -ContentType "application/json" `
  -Body (@{
    pseudonymous_child_id = $learnerId
    grade = 3
  } | ConvertTo-Json)

$learner
```

Request schema:

```json
{
  "pseudonymous_child_id": "PP1-C2-001",
  "grade": 3
}
```

`grade` must be an integer from 1 through 5. Registration is idempotent for the
same learner and grade; attempting to reuse the ID with another grade is an
integration error.

### 2. Start or resume a session

```powershell
$session = Invoke-RestMethod `
  -Method Post `
  -Uri "$baseUri/adaptive-questions/learners/$learnerId/sessions"

$sessionId = $session.session_id
$session
```

The response includes `session_id`, `learner_id`, `status`, `started_at`, and
`resumed`. A resumed database row can also include `updated_at` and `ended_at`.
The endpoint resumes the learner's latest active or paused session when one
exists.

### 3. Request the next task

```powershell
$next = Invoke-RestMethod `
  -Method Get `
  -Uri "$baseUri/adaptive-questions/sessions/$sessionId/next"

$next | ConvertTo-Json -Depth 12
```

The stateful response has this shape:

```json
{
  "recommendation_id": "<uuid>",
  "session_id": "<uuid>",
  "task_id": "<uuid>",
  "recommendation": {
    "recommended_task_type": "COUNTING",
    "recommended_difficulty": "Tier A",
    "predicted_success_probability": null,
    "decision_status": "abstained",
    "recommendation_reason": "FOUNDATIONAL_GRADE_APPROPRIATE_START",
    "history_window_size": 0,
    "model_version": null
  },
  "robot_task": {
    "question_id": "<same value as task_id>",
    "game_type": "COUNTING",
    "tier": "Tier A",
    "prompt": "Look at the numbered cube showing 4. Place 4 colored cubes.",
    "numbered_cubes": [4],
    "robot_action": "Present one numbered cube showing 4",
    "metadata": {
      "target_number": 4,
      "curriculum": {
        "grade": 3,
        "eligible": true,
        "evidence": "NIE G1 PDF 36,44,56",
        "policy_version": "NIE_CONSERVATIVE_PP1_v2",
        "verification_status": "LOCAL_NIE_AUDIT_WITH_PROJECT_TASK_ADAPTATION",
        "scope": "foundational task sample; not full grade attainment"
      }
    },
    "voice_instruction_text": "Look at the numbered cube showing 4. Place 4 colored cubes.",
    "placement_zones": ["left"],
    "hardware_profile_version": "ROBOT_TWO_NUMBER_CUBES_v1"
  },
  "assessment_status": {},
  "provenance": {}
}
```

Values and generated wording vary by question. `assessment_status` and
`provenance` contain the full current counters and decision trace. At a terminal
state, `task_id`, `recommendation`, and `robot_task` are `null`. Calling this
endpoint while a question is active returns the same task.

The robot task deliberately omits C2's stored expected answer. The robot layer
may render `prompt`, `voice_instruction_text`, cubes, placement zones, and
`robot_action`; it must preserve `question_id` for the C1 event.

### 4. Submit a C1-validated response

C1 supplies the trusted expected answer and its validated detection outcome.
The following example uses variables so the IDs always match the issued task.
Replace `$expectedAnswer`, `$detectedAnswer`, and cube evidence with the actual
trusted C1 values.

```powershell
$expectedAnswer = [double]$next.robot_task.metadata.target_number
$detectedAnswer = $expectedAnswer
$startedAt = (Get-Date).ToUniversalTime().AddSeconds(-3).ToString("o")
$completedAt = (Get-Date).ToUniversalTime().ToString("o")

$validatedBody = @{
  event = @{
    event_id = [guid]::NewGuid().ToString()
    pseudonymous_child_id = $learnerId
    session_id = $sessionId
    task_id = $next.task_id
    task_type = $next.robot_task.game_type
    difficulty_level = $next.robot_task.tier
    started_at = $startedAt
    completed_at = $completedAt
    model_version = if ($next.recommendation.model_version) {
      $next.recommendation.model_version
    } else {
      "C2_POLICY_FALLBACK"
    }
  }
  result = @{
    detected_cube_count = [int]$detectedAnswer
    cube_colours = @("red", "blue", "green", "yellow")
    cube_positions = @(
      @{ x = 0; y = 0 },
      @{ x = 1; y = 0 },
      @{ x = 2; y = 0 },
      @{ x = 3; y = 0 }
    )
    detected_answer = $detectedAnswer
    expected_answer = $expectedAnswer
    is_correct = ($detectedAnswer -eq $expectedAnswer)
    error_value = ($detectedAnswer - $expectedAnswer)
    response_time_ms = 3000
    attempt_count = 1
    hint_count = 0
    detection_confidence = 0.98
    validation_status = "valid"
  }
}

$ingested = Invoke-RestMethod `
  -Method Post `
  -Uri "$baseUri/adaptive-questions/validated-responses" `
  -ContentType "application/json" `
  -Body ($validatedBody | ConvertTo-Json -Depth 12)

$ingested | ConvertTo-Json -Depth 12
```

`AssessmentEvent` fields are `event_id`, `pseudonymous_child_id`, `session_id`,
`task_id`, `task_type`, `difficulty_level`, `started_at`, optional
`completed_at`, and `model_version`. `CubeAssessmentResult` requires
`detected_cube_count`, `cube_colours`, `cube_positions`, nullable
`detected_answer`, `expected_answer`, `is_correct`, `error_value`,
`response_time_ms`, `attempt_count`, `hint_count`, `detection_confidence`, and a
non-empty `validation_status`.

Only `validation_status == "valid"`, `attempt_count == 1`, and a completed event
are scored. Uncertain, invisible, missing-hand, invalid, incomplete, and retry
events remain quality evidence but do not become ordinary incorrect answers.
After ingestion, request `/next` again until `assessment_status.terminal` is
true.

## C4 evidence export

```powershell
$evidence = Invoke-RestMethod `
  -Method Get `
  -Uri "$baseUri/adaptive-questions/learners/$learnerId/evidence?session_id=$sessionId"

$evidence | ConvertTo-Json -Depth 20
```

The export is versioned as `C2_TO_C4_v2.0.0` and includes C2 provenance,
question-level response evidence, skill and mathematical-area summaries,
coverage, data quality, assessment status and stopping reason, and
recommendation history. Its scoring semantics are
`C2_DESCRIPTIVE_INDEX_v2.0.0`. C4 may combine this output with other evidence;
C2 neither creates a C4 `ScreeningProfile` nor makes a diagnostic claim.

## Integration ownership

| Owner | Responsibility |
| --- | --- |
| C2 | Learner/session projection, curriculum filtering, recommendation, question generation, response persistence, assessment status, and evidence export |
| C1 | Physical-cube perception/validation and the trusted `CubeAssessmentResult`, including expected-answer comparison |
| Robot controller | Hardware execution, speech/display, placement interaction, and safe device behaviour |
| Team orchestrator | Call order, correlation IDs, retry policy, transport-level error handling, and forwarding terminal evidence |
| C4 | Downstream screening/risk inference and reporting under its own validated contract |
| Team contract owners | Cross-component schema versioning and coordinated contract changes |

Do not change C2 schemas unilaterally. `pseudonymous_child_id` maps to the C2
learner ID; C2-issued `session_id` and `question_id` must be returned unchanged.

## Troubleshooting

- **Learner/session/task mismatch:** ensure all IDs came from the same active C2
  workflow and `task_id` equals the current `question_id`.
- **Game or tier mismatch:** send `robot_task.game_type` and `robot_task.tier`
  unchanged as `task_type` and `difficulty_level`.
- **Expected-answer mismatch:** C1 must use the trusted assessment context, not
  infer a general answer from the robot payload.
- **Response is stored but history does not advance:** confirm a completed event,
  `validation_status: "valid"`, and `attempt_count: 1`.
- **Repeated task:** this is expected while a question remains active. Submit its
  validated result before requesting another.
- **Duplicate event:** an identical `event_id` is idempotent; changed content
  under the same ID is rejected.
- **Model unavailable or abstained:** inspect response provenance and verify the
  files and hashes declared in `models/transferable/manifest.json`. C2 can
  continue through its fallback/abstention policy.
- **Database location:** set `C2_DATABASE_PATH` before process startup. The
  default is `.c2_runtime/component2.sqlite3` under the current directory.
- **Integration error transport:** the core exposes structured
  `IntegrationError` codes, but this router does not currently install a custom
  FastAPI exception-to-status mapping. Orchestrators should not depend on a
  component-specific HTTP status until the shared transport contract defines it.

The legacy `AdaptiveQuestionService.recommend_next(history, age_group,
curriculum)` protocol remains for source compatibility. It has no durable
learner/session identity and is not the integration path for the stateful
assessment.
