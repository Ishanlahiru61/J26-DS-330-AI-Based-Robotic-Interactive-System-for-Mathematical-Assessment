# API and internal contracts

Canonical Pydantic definitions live in
`backend/app/shared/contracts/schemas.py`. API modules may compose these models
but must not duplicate incompatible versions.

## AssessmentEvent

`event_id`, `pseudonymous_child_id`, `session_id`, `task_id`, `task_type`,
`difficulty_level`, `started_at`, `completed_at`, and `model_version`.

## CubeAssessmentResult

`detected_cube_count`, `cube_colours`, `cube_positions`, `detected_answer`,
`expected_answer`, `is_correct`, `error_value`, `response_time_ms`,
`attempt_count`, `hint_count`, `detection_confidence`, and `validation_status`.

## QuestionRecommendation

`recommended_task_type`, `recommended_difficulty`,
`predicted_success_probability`, `recommendation_reason`,
`history_window_size`, and `model_version`.

## BehaviorObservation

`behavior_label`, `behavior_confidence`, `behavior_duration_ms`,
`finger_count_estimate`, `hand_tracking_quality`, `head_yaw`, `head_pitch`,
`coarse_gaze_direction`, `attention_indicator`, `observation_status`, and
`model_version`.

Allowed observation statuses are `valid`, `uncertain`,
`insufficient_visibility`, and `no_hands_detected`. A low-quality or obstructed
view must not be silently represented as a valid observation.

## ScreeningProfile

`topic_mastery_scores`, `persistent_error_patterns`, `response_time_pattern`,
`finger_counting_pattern`, `attention_pattern`, `screening_risk_level`,
`screening_confidence`, `important_features`, `recommended_next_action`,
`model_version`, and `generated_at`.

The risk level vocabulary is `low`, `moderate`, or `high`. Thresholds are
configuration and must remain explicitly unvalidated until research validation
is complete. A risk level is educational screening support, not a diagnosis.

## REST and asynchronous work

Use REST for immediate session, task, result, and query operations. Use internal
events or background-task adapters for durable profile recomputation and other
long-running work. Event payloads should be versioned and idempotent.
