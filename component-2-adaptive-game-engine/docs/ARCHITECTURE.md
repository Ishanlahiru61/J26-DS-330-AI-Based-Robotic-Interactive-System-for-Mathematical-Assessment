# Architecture

The app is one local Python process. `streamlit_app.py` selects either the researcher or child view. Both call `Component2Service`; neither bypasses the backend.

1. `Database` resumes or creates a learner session and reads only that pseudonymous learner's validated first-attempt history.
2. `coverage.choose_adaptive_focus` receives only curriculum/hardware-eligible candidates and returns START, CONTINUE, ADVANCE, REINFORCE, SWITCH, COMPLETE or STOP while bounding repetition and prioritizing minimum evidence.
3. `local_state_model` produces an experimental per-child/per-skill Beta-Bernoulli success estimate from strictly prior valid events after two observations. The reused `recommendation_engine` combines that estimate with prerequisite and Tier A/B rules. It reports `EXPERIMENTAL_MODEL_ASSISTED` when the estimate is available and `RULE_FALLBACK` when it abstains.
4. `QuestionFactory` generates one private `Question`. Only its answer-free robot task is sent to `MockRobot` or the child view.
5. `MockC1` is authorized to call the private validation boundary. The answer and profile update commit in one SQLite transaction. Duplicates are idempotent; invalid, unanswered and timeout records are non-scoring.
6. `Component2Service.assessment_status` stops when every grade-applicable game has at least three valid observations or at the 48-interaction safeguard. `export_c4` emits mathematical evidence, status and answer-key-free recommendation history only. C4, not C2, owns any combined screening-risk decision.

The version-2 operational reporting map is A1 Number Sense (Counting, Comparison, Ordering), A2 Number Relationships (Place Value), A3 Arithmetic Calculation (Addition, Subtraction), and A4 Multiplicative Reasoning (Multiplication, Division). This follows the current requested operational grouping and supersedes the old two-games-per-area arrangement; it is not a clinical construct map. Historical questions retain their IDs. Profiles are recomputed from saved events using the versioned mapping and report pooled accuracy with sample counts.

The live local estimate is not produced by the frozen ASSISTments model. `H2_GB_RT_OFF_v1` remains available through research replay for its compatible public-data domain; local candidate IDs/context remain unsupported. The standalone `tools/demo_adaptive_assessment.py` demonstrates the actual live workflow with two temporary Grade 3 learners. See `ADAPTIVE_ML_FEASIBILITY_AUDIT.md` for the implemented boundary.
