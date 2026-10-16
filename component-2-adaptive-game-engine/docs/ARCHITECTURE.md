# Architecture

The app is one local Python process. `streamlit_app.py` selects either the researcher or child view. Both call `Component2Service`; neither bypasses the backend.

1. `Database` resumes or creates a learner session and reads only that pseudonymous learner's validated first-attempt history.
2. `coverage.choose_game` filters to curriculum/hardware-eligible games before choosing under-assessed coverage, repeated-error reinforcement, or least-recently-assessed rotation.
3. The reused `recommendation_engine` applies prerequisite, grade, Tier A/B, abstention and deterministic fallback rules within the chosen skill.
4. `QuestionFactory` generates one private `Question`. Only its answer-free robot task is sent to `MockRobot` or the child view.
5. `MockC1` is authorized to call the private validation boundary. The answer and profile update commit in one SQLite transaction. Duplicates are idempotent; invalid, unanswered and timeout records are non-scoring.
6. `Component2Service.export_c4` emits mathematical evidence only. C4, not C2, owns any combined screening-risk decision.

The version-2 operational reporting map is A1 Number Sense (Counting, Comparison, Ordering), A2 Number Relationships (Place Value), A3 Arithmetic Calculation (Addition, Subtraction), and A4 Multiplicative Reasoning (Multiplication, Division). This follows the current requested operational grouping and supersedes the old two-games-per-area arrangement; it is not a clinical construct map. Historical questions retain their IDs. Profiles are recomputed from saved events using the versioned mapping and report pooled accuracy with sample counts.

Both demonstrations are available in the same app: local recommendations use the persisted curriculum/history workflow, and research replay connects compatible observed-task inputs to frozen predictions and probability-dependent support actions. `predicted_probability` stays null in local recommendations because no validated local feature adapter exists. The replay does not claim counterfactual candidate ranking or local-game transfer. See `PP1_INTEGRATION_HANDOVER.md` for the implemented flow and exact constraints.
