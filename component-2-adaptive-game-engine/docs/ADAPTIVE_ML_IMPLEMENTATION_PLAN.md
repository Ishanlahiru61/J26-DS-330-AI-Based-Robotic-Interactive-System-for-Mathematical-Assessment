# Adaptive ML implementation plan

Authoritative baseline: the extracted contents of component-2-adaptive-game-engine-github-16-oct-11PM.zip. The original ZIP, external research workspace, frozen artifacts, notebooks, datasets and unrelated components will remain unchanged.

| File path | Planned action | Reason |
|---|---|---|
| docs/ADAPTIVE_ML_IMPLEMENTATION_PLAN.md | Create | Record the pre-edit scope and expected file set. |
| docs/ADAPTIVE_ML_FEASIBILITY_AUDIT.md | Create | Document why frozen ASSISTments scoring is invalid locally and define the bounded experimental local learner-state model. |
| config/adaptive_assessment_settings.json | Create | Version minimum evidence, reinforcement bounds, completion and model thresholds as inspectable educational heuristics. |
| src/recommendation_engine/local_state_model.py | Create | Estimate per-child, per-skill success from strictly prior valid local responses using a transparent Beta-Bernoulli posterior. |
| src/component2_app/coverage.py | Modify | Add a bounded continue/advance/reinforce/switch/complete focus decision while preserving grade eligibility and coverage. |
| src/recommendation_engine/recommender.py | Modify | Consume the experimental state estimate and policy action, retain safe fallback, and return one decision trace. |
| src/component2_app/database.py | Modify | Add read-only public recommendation-history retrieval and align synthetic fixtures with unfinished strong/reinforcement demonstrations; no migration change. |
| src/component2_app/service.py | Modify | Make next_task the single orchestration path, add stopping/coverage status, propagate experimental model provenance, and enrich the C4-compatible descriptive profile. |
| tools/demo_adaptive_assessment.py | Create | Run two fixed-seed Grade 3 assessments in temporary databases: stronger progression and bounded reinforcement. |
| tools/verify_pp1.py | Modify | Save the enhanced assessment/model verification evidence while retaining the full regression runner. |
| tests/test_adaptive_assessment.py | Create | Test the continuous workflow, experimental model validity, action changes, completion, summary accuracy, isolation and fallback. |
| tests/test_end_to_end.py | Modify if required | Update only intentional pathway/stopping expectations while retaining Grade 1–5, C1 and C4 coverage. |
| streamlit_app.py | Modify | Repair the authoritative ZIP's zero-byte entrypoint by wiring the existing researcher, child and research-replay views; no UI redesign. |
| views/child_view.py | Modify | Display adaptive decision type and distinguish completion from a no-candidate warning. |
| README.md | Modify | Document the live workflow, model distinction, commands and limitations. |
| docs/LIMITATIONS.md | Modify | Replace the obsolete rule-only statement with the exact experimental-local/frozen-research boundary. |
| docs/ARCHITECTURE.md | Modify | Describe the implemented live local state-model branch, decision types and stopping/export flow. |
| docs/PP1_INTEGRATION_HANDOVER.md | Modify | Add a supersession notice pointing to the authoritative adaptive-ML handover documents. |
| docs/ADAPTIVE_ML_CHANGE_LOG.md | Create | Record the exact final created/modified/deleted files and dependencies. |
| docs/ADAPTIVE_ML_GITHUB_TRANSFER.md | Create | Provide selective transfer and Windows reproduction instructions. |
| docs/ADAPTIVE_ML_PP1_EXPLANATION.md | Create | Provide the requested 60–90-second explanation and concise Q&A. |
| tools/change_inventory.py | Modify | Remove an inherited assumption that this already-present baseline tool was newly created, so the ZIP-relative change inventory is accurate. |
| requirements-research.txt | Modify | Add the Matplotlib version imported by the preserved Step-3 feature-engineering module during frozen inference. |

Files planned for preservation include src/recommendation_engine/model_scorer.py, src/recommendation_engine/feature_adapter.py, artifacts/model_manifest.json, contracts/c2_to_c4_profile.schema.json, migrations/001_initial.sql, config/activity_catalog.csv and all research/ content. No new package, large model training, schema migration or UI redesign is planned.

The operational local estimator will be labelled EXPERIMENTAL_MODEL_ASSISTED, not H2_GB_RT_OFF_v1. It is a transparent online learner model fitted only from the current child's strict-prior binary responses. Its thresholds are functional PP1 heuristics requiring validation; they are not clinical cutoffs or evidence of recommendation benefit.
