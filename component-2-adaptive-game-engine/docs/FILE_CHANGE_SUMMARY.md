# Final integration change list

All paths below are relative to `ASSISTments Dataset/06_PP1_Final_Component2/component-2-adaptive-game-engine`. Nothing is automatically transferred to the separate GitHub clone. No source files were deleted. Frozen research artifacts and copied research scripts are unchanged.

## Modified files

| File | Purpose |
|---|---|
| `README.md` | Current integration overview, evidence and execution entry points. |
| `config/activity_catalog.csv` | Conservative Grade 1–5 eligibility and operational area mapping. |
| `config/game_definitions.json` | Versioned operational mapping, including Ordering in Number Sense. |
| `contracts/c2_to_c4_profile.schema.json` | Explicit v2 descriptive skill/area counts, coverage, tier and data-quality contract. |
| `docs/ARCHITECTURE.md` | Updated actual flow and area mapping. |
| `requirements-research.txt` | Resolve PyArrow/Streamlit dependency conflict. |
| `src/component2_app/coverage.py` | Grade-aware area mapping and bounded same-game reinforcement. |
| `src/component2_app/database.py` | Validated chronological single-response writes, C1 context, history-derived aggregates, session safeguards and deterministic seed history. |
| `src/component2_app/mocks.py` | Reject mismatched learner/session validation context. |
| `src/component2_app/questions.py` | Grade-aware generation, curriculum metadata and bounded Grade 1 addition. Existing cube and spoken-instruction contracts retained. |
| `src/component2_app/service.py` | Unified C1 ingestion, history-derived profiles, recommendation explanations, Tier A evidence and C4 v2 export. |
| `src/recommendation_engine/curriculum_filter.py` | Enforce question-level curriculum support alongside catalog constraints. |
| `src/recommendation_engine/fallback.py` | Preserve selected-skill reinforcement semantics. |
| `src/recommendation_engine/feature_adapter.py` | Explicitly reject unsupported local-to-ASSISTments encodings. |
| `src/recommendation_engine/learner_state.py` | Track Tier A exposure and correctness separately. |
| `src/recommendation_engine/model_scorer.py` | Hash-verified frozen pipeline, genuine raw research input preprocessing, RT-OFF projection and calibrated inference. |
| `src/recommendation_engine/progression_rules.py` | Base Tier B readiness on Tier A evidence. |
| `src/recommendation_engine/recommender.py` | Version metadata for updated policy and catalog. |
| `streamlit_app.py` | Add explicit research replay view. |
| `tests/test_backend.py` | Update regression expectations for v2 mapping and pooled observed accuracy. |
| `tests/test_streamlit_ui.py` | Grade 1/5 creation and repeated-answer smoke flows, export, skip, pause and replay-error checks. |
| `views/child_view.py` | Show decision reasons and handle paused/ended/no-candidate states. |
| `views/researcher_view.py` | Learner creation, persistent selection, eligibility and C4 snapshot inspection. |

## New files

| File | Purpose |
|---|---|
| `contracts/c1_ingestion.schema.json` | Future C1 validated-answer input boundary. |
| `docs/MODEL_COMPATIBILITY_AUDIT.md` | Exact model schema, artifact relationships and unsupported local-transfer gap. |
| `docs/PP1_INTEGRATION_HANDOVER.md` | Actual architecture diagram, curriculum decisions, contracts, commands, demo and limitations. |
| `docs/FILE_CHANGE_SUMMARY.md` | This file-by-file transfer checklist. |
| `requirements-test.txt` | Runtime and JSON-schema test dependencies. |
| `src/component2_app/research_replay.py` | Genuine frozen-history inference connected to inspectable illustrative support decisions. |
| `src/recommendation_engine/curriculum.py` | Central conservative curriculum support and evidence metadata. |
| `tests/test_end_to_end.py` | Grade 1–5, history, progression, coverage, contract and safety tests. |
| `tests/test_research_integration.py` | Actual frozen-pipeline and compatibility tests; no local ML claim. |
| `tools/change_inventory.py` | Source-hash inventory without Git access. Created immediately before recording the source baseline. |
| `tools/run_research_replay.py` | Independent replay evidence command. |
| `tools/verify_pp1.py` | Actual test report and five grade-specific workflow/profile examples. |
| `views/replay_view.py` | Clearly labelled research-data ML demonstration. |

## Generated evidence

`docs/integration_evidence/` contains `source_before.json`, `changed_files.json`, `test_results.txt`, `verification_summary.json`, `research_replay.json`, and three JSON files for each grade 1–5: `grade_N_recommendation.json`, `grade_N_c4_profile.json`, `grade_N_workflow.json`.

These are synthetic verification examples, not clinical or pilot-study results. See `verification_summary.json` for actual outcomes rather than treating this checklist as a test result. `changed_files.json` independently enumerates source additions/modifications and generated evidence. Historical earlier test reports elsewhere in the project describe earlier runs, not this final integration run.

Copy the source/config/contracts/tests/tools/docs files above into the corresponding Component 2 folder after comparing any existing changes in your separate clone. Do not copy `.venv`, caches, databases or the external dataset. The frozen artifacts remain external and are located through `C2_RESEARCH_WORKSPACE`. See the handover for exact installation, verification and UI commands. No commit or push was performed.
