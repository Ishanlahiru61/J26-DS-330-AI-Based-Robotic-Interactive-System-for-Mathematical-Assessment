# Adaptive ML change log

All paths are relative to component-2-adaptive-game-engine in the extracted authoritative ZIP workspace. No source file, model artifact, dataset, notebook, migration or contract was deleted. The ZIP's archived research/assistments/step3/__pycache__ directory was excluded from the working copy as requested; it is generated cache, not source.

## Created

| Path | Purpose |
|---|---|
| config/adaptive_assessment_settings.json | Versioned evidence, model, advance, reinforcement, repetition and stopping heuristics. |
| src/recommendation_engine/local_state_model.py | Strict-prior experimental Beta-Bernoulli per-child/per-skill estimator. |
| tests/test_adaptive_assessment.py | Live model, advance, reinforcement, complete coverage, summary and isolation tests. |
| tools/demo_adaptive_assessment.py | Standalone fixed-seed two-learner Grade 3 assessment. |
| docs/ADAPTIVE_ML_IMPLEMENTATION_PLAN.md | Pre-edit scope and planned file inventory. |
| docs/ADAPTIVE_ML_FEASIBILITY_AUDIT.md | Frozen-model compatibility decision, operational model and claim boundary. |
| docs/ADAPTIVE_ML_CHANGE_LOG.md | This exact file record. |
| docs/ADAPTIVE_ML_GITHUB_TRANSFER.md | Selective integration and reproduction guide. |
| docs/ADAPTIVE_ML_PP1_EXPLANATION.md | 60–90-second explanation and Q&A. |

## Modified

| Path | Change |
|---|---|
| README.md | Described the repeated live workflow, estimator/model distinction, stopping rule and demo. |
| src/component2_app/coverage.py | Added bounded START/CONTINUE/ADVANCE/REINFORCE/SWITCH/COMPLETE focus policy while preserving the original choose_game function. |
| src/component2_app/database.py | Added compact answer-key-free recommendation history and smaller unfinished synthetic strong/reinforcement fixtures. No schema migration. |
| src/component2_app/service.py | Unified history, model estimate, curriculum candidates, adaptive decision, task, completion and summary in next_task; added assessment_status and enriched the existing C4 v2 profile. |
| src/recommendation_engine/recommender.py | Added the experimental local-model branch, provenance, candidate/exclusion trace and explicit fallback while preserving the original entry point. |
| streamlit_app.py | Repaired the authoritative ZIP's zero-byte entrypoint using its existing views. |
| views/child_view.py | Displayed decision type and distinguished completion from missing activity. |
| tests/test_backend.py | Made coverage tests stop at the new terminal response and asserted model-supported reinforcement. |
| tests/test_end_to_end.py | Made Grade 1–5 and repeated-error tests completion-aware and accepted only intentional new modes/estimates. |
| tools/verify_pp1.py | Made all five grade simulations run to the documented stopping condition and report model-assisted decision counts. |
| tools/change_inventory.py | Removed an inherited assumption that this baseline tool was newly created. |
| docs/LIMITATIONS.md | Replaced the obsolete rule-only statement with the exact experimental local/frozen research distinction. |
| docs/ARCHITECTURE.md | Updated the actual model/fallback/coverage/stopping flow. |
| docs/PP1_INTEGRATION_HANDOVER.md | Added a supersession notice without deleting historical evidence. |
| requirements-research.txt | Added matplotlib==3.10.7 because the preserved Step-3 module imported during frozen scoring requires it. |

## Dependencies and artifacts

The live local model uses only Python's standard library. One missing existing-research dependency was added: matplotlib==3.10.7, required because the preserved Step-3 module imported during frozen scoring imports Matplotlib. The remaining pinned research/test/runtime requirements are unchanged. No model was trained, replaced or copied.

The final verification regenerates docs/integration_evidence/test_results.txt, verification_summary.json, grade-specific workflow/recommendation/C4 JSON files, source_before.json and changed_files.json. These are functional test evidence, not clinical results.
