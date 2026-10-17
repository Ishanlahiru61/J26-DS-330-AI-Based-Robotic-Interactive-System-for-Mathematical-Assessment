# Selective GitHub transfer

Do not replace the whole GitHub component directory. Compare the target clone and selectively merge the files below.

## Copy as new files

- config/adaptive_assessment_settings.json
- src/recommendation_engine/local_state_model.py
- tests/test_adaptive_assessment.py
- tools/demo_adaptive_assessment.py
- docs/ADAPTIVE_ML_IMPLEMENTATION_PLAN.md
- docs/ADAPTIVE_ML_FEASIBILITY_AUDIT.md
- docs/ADAPTIVE_ML_CHANGE_LOG.md
- docs/ADAPTIVE_ML_GITHUB_TRANSFER.md
- docs/ADAPTIVE_ML_PP1_EXPLANATION.md

## Replace or carefully merge

- README.md
- src/component2_app/coverage.py
- src/component2_app/database.py
- src/component2_app/service.py
- src/recommendation_engine/recommender.py
- streamlit_app.py
- views/child_view.py
- tests/test_backend.py
- tests/test_end_to_end.py
- tools/verify_pp1.py
- tools/change_inventory.py
- docs/LIMITATIONS.md
- docs/ARCHITECTURE.md
- docs/PP1_INTEGRATION_HANDOVER.md
- requirements-research.txt

`matplotlib==3.10.7` is now explicitly declared in `requirements-research.txt`; the preserved ASSISTments Step 3 scoring module imports it during frozen-model replay.

streamlit_app.py is zero bytes in the supplied ZIP, so replacing it with the repaired entrypoint is appropriate unless the separate clone already contains an independent working version. For every other modified file, compare first and preserve unrelated clone changes.

## Must remain unchanged

Keep artifacts/model_manifest.json, config/activity_catalog.csv, config/game_definitions.json, contracts/c2_to_c4_profile.schema.json, migrations/001_initial.sql, src/recommendation_engine/model_scorer.py, src/recommendation_engine/feature_adapter.py and all research/ notebooks, scripts, evidence and visualizations. Do not copy virtual environments, __pycache__, .pyc, databases, temporary directories, large Parquet/CSV datasets or duplicate model artifacts. Do not commit or push automatically.

## Windows PowerShell setup and verification

From the target Component 2 directory:

    py -3.12 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements-runtime.txt -r requirements-test.txt -r requirements-research.txt
    $env:PYTHONPATH = 'src'
    $env:C2_RESEARCH_WORKSPACE = 'C:\path\to\ASSISTments Dataset'
    .\.venv\Scripts\python.exe tools\verify_pp1.py

The external research workspace is required only for the frozen H2_GB_RT_OFF_v1 loading/reproduction tests and research replay. Its model, preprocessor, calibrator, prediction file and chronological history must retain the relative paths in artifacts/model_manifest.json and docs/MODEL_COMPATIBILITY_AUDIT.md. The live local assessment and its demo do not require the external artifacts.

Run the complete standalone demonstration:

    .\.venv\Scripts\python.exe tools\demo_adaptive_assessment.py

Optional UI command:

    .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py

## Expected behavior after transfer

- Early/insufficient local evidence: RULE_FALLBACK with null probability.
- Two or more valid same-skill observations: EXPERIMENTAL_MODEL_ASSISTED with LOCAL_BETA_BERNOULLI_v1 provenance.
- Strong evidence plus Tier B readiness: ADVANCE.
- Low estimate: bounded REINFORCE.
- Sufficient or bounded skill evidence: SWITCH to an under-assessed eligible skill.
- Three valid observations for every grade-applicable game: ASSESSMENT_COMPLETE.
- Frozen XGBoost remains research-only and must never be named as the source of a local estimate.

## Remaining limitations

The local estimator and thresholds have functional tests but no real-child calibration, psychometric validation or demonstrated learning/assessment benefit. The C4 payload remains descriptive. C1 hardware transport, expert curriculum review, local data collection, model validation and prospective policy evaluation remain future work.
