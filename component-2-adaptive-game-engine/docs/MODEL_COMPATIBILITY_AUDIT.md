# Frozen-model compatibility audit

## Verified artifact relationships

All paths below are relative to the external `ASSISTments Dataset` workspace. They come from the existing `artifacts/model_manifest.json`, not inferred naming.

| Artifact | Verified path |
|---|---|
| XGBoost model | `03_Experimental_Modeling/models/gradient_boosting/H2_GB_RT_OFF_v1.json` |
| Shared H2 preprocessor | `03_Experimental_Modeling/models/preprocessors/step3_h2_rt_on_preprocessor_v1.joblib` |
| Validation-fitted Platt calibrator | `03_Experimental_Modeling/models/calibration/H2_GB_RT_OFF_platt_v1.joblib` |
| Saved reference predictions | `03_Experimental_Modeling/data/prepared/predictions_H2_GB_RT_OFF_test_platt_retained.parquet` |
| Frozen strict-prior input rows | `02_Chronological_Learner_History/data/processed/ASSISTments_2012_2013_history_features.parquet` |

Every model/preprocessor/calibrator SHA-256 must match the existing manifest before replay loads it. The saved preprocessing object refers to `__main__.RTStats`; a compatibility dataclass allows loading without fitting or altering the artifact. The original engineering function is reused from the existing preserved `research/assistments/step3/run_step3_pipeline.py`.

The preprocessor accepts 42 raw H2 fields. Template identity becomes log1p training-frequency encoding. Five categorical fields (`skill_id`, `problem_type`, `tutor_mode`, `previous_skill_id`, `previous_problem_type`) use the saved OneHotEncoder with infrequent-category handling and min_frequency=1000. Numeric fields use saved median imputation and StandardScaler. The shared representation has 339 transformed columns; removing columns whose source is one of the four RT fields gives 335 columns, exactly the frozen model width. Platt calibration receives the clipped native-probability logit.

## Exact raw schema and local availability

The table follows the original saved preprocessor order. “Analogue” means the local event stream contains ingredients; it does not authorize applying the public model to a local candidate.

| # | Field | Local compatibility |
|---:|---|---|
|1|skill_id|No validated public skill identity mapping|
|2|problem_type|No validated platform/task type mapping|
|3|tutor_mode|Unavailable ASSISTments platform context|
|4|original|Unavailable platform flag; cannot substitute arbitrary value|
|5|position|Local sequence exists, but platform-position equivalence unvalidated|
|6|template_id|No validated public template identity or training-frequency equivalent|
|7|previous_correct_raw|Prior local correctness analogue; source may include fractional observations|
|8|previous_attempt_count|Local design records one attempt; platform attempt semantics differ|
|9|previous_ms_first_response|Excluded by RT-OFF; optional local response time is not adaptation evidence|
|10|previous_skill_id|Local skill exists; public identity mapping missing|
|11|previous_problem_type|Local game type exists; public type mapping missing|
|12|time_since_previous_seconds|Timestamp-derived analogue; retained by RT-OFF|
|13|has_strict_prior_interaction|Prior-event analogue, requiring equivalent tie handling|
|14|previous_interaction_rt_valid|Excluded by RT-OFF|
|15|prior_interaction_count|Local validated-event count analogue; event eligibility semantics differ|
|16|prior_target_eligible_count|Local scored-event count analogue|
|17|prior_full_correct_count|Local correct-response count analogue|
|18|prior_incorrect_count|Local incorrect-response count analogue|
|19|prior_correct_rate|Local observed-accuracy analogue|
|20|prior_valid_rt_count|Excluded by RT-OFF|
|21|prior_mean_ms_first_response|Excluded by RT-OFF|
|22|prior_mean_attempt_count|Local value would reflect one-response design; domain difference|
|23|prior_skill_exposure_count|Local per-skill count; public skill equivalence missing|
|24|prior_skill_target_eligible_count|Local per-skill scored count; identity equivalence missing|
|25|prior_skill_full_correct_count|Local per-skill correct count; identity equivalence missing|
|26|prior_skill_correct_rate|Local per-skill accuracy; identity equivalence missing|
|27|prior_same_problem_exposure_count|Generated local question identities differ from public problem identities|
|28|prior_same_assistment_exposure_count|No local ASSISTment identity|
|29|prior_same_problem_type_exposure_count|Local game exposure analogue; task-type equivalence missing|
|30|lag2_target_first_attempt_correct|Local chronological binary-outcome analogue|
|31|lag3_target_first_attempt_correct|Local chronological binary-outcome analogue|
|32|lag4_target_first_attempt_correct|Local chronological binary-outcome analogue|
|33|lag5_target_first_attempt_correct|Local chronological binary-outcome analogue|
|34|recent3_target_eligible_count|Local recent scored-window analogue|
|35|recent3_full_correct_count|Local recent correct-count analogue|
|36|recent3_correct_rate|Local recent rate analogue|
|37|recent5_target_eligible_count|Local recent scored-window analogue|
|38|recent5_full_correct_count|Local recent correct-count analogue|
|39|recent5_correct_rate|Local recent rate analogue|
|40|interactions_since_last_full_correct|Local recency/count analogue|
|41|previous_same_skill_correct_raw|Local per-skill prior outcome; public skill equivalence missing|
|42|time_since_previous_skill_seconds|Local timestamp/skill analogue; retained by RT-OFF|

The target is binary first-attempt correctness for an observed ASSISTments task. This resembles local correctness structurally, but different learners, language, curriculum, task selection, motor responses and validation mechanisms produce domain shift. It does not establish the potential outcome of selecting a different task or the ranking benefit of a recommendation.

## Implemented boundary

Local candidate adaptation fails explicitly, including when someone merely supplies two public-ID-looking strings. No missing public identifiers are filled with invented values. Legitimate source missing values in the research observations are handled by the original saved imputation/encoding pipeline.

Research replay reads a deterministic bounded set of eight saved test observations, aligns their actual history rows by problem_log_id, orders them within learners, and scores only the declared raw features. The target is revealed in the trace after prediction and decision. Flipping the current targets must leave probabilities unchanged. All fresh probabilities must match saved references within 2e-6.

The replay policy genuinely uses probability to choose an illustrative support action; it cannot choose an unobserved local game. Its thresholds require future evaluation. This is the maximum supported operational link without fabricating candidate feature vectors.

## Minimum requirements for learned local recommendations

1. An approved local task/context representation with evidence of compatibility with the frozen input schema, including skill/template/problem-type exposure identity and platform-specific context. A label-only skill crosswalk is insufficient.
2. A versioned history adapter reproducing target eligibility, missing values, strict-prior timestamp groups, exposures and recency semantics; complete dimension/category checks are necessary but not sufficient.
3. Representative local candidate-response observations for Grades 1–5 and the robot/UI setting, with leakage-safe evaluation of discrimination, calibration, subgroup performance and abstention.
4. Prospective or otherwise justified policy evaluation of coverage, difficulty and support effects. Correctness prediction alone does not establish instructional benefit or clinical validity.

If a valid representation cannot be established for this frozen model, continue the explicit fallback. A future locally trained model would be a separate authorized research task; this PP1 implementation does not retrain anything.
