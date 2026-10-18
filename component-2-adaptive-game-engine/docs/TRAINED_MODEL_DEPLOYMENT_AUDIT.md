# Trained Model Deployment Audit

Date: 17 October 2026

## Conclusion

`H2_GB_RT_OFF_v1` remains the research winner. Its retained Platt-calibrated held-out test performance is ROC-AUC 0.750158, log loss 0.534442 and Brier score 0.178955 on 937,043 observations. It cannot be used directly for local cube-game candidates: its contract includes ASSISTments skill/problem/template/tutor/position/platform context and repeated-problem/assistment features for which no evidence-based local mapping exists.

Among the original eight, `H0_LR_RT_OFF_v1` is the most portable artifact and simplest schema (12 RT-off inputs; linear model; native calibration retained), but it is **not directly deployable**. H0 still requires current `skill_id`, `problem_type`, `tutor_mode`, `original`, `position`, `template_id`, prior ASSISTments skill/problem-type identity and elapsed platform time. Choosing it solely for convenience would sacrifice measured performance without solving the scientific mismatch.

The implemented pathway is therefore Option B: `TRANSFER_H2_GB_RT_OFF_v1`, a newly named XGBoost classifier trained on a reduced numeric schema shared by the frozen ASSISTments history and the local service. The frozen eight models and their conclusions remain unchanged.

## Sources inspected

- frozen Step 2 pipeline, chronology report, validation artifacts and 6,123,270-row history parquet;
- original train/validation/test learner partitions;
- Step 3 training/finalization scripts and notebook;
- experiment registry, feature contract, model results, calibration decisions, input provenance, feature importance and independent validation;
- all eight saved model artifacts, the shared train-only preprocessor and both retained GB calibration objects;
- current local catalog, history state, curriculum filters, progression rules, C1/C4 contracts and service tests.

No original research artifact was modified.

## Original eight configurations

All results below are native held-out test results unless noted. Each split is learner-disjoint (train 4,259,096; validation 921,808; test 937,043 rows).

| Model | History | Algorithm | RT | Inputs | ROC-AUC | Log loss | Brier | Calibration | Local deployment assessment |
|---|---:|---|---|---:|---:|---:|---:|---|---|
| H0_LR_RT_OFF | H0 | SGD logistic | Off | 12 | 0.664176 | 0.591088 | 0.201599 | native retained; Platt rejected | Most portable original, but invalid platform/ID fields remain |
| H0_LR_RT_ON | H0 | SGD logistic | On | 14 | 0.663709 | 0.590826 | 0.201424 | native retained; Platt rejected | Adds unsupported RT semantics; no benefit |
| H1_LR_RT_OFF | H1 | SGD logistic | Off | 25 | 0.704559 | 0.569530 | 0.192496 | native | Better history; still includes ID/platform/repeated-task fields |
| H1_LR_RT_ON | H1 | SGD logistic | On | 29 | 0.703898 | 0.569838 | 0.192580 | native | Adds RT and performs slightly worse |
| H2_LR_RT_OFF | H2 | SGD logistic | Off | 38 | 0.713327 | 0.562612 | 0.189680 | native | Strongest original LR; incompatible candidate context remains |
| H2_LR_RT_ON | H2 | SGD logistic | On | 42 | 0.713158 | 0.562774 | 0.189732 | native | Adds unsupported RT with no improvement |
| H2_GB_RT_OFF | H2 | XGBoost | Off | 38 | 0.750158 | 0.534998 | 0.179078 | Platt retained: log loss 0.534442, Brier 0.178955 | Research winner; not directly transferable |
| H2_GB_RT_ON | H2 | XGBoost | On | 42 | 0.749632 | 0.535337 | 0.179220 | Platt retained: log loss 0.534814, Brier 0.179109 | Slightly worse and needs RT context |

RT-OFF is preferable for this local contract because response time is optional and the original comparison found no predictive gain from RT-ON. The original shared preprocessor performs train-fitted categorical grouping/one-hot encoding, numeric imputation/scaling, frequency encoding and robust RT transformation. The GB winners additionally require their validation-fit Platt objects.

## Complete feature compatibility matrix

Legend: A = directly reproducible with matching prior-only meaning; B = defensible transformation; C = unsupported/not transferable. “Used by” is H0/H1/H2, with RT-ON where applicable.

| Feature | Used by | Class | Local treatment and rationale |
|---|---|---:|---|
| skill_id | H0/H1/H2 | C | ASSISTments taxonomy identifier; local game skill must not be assigned an invented ID |
| problem_type | H0/H1/H2 | C | ASSISTments response/widget type is not a cube-game type |
| tutor_mode | H0/H1/H2 | C | ASSISTments tutor behavior is absent locally |
| original | H0/H1/H2 | C | ASSISTments content-origin flag has no local equivalent |
| position | H0/H1/H2 | C | ASSISTments assignment position is not local chronological order |
| template_id | H0/H1/H2 | C | ASSISTments template identifier cannot be invented for games |
| previous_correct_raw | H0/H1/H2 | A | Previous valid first-attempt binary correctness |
| previous_attempt_count | H0/H1/H2 | A | C1/local contract records attempts; operational scored events are first attempt |
| previous_ms_first_response | RT-ON | A, optional | Reproducible only when positive response time is supplied; excluded from deployment schema |
| previous_skill_id | H0/H1/H2 | C | Identifier mismatch; not replaced by a fabricated local ID |
| previous_problem_type | H0/H1/H2 | C | Platform widget context is absent |
| time_since_previous_seconds | H0/H1/H2 | A | Reproducible from prior event timestamps, but excluded from v1 to minimize clock/domain shift |
| has_strict_prior_interaction | H0/H1/H2 | A | Derived before the candidate response |
| previous_interaction_rt_valid | RT-ON | A, optional | Reproducible when RT is present; excluded with all RT fields |
| prior_interaction_count | H1/H2 | A | Count of earlier valid events |
| prior_target_eligible_count | H1/H2 | B | Local valid first-attempt binary events use the same operational meaning |
| prior_full_correct_count | H1/H2 | A | Earlier correct binary events |
| prior_incorrect_count | H1/H2 | A | Earlier incorrect binary events |
| prior_correct_rate | H1/H2 | A | Earlier correct / eligible count; undefined remains missing |
| prior_valid_rt_count | RT-ON H1/H2 | A, optional | Reproducible but excluded |
| prior_mean_ms_first_response | RT-ON H1/H2 | B, optional | Units reproduce, but platform timing process differs; excluded |
| prior_mean_attempt_count | H1/H2 | A | Earlier mean attempts; scored local events are first-attempt observations |
| prior_skill_exposure_count | H1/H2 | B | Computed for the candidate's named local operational skill, without mapping to ASSISTments IDs |
| prior_skill_target_eligible_count | H1/H2 | B | Same-skill earlier valid binary events |
| prior_skill_full_correct_count | H1/H2 | B | Same-skill earlier correct events |
| prior_skill_correct_rate | H1/H2 | B | Same-skill earlier accuracy; cross-domain semantic assumption documented |
| prior_same_problem_exposure_count | H1/H2 | C | No equivalence between a generated local question and ASSISTments problem ID |
| prior_same_assistment_exposure_count | H1/H2 | C | Assistment container does not exist locally |
| prior_same_problem_type_exposure_count | H1/H2 | C | Widget/problem-type exposure is not a local game exposure |
| lag2_target_first_attempt_correct | H2 | A | Second strictly prior valid response |
| lag3_target_first_attempt_correct | H2 | A | Third strictly prior valid response |
| lag4_target_first_attempt_correct | H2 | A | Fourth strictly prior valid response |
| lag5_target_first_attempt_correct | H2 | A | Fifth strictly prior valid response |
| recent3_target_eligible_count | H2 | A | Count in strictly prior three-event window |
| recent3_full_correct_count | H2 | A | Correct count in strictly prior three-event window |
| recent3_correct_rate | H2 | A | Prior-only recent accuracy |
| recent5_target_eligible_count | H2 | A | Count in strictly prior five-event window |
| recent5_full_correct_count | H2 | A | Correct count in strictly prior five-event window |
| recent5_correct_rate | H2 | A | Prior-only recent accuracy |
| interactions_since_last_full_correct | H2 | A | Prior interactions since last correct; missing when none exists |
| previous_same_skill_correct_raw | H2 | B | Previous correctness in candidate's local operational skill |
| time_since_previous_skill_seconds | H2 | B | Reproducible by skill/timestamp, but excluded from v1 to reduce domain shift |

No current candidate outcome, future response, synthetic ID, arbitrary default, grade proxy, or fabricated tier-difficulty value is used.

## Why direct H2 GB transfer is invalid

The frozen preprocessor can technically ignore unseen categories and impute missing numeric values, but that is not scientific compatibility. Supplying local names as unseen ASSISTments categories would collapse informative current skill/problem context; supplying zeros or fake IDs would create undocumented positions in the training distribution. H2 also uses repeated problem, assistment and problem-type exposures. A successful `predict_proba` call under those substitutions would demonstrate software execution only—not a valid cube-game prediction.

## Deployment-compatible model

`TRANSFER_H2_GB_RT_OFF_v1` uses 25 numeric features: prior outcome/attempt context, global prior counts/rates, compact lags/recent windows, candidate-skill prior counts/rate, and previous same-skill correctness. It excludes current outcomes, all identifiers, response time, platform context, repeated problem/assistment fields, grade and tier.

- Algorithm: XGBoost binary classifier, 350-tree ceiling with validation early stopping.
- Split: original frozen learner-disjoint partitions.
- Training: deterministic 1,200,000-row within-train sample.
- Validation: deterministic 300,000-row within-validation sample.
- Test: complete 937,043-row frozen test partition.
- Native test: ROC-AUC 0.712077; log loss 0.563200; Brier 0.190298.
- Platt attempt: test log loss 0.563208 and Brier 0.190301; rejected by the predeclared validation improvement rule. Native probability is used.
- Beta-Bernoulli common test subset (316,502 rows with at least two same-skill priors; 33.78% support): ROC-AUC 0.679452; log loss 0.589205; Brier 0.201510.

The model's lower performance than the research winner is the measured cost of removing non-transferable context. Its better held-out metrics than the Beta baseline are evidence for using it as primary **within the ASSISTments domain**, not proof of local superiority.

## Live prediction and recommendation policy

The service scores every curriculum-eligible candidate skill after at least one valid prior response. Curriculum, prerequisites, hardware safety and coverage define the admissible set. Within equally under-covered games, the policy selects the valid prediction closest to the explicit 0.65 challenge target. For the current skill, predictions below 0.40 support bounded Tier A reinforcement; predictions at or above 0.75 may support Tier B only when the independent readiness rule is also satisfied. Thus model output changes both game ranking and difficulty action but cannot override safety or coverage.

At zero history, artifact failure, timestamp invalidity or inference failure, the trained predictor abstains. `LOCAL_BETA_BERNOULLI_v1` then provides `(correct + 1) / (attempts + 2)` only after two same-skill observations; otherwise the deterministic rule fallback selects a safe activity. Every result records source, version, validity, assumptions and abstention reason.

## Evidence boundary

Held-out metrics are retrospective ASSISTments predictive evidence. The end-to-end strong/struggling runs are deterministic functional simulations. There is no real-child local dataset, prospective calibration, clinical validation, fairness evaluation, or evidence that 0.40/0.65/0.75 are educationally optimal. C4 receives descriptive evidence only; no dyscalculia risk or diagnosis is produced.
