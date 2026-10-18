# Adaptive ML feasibility audit

> Historical 16 October audit. Superseded on 17 October 2026 by `TRAINED_MODEL_DEPLOYMENT_AUDIT.md`, which audits all eight models and documents the implemented transferable trained predictor. The Beta-Bernoulli-only conclusion below is retained as provenance, not the current architecture.

## Baseline inspected

The authoritative baseline was extracted from component-2-adaptive-game-engine-github-16-oct-11PM.zip into a new refinement workspace. The ZIP already contained SQLite history, Grade 1–5 curriculum filtering, eight games/two tiers, coverage-first selection, C1/C4 contracts, a frozen XGBoost scorer, research replay and tests. Its live service always emitted rule fallback with null probability. The archived streamlit_app.py was zero bytes, explaining the baseline UI failures despite a stale verification report claiming 60 passes.

The original ZIP and all external research artifacts remain unchanged. The archived Python cache was deliberately excluded from the working copy.

## Frozen model compatibility decision

H2_GB_RT_OFF_v1 predicts calibrated binary first-attempt correctness for a represented ASSISTments observation. Its frozen pipeline requires 42 ordered H2 raw fields; four response-time-history fields are excluded to produce the model's 335 transformed inputs. History fields include global, recent and same-skill evidence. Candidate context includes ASSISTments skill_id, template_id, problem/problem-type exposure, tutor_mode, original and position semantics.

The local cube games have correctness, timestamps and skill labels, but no validated equivalence for ASSISTments skill/template/problem/assistment identity or platform context. A label crosswalk or invented ID would not preserve the train-time representation. The frozen model therefore cannot distinguish local Counting, Comparison, Ordering, Place Value or operation candidates defensibly. It remains hash-verified and tested for the public research domain but does not score live local games.

## Smallest live model path

With no labeled local child dataset, a new supervised local XGBoost model would be unjustified. The implemented live prototype uses LOCAL_BETA_BERNOULLI_v1, a transparent online per-child, per-operational-skill learner-state model:

    posterior success estimate = (correct + 1) / (valid attempts + 2)

This is a Beta(1,1)-Bernoulli posterior mean. It uses only the current child's strictly prior valid first responses. It abstains before two observations and reports its evidence count and posterior standard deviation. It is compatible with the local event semantics by construction, but it has no real-child calibration or recommendation-outcome validation. Outputs are labelled EXPERIMENTAL_MODEL_ASSISTED, never H2_GB_RT_OFF_v1.

This model is a probabilistic learner-state estimator, not a trained population classifier. It is the smallest meaningful live model connection available without manufacturing features or training on synthetic labels.

## Integrated decision policy

All thresholds are versioned in config/adaptive_assessment_settings.json:

| Setting | Value | Rationale and boundary |
|---|---:|---|
| Minimum observations per skill | 3 | Preserves the existing coverage evidence level and prevents one answer being treated as sufficient. |
| Minimum observations for an estimate | 2 | Avoids presenting a one-answer posterior as adaptive evidence. |
| Advance threshold | 0.75 | Aligns with the existing Tier B engineering threshold; not a mastery or clinical cutoff. |
| Reinforce threshold | 0.40 | Conservative functional trigger for repeated difficulty; unvalidated educational heuristic. |
| Maximum consecutive same game | 4 | Allows one bounded advance/reinforcement question beyond the three-evidence minimum. |
| Maximum interactions | 48 | Fail-safe of six questions per provisional game; not a psychometric stopping rule. |

After every saved answer, next_task reconstructs strict-prior state and runs one backend pathway. Hard grade, prerequisite, active, Tier B and robot/cube constraints create the eligible set. The adaptive focus policy then returns:

- START for the first under-assessed eligible skill;
- CONTINUE until minimum evidence is obtained;
- ADVANCE to Tier B when the estimate and existing Tier A evidence both support it;
- REINFORCE at Tier A after a low estimate, bounded by the repetition limit;
- SWITCH to the least-assessed eligible area/skill once evidence is sufficient or reinforcement is bounded;
- COMPLETE when all grade-applicable games meet the evidence minimum;
- STOP at the maximum-interaction safeguard.

If the local estimator abstains, the same pathway uses RULE_FALLBACK. Curriculum and coverage always override model preference. The model never bypasses a hard eligibility rule.

## Meaning of the demonstration

The deterministic Grade 3 demonstration uses temporary databases. With all-correct responses, the stronger learner advances where Tier B is grade-eligible and completes eight-game coverage in 29 responses. With repeated incorrect responses, the second learner receives bounded reinforcement and completes coverage in 31 responses. This proves functional history sensitivity, policy branching, coverage and stopping—not predictive accuracy, learning benefit or clinical validity.

## Output and summary

Each recommendation includes learner/session, game, area, skill, tier, decision type, mode, strict-prior counts, eligible/excluded candidates, model provenance/estimate when available, fallback reason and current assessment status. Public tasks exclude expected answers.

The unchanged C2_TO_C4_v2.0.0 export now carries additional allowed descriptive fields: assessment status, minimum-evidence state, areas requiring further assessment and compact recommendation history. It retains counts, accuracy, recent evidence and tier breakdown. It does not compute dyscalculia risk, mastery, diagnosis or C3 behavior.

## Required next research

Before real-child use, obtain appropriately approved local robot/UI interaction data across Grades 1–5; validate item/task representation, discrimination and calibration; examine grade and subgroup behavior; set evidence/advance/reinforcement thresholds with curriculum and measurement expertise; compare policy variants offline where identifiable; and prospectively evaluate usability, safety, coverage and learning/assessment consequences. A future supervised local model must be separately versioned and must not silently replace the frozen research model.
