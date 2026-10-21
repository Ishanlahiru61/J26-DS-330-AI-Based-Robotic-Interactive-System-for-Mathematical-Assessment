# C2 — Reproducible Evaluation

**Component:** History-Aware Adaptive Game Recommendation Engine  
**Project:** AI-Based Robotic Interactive System for Mathematical Assessment and Early Dyscalculia Screening Support  
**Project ID:** J26-DS-330

## 1. Overview

This folder contains the reproducible evaluation of Component 2 (C2), which recommends the next mathematical activity and difficulty level using learner interaction history.

C2 follows a **hybrid recommendation approach**:

1. **XGBoost** estimates mathematical response-success probabilities using historical learner information.
2. **Recommendation policy** selects an appropriate next activity using predicted probabilities, curriculum eligibility, skill coverage, prerequisites and progression requirements.
3. **Safety and fallback mechanisms** handle insufficient history, unavailable predictions and assessment stopping conditions.

The evaluation covers three areas:

- **Predictive-model performance:** How effectively the trained model predicts correctness.
- **Recommendation-engine behavior:** How learner history influences activity ranking and selection while respecting constraints.
- **End-to-end reliability:** Whether the component completes synthetic assessments and produces valid evidence for downstream integration.

The evaluation uses the frozen C2 implementation. It does not retrain, replace or modify the deployed XGBoost model or production recommendation policy.

---

## 2. Final C2 Evaluation Scorecard

The recommendation engine was evaluated using **120 matched pairs of synthetic learner histories, 20 invariant control pairs and 25 simulated assessment sessions**, covering Grades 1–5.

### 2.1 Recommendation behavior and reliability

| Metric | Verified result | Explanation |
|---|---|---|
| **XGBoost Score Sensitivity** | **92.50% (111/120)** | Changing learner history changed at least one common candidate's predicted success score. |
| **Candidate-Ranking Sensitivity** | **62.50% (75/120)** | Different learner histories changed the ordering of candidate activities. |
| **Final-Activity Sensitivity** | **18.33% (22/120)** | Different histories led to different final activities after applying recommendation constraints. |
| **Curriculum Compliance Rate** | **100% (280/280)** | All controlled decisions followed the implemented grade and curriculum eligibility rules. |
| **Tier-Policy Compliance** | **100% (280/280)** | All controlled decisions followed the implemented Tier A/B progression rules. |
| **Invariant Stability** | **100% (20/20)** | Equivalent learner histories produced identical observable decisions. |
| **Policy-Expectation Agreement** | **94.76% (398/420)** | The engine satisfied predefined internal expectations covering prediction direction, coverage and stability. |
| **Synthetic Session Completion** | **100% (25/25)** | All expanded simulated assessments completed under the configured assessment rules. |
| **Mean Skill Coverage** | **100%** | All applicable targeted mathematical skills were covered in the expanded simulated sessions. |

### 2.2 Overall evaluation conclusion

**There is no single overall recommendation accuracy percentage established by this evaluation.**

The main findings are:

- **History responsiveness:** XGBoost scores changed in 92.50% of matched comparisons, while candidate rankings changed in 62.50%.
- **Controlled adaptation:** Final selected activities changed in 18.33% of comparisons; curriculum, coverage and progression constraints limited some changes.
- **Policy adherence:** Curriculum and tier-policy compliance were both 100% across 280 controlled decisions.
- **Stability:** All 20 equivalent-history control pairs produced consistent decisions.
- **Operational reliability:** All 25 expanded synthetic assessments completed with 100% mean skill coverage.
- **Internal expectation agreement:** The predefined specification checks achieved 94.76% overall agreement.

The results demonstrate that **C2 responds to learner history at the prediction and ranking levels while maintaining curriculum-constrained activity selection**.

They do not establish that the recommended activities improve actual children's learning outcomes.

### 2.3 Understanding the 18.33% activity-change rate

Final-activity sensitivity measures how often two different learner histories produce different final recommended activities.

In the expanded evaluation:

- 111/120 pairs changed at least one candidate's prediction.
- 75/120 pairs changed candidate rankings.
- 22/120 pairs changed the final selected activity.

A predicted-probability change does not necessarily require a different mathematical activity.

The policy may continue selecting the same activity when:

- A required skill has not received sufficient assessment coverage.
- The learner must complete foundation activities.
- Curriculum eligibility or prerequisites restrict candidate selection.
- The highest-ranked unconstrained activity is not currently admissible.
- The same activity remains the preferred admissible choice.

Therefore, **18.33% is an activity-change frequency, not recommendation accuracy**. A higher activity-change rate is not automatically better.

Among 96 matched-history pairs with identical eligible candidate sets, final activities changed in **17/96 (17.71%)**.

---

## 3. Five Main C2 Recommendation Evaluation Metrics

The following five measures summarize the contribution and operational behavior of the recommendation engine.

| Metric | Definition | Result | Evaluation source | Important limitation |
|---|---|---|---|---|
| **Skill Coverage Rate** | Mean proportion of applicable targeted game skills assessed before stopping | **100% mean** | 15 original hybrid sessions; confirmed in 25 expanded sessions | Coverage is not proof of learning |
| **Curriculum Compliance Rate** | Curriculum-valid selections / evaluated selections | **100% (280/280)** | Expanded controlled evaluation | Encoded rules have not been independently pedagogically validated |
| **History Sensitivity** | Proportion of matched-history comparisons producing changes in predictions, rankings or recommendations | **92.50% score; 62.50% ranking; 18.33% activity** | 120 matched-history pairs | Sensitivity does not directly measure educational recommendation quality |
| **Tier-Policy Compliance** | Tier selections satisfying implemented progression rules / evaluated selections | **100% (280/280)** | Expanded controlled evaluation | Not independent difficulty appropriateness |
| **Session Completion Rate** | Assessments completing required evidence conditions within the configured response limit / evaluated assessments | **100% (25/25)** | Expanded synthetic sessions | Fictional learners rather than actual child participants |

These metrics evaluate different properties and must not be combined into an arbitrary weighted C2 accuracy percentage.

---

## 4. Predictive Model Evaluation

### 4.1 Model comparison

The original research winner and deployed transferable model are distinct trained artifacts.

| Metric | Original research winner | Deployed transferable model |
|---|---:|---:|
| Model | `H2_GB_RT_OFF_v1` | `TRANSFER_H2_GB_RT_OFF_v1` |
| Algorithm | Gradient Boosting / XGBoost | XGBoost |
| **ROC-AUC** | **0.750158** | **0.712077** |
| Average Precision | 0.851203 | Not reported |
| Log Loss | 0.534442 | 0.563200 |
| Brier Score | 0.178955 | 0.190298 |
| ECE-10 | 0.003847 | Not reported |
| Accuracy at threshold 0.5 | 0.734105 | Not reported |
| F1-score | 0.821907 | Not reported |
| Calibration | Platt scaling retained for reported test results | Disabled at runtime |

The original research winner was evaluated on **937,043 held-out ASSISTments interactions from 7,001 learners**.

**Important:** The 0.750158 ROC-AUC belongs to the original research model. The actual deployed transferable model has a stored ROC-AUC of 0.712077.

Both are source-domain results and do not establish local calibration for children using robotic cube-based activities.

### 4.2 Interpretation

- **ROC-AUC:** Measures how well the model ranks correct versus incorrect responses.
- **Average Precision:** Summarizes precision across the precision-recall curve.
- **Accuracy:** Measures correctness classification at a specified probability threshold.
- **Log Loss:** Evaluates prediction probabilities and penalizes confident errors.
- **Brier Score:** Measures average squared probability error.
- **ECE-10:** Estimates calibration error using ten equal-frequency probability bins.
- **F1-score:** Combines classification precision and recall.

These evaluate **correctness prediction**, not whether the final recommended activity is the best educational choice.

The original research experiments evaluated eight model configurations, including Logistic Regression and Gradient Boosting variants with different history and response-time feature settings.

---

## 5. Expanded History-Sensitivity Evaluation

### 5.1 Experimental design

The expanded experiment was specified in `expanded_config.json` and executed using `expanded_evaluation.py`.

| Configuration | Value |
|---|---|
| Grades | 1–5 |
| Assessment stages | Early, middle, late |
| History contrasts | 4 |
| Deterministic replicates per combination | 2 |
| Matched-history pairs | **120** |
| Invariant control pairs | **20** |
| Total pairs | **140** |
| Total evaluated decisions | **280** |
| Random seed | `20261022` |

The four history contrasts were:

1. Strong versus struggling performance.
2. Improving versus declining performance.
3. Skill-specific strength redistribution.
4. Recent-correct versus recent-incorrect history.

All scenarios were generated without examining recommendation outcomes, and no scenario pairs were removed based on the resulting scores.

### 5.2 Expanded results

| Metric | Numerator / denominator | Result |
|---|---:|---:|
| XGBoost score changes | 111/120 | **92.50%** |
| Candidate-ranking changes | 75/120 | **62.50%** |
| Pre-constraint preferred-skill changes | 65/120 | **54.17%** |
| Final-activity changes | 22/120 | **18.33%** |
| Activity changes with identical eligible candidates | 17/96 | **17.71%** |
| Final tier changes | 2/120 | **1.67%** |
| Curriculum compliance | 280/280 | **100%** |
| Tier-policy compliance | 280/280 | **100%** |
| Invariant stability | 20/20 | **100%** |

Mean absolute probability difference between history variants: **0.2467**.

The results indicate that predictions and candidate ordering are responsive to historical performance, while final activity choices remain subject to assessment-policy constraints.

### 5.3 Policy-Expectation Agreement

Policy-Expectation Agreement is a **study-specific internal specification measure**.

| Expectation category | Result |
|---|---:|
| Prediction-direction agreement | 98/120 — 81.67% |
| Coverage/stopping agreement | 280/280 — 100% |
| Invariant agreement | 20/20 — 100% |
| **Overall Policy-Expectation Agreement** | **398/420 — 94.76%** |

The overall percentage is influenced substantially by the 280 coverage/stopping checks.

**It must not be presented as recommendation accuracy, independently rated educational suitability or clinical validity.**

### 5.4 Original controlled experiment

The original evaluation used six matched-history pairs and three invariant pairs.

| Metric | Original result |
|---|---:|
| Model-score changes | 5/6 — 83.33% |
| Final-activity changes | 1/6 — 16.67% |
| Curriculum compliance | 18/18 — 100% |
| Tier-policy compliance | 18/18 — 100% |
| Invariant stability | 3/3 — 100% |

The expanded experiment strengthens the behavioral evaluation while preserving the original results for traceability.

---

## 6. Baseline Comparison

C2 was compared against fixed sequencing and a rule-only adaptive policy under controlled synthetic response profiles.

| System | Completed sessions | Mean valid responses | Mean skill coverage | Curriculum compliance | Tier compliance | ML usage |
|---|---:|---:|---:|---:|---:|---:|
| Fixed sequence | 5/15 | 23.40 | 100% | 100% | 100% | 0% |
| Rule-only | 15/15 | 20.13 | 100% | 100% | 100% | 0% |
| **Hybrid C2** | **15/15** | **19.53** | **100%** | **100%** | **100%** | **94.88%** |

The fixed-sequence baseline follows a deterministic grade-eligible activity order.

The rule-only baseline uses local historical performance estimates without XGBoost.

The hybrid system uses the deployed predictor with the actual curriculum, coverage and progression policies.

**Interpretation:** The hybrid system completed all 15 controlled sessions with slightly fewer mean valid responses than the rule-only baseline.

**Limitation:** Synthetic correctness was programmed independently of the selected activity. These comparisons do not demonstrate improved learning outcomes, improved screening effectiveness or causal recommendation utility.

---

## 7. End-to-End Simulation Evidence

The original preserved simulation evaluated 25 fictional learners using the complete C2 assessment workflow.

| Metric | Result |
|---|---:|
| Fictional learners | 25 |
| Total scored interactions | 489 |
| XGBoost-assisted recommendations | 464 |
| Cold-start fallbacks | 25 |
| Completed assessments | **25/25** |
| Curriculum-valid selections | **489/489** |
| Valid C4 evidence exports | **25/25** |
| Valid responses per completed session | 16–23 |

All 25 programmed assessment sessions completed without recorded curriculum violations.

These tests verify system operation, safe fallback, assessment stopping and C4-compatible evidence generation under synthetic inputs.

They do not demonstrate improved mathematical learning or clinically valid dyscalculia screening.

---

## 8. Reproducing the Results

### 8.1 Evaluation files

The evaluation folder contains:

| File | Purpose |
|---|---|
| `README.md` | Metric definitions, results and instructions |
| `requirements-evaluation.txt` | Evaluation-specific Python dependencies |
| `evaluation_runner.py` | Complete original evaluation and combined scorecard |
| `evaluation_metrics.py` | Shared metric calculations |
| `expanded_evaluation.py` | Expanded 120-pair history evaluation |
| `scenario_definitions.json` | Original controlled scenarios |
| `expanded_config.json` | Expanded experiment design and random seed |
| `results_summary.csv` | Consolidated machine-readable metrics |
| `results_manifest.json` | Evidence provenance, hashes and results |
| `01_Predictive_Model_Evaluation.ipynb` | Predictive-model evaluation |
| `02_Recommendation_Policy_Evaluation.ipynb` | Recommendation-policy and expanded-history evaluation |
| `03_End_to_End_C2_Evaluation.ipynb` | End-to-end simulation and scorecard |

### 8.2 Running the evaluation

Open a terminal inside:

`backend/training/adaptive_questions/evaluation/`

Install evaluation-only dependencies if required:

```powershell
python -m pip install -r requirements-evaluation.txt
```

Run the expanded recommendation evaluation:

```powershell
python expanded_evaluation.py
```

Run the complete evaluator:

```powershell
python evaluation_runner.py
```

The default evaluation uses the repository-relative backend and can verify tracked experiment results and model hashes without requiring the original large ASSISTments dataset.

### 8.3 Recomputing historical prediction-level metrics

For full historical prediction-row recomputation and original synthetic simulation reconciliation, supply the external evidence paths:

```powershell
python evaluation_runner.py `
  --historical-root "<external ASSISTments evidence root>" `
  --research-root "<external Research Documents root>"
```

The historical research directory must contain:

- `03_Experimental_Modeling/data/prepared/predictions_H2_GB_RT_OFF_test_platt_retained.parquet`
- `03_Experimental_Modeling/data/prepared/predictions_H2_GB_RT_OFF_test_native.parquet`
- `03_Experimental_Modeling/data/prepared/test_supervised_rows.parquet`

These large files are intentionally not included in the evaluation folder or GitHub repository.

Without them, the evaluator verifies tracked aggregate model results, experiment records, model integrity and recommendation behavior while marking prediction-row recomputation as unavailable.

### 8.4 Evaluation notebooks

**01 — Predictive Model Evaluation**

Shows XGBoost research/deployment identity, predictive metrics, calibration evidence and model integrity checks.

**02 — Recommendation Policy Evaluation**

Shows the original six-pair evaluation, expanded 120-pair history experiment, recommendation behavior, safety compliance and baseline comparison.

**03 — End-to-End C2 Evaluation**

Shows completion rates, skill coverage, synthetic integration evidence and the consolidated final scorecard.

The notebooks use shared evaluation functions rather than substituting hardcoded results for real execution.

---

## 9. PP1 Presentation and Viva Reference

### 9.1 What to show during PP1

For a short demonstration:

1. Open this README and show **Section 2 — Final C2 Evaluation Scorecard**.
2. Open `02_Recommendation_Policy_Evaluation.ipynb` to explain the history-sensitivity experiment.
3. Show `expanded_config.json` to demonstrate the declared test design.
4. Run `python expanded_evaluation.py` if the examiner requests reproducibility.
5. Show `01_Predictive_Model_Evaluation.ipynb` only when discussing XGBoost predictive results.

### 9.2 Short evaluation explanation

> My C2 recommendation engine uses XGBoost predictions and curriculum-constrained decision rules. I evaluated it using 120 matched pairs of synthetic learner histories and 20 invariant controls across Grades 1–5. Model predictions changed in 92.5% of comparisons, candidate rankings changed in 62.5%, and final recommended activities changed in 18.3%. All 280 controlled decisions complied with the implemented curriculum and tier policies, and all 25 expanded simulated assessments completed. These findings establish history-responsive prototype behavior and internal policy compliance, while real-child educational effectiveness remains to be evaluated.

### 9.3 Common viva questions

**Why is the final-activity change rate only 18.33%?**

The engine considers predicted success probabilities, but curriculum coverage and assessment requirements constrain which activity can be selected. Changing a probability does not always require changing the recommended activity.

**Why not use Precision@K, Recall@K or NDCG@K?**

C2 produces one next activity and currently lacks independent relevance labels for alternative recommendations. These ranking metrics cannot yet be estimated meaningfully.

**Is Policy-Expectation Agreement the overall recommendation accuracy?**

No. It measures agreement with the implementation's predefined behavioral expectations, not independently verified recommendation correctness.

**Does 100% curriculum compliance prove educational appropriateness?**

No. It verifies that the selected activities followed encoded grade and curriculum rules. Independent curriculum or expert validation is still required.

**Does XGBoost choose the final activity?**

No. XGBoost estimates success probabilities; the recommendation policy chooses the final game and difficulty using those scores and assessment constraints.

**How do you handle learners without sufficient history?**

C2 uses a safe rule-based fallback or abstention pathway when trained inference is unavailable or unsupported.

---

## 10. Evidence Boundaries and Future Evaluation

This is a prototype-stage engineering and recommendation-behavior evaluation.

Current limitations:

- Learner histories and assessment outcomes are synthetic.
- Curriculum and tier metrics measure adherence to encoded rules.
- XGBoost was trained using source-domain ASSISTments interactions.
- The deployed transferable model is not calibrated using local child data.
- Independent relevance labels for ideal next activities are unavailable.
- Randomized learner outcomes and counterfactual outcomes for unselected activities are unavailable.
- No improved learning, screening accuracy or clinical validity has been demonstrated.

Future validation should include qualified expert ratings of activity and difficulty appropriateness, real-user usability studies, and properly designed comparisons between the hybrid recommender and suitable rule-only baselines.

**The current research contribution is a reproducibly evaluated, history-aware, curriculum-constrained adaptive mathematical assessment recommendation engine.**