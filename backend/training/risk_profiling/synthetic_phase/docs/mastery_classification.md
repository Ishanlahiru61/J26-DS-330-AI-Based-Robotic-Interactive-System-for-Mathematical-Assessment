# Mastery classification (proposal objectives 3–5)

The proposal's main task is to classify each child into three mastery tiers: **High Performer**,
**Progressing** and **Needs Urgent Assistance**. These are educational levels, not diagnoses. Four parts were
added on top of the dyscalculia-risk work, each with its own script, results file and tests.
On synthetic data the selected **SVM** reaches test macro-F1 0.48 and catches 68% of children needing urgent
help (accuracy-rule baseline: 41%). The proposed stacking ensemble did not improve on single models.

## 1. Labels and features (`src/mastery_data.py`, `results/mastery_labels.json`)

**Labels (proposal 3.5.1).** The rubric must be independent of the model's inputs, or the model simply relearns
the rule. In the synthetic phase the rubric is applied to each child's **true, hidden ability**, which stands
in for an independent teacher assessment. The models only see the noisy session data. In the pilot, the label is
the teacher's rating on the NIE 3-level mastery scale.

| Tier | Rule on true ability (mean of 4 accuracy z-scores, standardised; positive = worse) | NIE mastery scale | Share |
|---|---|---|---|
| Needs Urgent Assistance | ≥ 1.0 (worst 16%) | Needs further help | 15% |
| Progressing | between −1.0 and 1.0 | Shows difficulty | 69% |
| High Performer | ≤ −1.0 (best 16%) | Does it successfully | 15% |

The ±1.0 cut-offs reuse the 16th-percentile criterion of the meta-analysis (register F1) symmetrically; this is
a design decision, not a published mastery standard.

**Features (proposal 2.2, 3.3, 3.4).** 17 z-scores against same grade and age, in four nested groups:

| Group | Features added |
|---|---|
| Accuracy only | Overall accuracy |
| Performance | Topic-wise (C2 area) accuracy, error severity |
| Multimodal | Topic-wise speed, attempts, speed variability, robot-demo rate, finger-counting rate |
| Historical | Session-to-session trends in accuracy, speed and attempts |

Hint dependency is not included because no component supplies hints (register B1b).

## 2. Models and stacking (`src/mastery_models.py`, `results/mastery_models.json`)

All models were tuned by stratified 5-fold cross-validation on training children (macro-F1), with class
weights for the imbalance, and compared on 445 validation children.

| Model | Validation macro-F1 | Recall: High | Recall: Progressing | Recall: Needs Urgent |
|---|---|---|---|---|
| Accuracy rule (baseline) | 0.47 | 0.28 | 0.72 | 0.43 |
| Random Forest | 0.51 | 0.52 | 0.58 | 0.65 |
| XGBoost | **0.52** | 0.40 | 0.68 | 0.59 |
| **SVM (selected)** | 0.49 | 0.65 | 0.48 | 0.70 |
| Stacking: RF + SVM → XGBoost meta-learner | 0.45 | 0.70 | 0.38 | **0.73** |

**Stacking (proposal 3.5.2–3.5.3)** was built as proposed: RF and SVM produce out-of-fold class probabilities
(5-fold CV of the training set), an XGBoost meta-learner is trained on them, and the bases are refitted on all
training children. Its internal CV score (0.59) was optimistic; **nested cross-validation**, which refits the
whole stack in each fold, gave 0.48, below RF and XGBoost (about 0.54). So stacking does **not** justify its extra
complexity here, which is the check the proposal itself requires (3.5.4).

**Selection.** Among models not reliably worse than the best on **both** macro-F1 and Needs-Urgent recall
(paired bootstrap, 1,000 resamples), the simplest was chosen: **SVM**. The recall criterion is the proposal's own
requirement (3.6.2). It was added after a first run whose macro-F1-only rule chose the accuracy rule, which
misses most children needing help; both results are kept in the results file.

**Corrections made during development (documented for transparency).**
1. SVM class decisions first came from its Platt probabilities, which ignore class weights; fixed to use the
   class-weighted decision.
2. Stacking first corrected class imbalance twice (bases and meta-learner); the meta-learner's weighting is now tuned.
3. A first version of the script scored the accuracy rule (no trained parameters) on the test split. Nothing from
   that run was used to fit any model; test evaluation was then moved to a separate final step.

## 3. Ablation (`src/mastery_ablation.py`, `results/mastery_ablation.json`)

| Feature set (features) | SVM: train CV / validation macro-F1 | XGBoost: train CV / validation |
|---|---|---|
| Accuracy only (1) | 0.42 / 0.43 | 0.43 / 0.42 |
| Performance (6) | 0.50 / 0.46 | 0.51 / 0.46 |
| Multimodal (14) | 0.51 / 0.49 | 0.54 / 0.50 |
| Historical (17) | 0.51 / 0.49 | 0.54 / 0.52 |

Multimodal information improves mastery classification over accuracy alone by about 0.07–0.11 macro-F1.
Historical trends add nothing **in synthetic data**, because simulated children do not learn between sessions;
this must be re-tested with real multi-session data. Single steps on validation are within bootstrap noise.

## 4. Explanations (`src/mastery_explain.py`, `results/mastery_explanations.json`)

Target: the probability of **Needs Urgent Assistance**.

- **SHAP (global).** Largest contributions: number-relationships accuracy, attempts, number-sense accuracy,
  number-sense speed, arithmetic accuracy.
- **Stability across folds (proposal 3.8.1).** Refitting on each of 5 training folds gave mean pairwise Spearman
  0.88 (minimum 0.83) for the feature ranking; 4 of the top 5 features were the same in every fold.
- **LIME (local).** For one correctly classified child per tier, LIME contributions (local slope × the child's
  standardised value) agree with SHAP: Spearman 0.70, 0.84 and 0.83.
- **Direction check.** Worse accuracy and more attempts push toward Needs Urgent Assistance (Spearman ≈ 0.95).
  The SVM also uses features that carry no real signal in the simulation (finger counting, accuracy trend), and
  faster correct answers push toward "urgent". These are artefacts of the synthetic data and a reason why real
  data are needed before the explanations can be trusted.

Explanations are associations with the model's output, not causes (proposal 1.2.5).

## 5. Final test (`src/mastery_final.py`, `results/mastery_final_evaluation.json`)

The selected SVM was evaluated once on 436 test children (73 / 300 / 63 per tier).

| Model | Accuracy | Macro-F1 | Recall: High / Progressing / Needs Urgent | Needs Urgent recall 95% CI | ROC-AUC (one-vs-rest) |
|---|---|---|---|---|---|
| **SVM** | 0.51 | 0.48 | 0.53 / 0.47 / **0.68** | 0.56–0.78 | 0.71 |
| Accuracy rule | 0.62 | 0.48 | 0.27 / 0.75 / 0.41 | 0.30–0.54 | — |

Confusion matrix (rows true, columns predicted: High, Progressing, Urgent): High 39 / 27 / 7;
Progressing 89 / 140 / 71; Urgent 5 / 15 / 43.

By grade, macro-F1 was 0.48 (Grade 1), 0.52 (Grade 2) and 0.40 (Grade 3); Grade 3 is again the weakest, as in
the risk model. The SVM has the same macro-F1 as the baseline but catches far more children who need urgent
help, at the cost of more "Progressing" children being flagged (a screening trade-off, as in 4-S).

## Limitations

1. Labels come from simulated true ability; real labels need a teacher's NIE mastery rating.
2. The ±1.0 tier cut-offs are a design decision; they should be reviewed by an educational expert.
3. Historical features cannot be evaluated on simulated data without learning.
4. Dynamic profiling (proposal 3.7) and the dashboard are not yet implemented.
