# Model Selection Justification

## Decision

Select **`H2_GB_RT_OFF_v1` with the retained validation-fitted Platt calibrator** for the frozen ASSISTments research demonstration. Do not use it as a local eight-game ranker without a validated transfer layer.

| Selection criterion | Candidate evidence | Final decision | Justification |
|---|---|---|---|
| History representation | LR RT-OFF test log loss: H0 0.591088, H1 0.569530, H2 0.562612 | H2 | Accumulated and compact ordered history improved probability quality overall |
| Model family | H2 RT-OFF native test log loss: LR 0.562612 vs GB 0.534998; ROC-AUC 0.713327 vs 0.750158 | GB | Nonlinear gain was material enough to justify complexity |
| Response time | H2 GB RT ON worsened test log loss by 0.000339; LR effects were negligible/inconsistent | RT OFF | No practical predictive gain; reduces timing-quality and transfer dependence |
| Calibration | Selected validation ECE 0.010771 triggered Platt; validation rule passed | Retain Platt | Validation-only fit reduced test log loss to 0.534442 and ECE to 0.003847 |
| Complexity control | Candidates within 0.001 of best validation log loss were compared by complexity | Simpler eligible configuration | Frozen selection rule avoids tiny metric differences automatically winning |
| Test isolation | Selection record contains validation log loss/Brier; test metrics evaluated afterward | Accept final test once | Prevents direct test optimization |
| Deployment boundary | No validated local candidate encoding or transfer study | Research-only model | Prevents unsupported robot-game probabilities |

## Why H2

H1 adds cumulative learner, skill and exposure summaries to the H0 current/immediate-history baseline. H2 adds compact order-sensitive lags, recent-three/recent-five rates, recency and same-skill context. The matched LR comparisons show lower overall log loss from H0→H1 and H1→H2. Learner-level paired bootstrap intervals for both reductions excluded zero, although gains were not uniform for the earliest events. H2 is therefore the strongest supported history representation, not a claim that every H2 field is independently causal.

## Why gradient boosting

The regularized SGD logistic model is the transparent baseline. XGBoost can represent nonlinear thresholds and interactions among recent rate, exposure, context and recency. On H2 RT OFF, the GB model reduced native test log loss by 0.027614 and Brier by 0.010603 while increasing ROC-AUC by 0.036831 relative to LR. The equal-learner-weighted log-loss gain was 0.029154 (95% bootstrap interval 0.027794–0.030683). This is large enough to accept a 2.17 MB, less directly interpretable model instead of the 4.6 KB LR artifact.

## Why RT OFF

RT ON added only four strictly historical fields and used robust train-only processing. It was therefore a fair matched ablation, not a different dataset. The effects were too small and inconsistent to justify timing dependence. RT OFF also avoids relying on a noisy public-platform measurement whose meaning may change with language, motor action, robot delay and cube validation.

## Why Platt scaling

The frozen rule attempted calibration only when native validation ECE was at least 0.01. A calibrator was retained only if validation log loss improved by at least 0.0005 and ECE by at least 0.002. Platt scaling fits a logistic mapping from native probability logits to observed validation outcomes. It was fitted on 921,808 validation rows, never test labels. For the selected model it produced a test calibration intercept of −0.000965 and slope of 0.994885, close to ideal 0 and 1.

## Rejected alternatives

- H0/H1 LR: weaker probability quality; retained as scientific baselines.
- H2 LR: simpler and interpretable, but materially weaker than H2 GB.
- RT ON variants: no useful incremental gain.
- Uncalibrated H2 GB RT OFF: same discrimination but slightly worse log loss, Brier and ECE.
- Neural networks/DKT/LSTM: explicitly outside the proposal methodology; unnecessary for answering the controlled history/RT/model-family questions.
- Direct local ranking: unsupported because ASSISTments identifiers and tasks have no validated mapping to the robot catalog.

## What the final model predicts

It estimates the probability that the observed ASSISTments interaction will have exact binary first-attempt correctness, conditional on the current ASSISTments context and strictly prior history. It does not predict the best next game, causal benefit, mastery, Grade 3 attainment, dyscalculia status, clinical risk or referral need.

