# C2 — Quick Reference

**Component:** History-Aware Adaptive Game Recommendation Engine  
**Project:** J26-DS-330 — AI-Based Robotic Interactive System for Mathematical Assessment and Early Dyscalculia Screening Support

## 1. What C2 does

C2 recommends the **next mathematical activity and difficulty tier** using a learner's earlier correct/incorrect responses and skill-specific interaction history. It supports mathematical **assessment and early dyscalculia screening**, not diagnosis.

**Decision flow:** Learner grade and history → grade/curriculum-eligible candidates → XGBoost success-probability estimates (when supported) → skill coverage, prerequisites and Tier A/B progression rules → selected activity and tier → validated response saved for the next decision and downstream C4 evidence. When trained inference is unavailable or history is insufficient, use the documented rule-based fallback or abstain.

**Critical point:** XGBoost **does not directly choose the final game**; the constrained recommendation policy does.

## 2. Evaluation scorecard

The expanded experiment used **120 matched synthetic-history pairs**, **20 invariant-control pairs** and **25 synthetic assessment sessions**, across Grades 1–5.

| Metric | Verified result | Meaning |
|---|---:|---|
| XGBoost score sensitivity | **111/120 (92.50%)** | A history change altered at least one common candidate-skill predicted score |
| Candidate-ranking sensitivity | **75/120 (62.50%)** | Candidate order changed with history |
| Final-activity sensitivity | **22/120 (18.33%)** | Final selected activity changed after policy constraints |
| Curriculum compliance | **280/280 (100%)** | Selections obeyed encoded grade/curriculum rules |
| Tier-policy compliance | **280/280 (100%)** | Selections obeyed implemented Tier A/B rules |
| Invariant stability | **20/20 (100%)** | Equivalent histories produced identical observable decisions |
| Policy-Expectation Agreement | **398/420 (94.76%)** | Internal specification agreement, **not recommendation accuracy** |
| Synthetic session completion | **25/25 (100%)** | Sessions completed under the configured stopping rules |
| Mean skill coverage | **100%** | All applicable targeted skills assessed in the expanded simulated sessions |

**Policy-Expectation Agreement breakdown:** prediction direction **98/120**, coverage/stopping **280/280**, invariance **20/20**. Its overall value is substantially influenced by coverage checks; do **not** call it overall accuracy.

**Why only 18.33% final-activity changes?** History changed model outputs more often than final tasks because curriculum eligibility, foundational tasks, skill coverage and tier prerequisites limit choices. Changing a score does not necessarily require changing the activity. The strictly equal-eligible-candidate subset changed activities in **17/96 (17.71%)**.

## 3. Predictive model — keep two artifacts separate

| Measure | Research winner `H2_GB_RT_OFF_v1` | Deployed `TRANSFER_H2_GB_RT_OFF_v1` |
|---|---:|---:|
| ROC-AUC | **0.750158** | **0.712077** |
| Log Loss | 0.534442 | 0.563200 |
| Brier Score | 0.178955 | 0.190298 |
| Average Precision | 0.851203 | Not reported |
| Accuracy @ 0.5 / F1 | 0.734105 / 0.821907 | Not reported |
| Calibration | Platt retained for research result | Runtime calibration disabled |

Research evaluation: **937,043 held-out ASSISTments interactions from 7,001 learners**. These are **correctness-prediction metrics** on source-domain data, not recommendation effectiveness or local-child validation.

## 4. Other evidence

- **Original controlled experiment:** 5/6 model-score changes; 1/6 final-activity changes; 18/18 curriculum and tier compliance.
- **Baselines (15 synthetic sessions each):** fixed sequence completed **5/15** (23.40 mean responses); rule-only **15/15** (20.13); hybrid C2 **15/15** (19.53). All had **100% mean skill coverage**. Synthetic responses were activity-independent, so this is **not proof that hybrid recommendations improve learning**.
- **Preserved 25-fictional-learner integration run:** 489 scored interactions; 464 ML-assisted decisions; 25 cold-start fallbacks; **25/25** completions and valid C4 exports; **489/489** curriculum-valid selections.
- **Reproducibility:** expanded evaluation ran successfully locally with the same 120/20 design and matching metric values.
