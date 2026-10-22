# Recommendation Infrastructure — Component 2

The History-Aware Adaptive Game Recommendation Engine combines **XGBoost correctness predictions, learner history, and rule-based policies** to recommend the next mathematical activity and difficulty tier.

The system supports **Grades 1–5**, prioritizes mathematical assessment coverage, and provides safe fallback behavior. It supports early mathematical difficulty screening, **not clinical diagnosis**.

## 1. Grade and Difficulty Eligibility

**A = Tier A (basic), B = Tier B (advanced), — = unavailable.**

| Game Type | G1 | G2 | G3 | G4 | G5 |
|---|---|---|---|---|---|
| Counting | A/B | A/B | A/B | A/B | A/B |
| Comparison | A/B | A/B | A/B | A/B | A/B |
| Ordering | A | A/B | A/B | A/B | A/B |
| Place Value | — | A/B | A/B | A/B | A/B |
| Addition | A | A/B | A/B | A/B | A/B |
| Subtraction | A | A/B | A/B | A/B | A/B |
| Multiplication | — | — | A | A | A/B |
| Division | — | — | A | A/B | A/B |

**Defined in:** `../domain/curriculum.py`, `../config/activity_catalog.csv`

## 2. Prerequisite Rules

| Game Type | Required Previous Exposure |
|---|---|
| Counting, Place Value | None |
| Comparison, Ordering, Addition, Subtraction, Multiplication | 2 Counting exposures |
| Division | 2 Multiplication exposures |

An exposure is a valid previous interaction; it does not have to be correct.

**Enforced in:** `curriculum_filter.py`

## 3. Tier A → Tier B Progression

Tier B becomes eligible only when **all** conditions are satisfied for the relevant skill:

- At least **3 Tier A exposures**
- At least **75% cumulative correctness**
- At least **2 correct responses among the recent 3**
- Grade and curriculum permit Tier B

Eligibility does not guarantee immediate advancement. Assessment coverage can take priority.

**Enforced in:** `progression_rules.py`  
**Configured in:** `../config/recommendation_settings.json`

## 4. Assessment Coverage Rules

The engine prioritizes insufficiently assessed grade-applicable games and mathematical areas.

| Coverage Requirement | Minimum |
|---|---|
| Initial coverage | 1 valid response per applicable game |
| Normal game evidence | 2 valid responses per applicable game |
| Mixed correct/incorrect outcomes | 3 valid responses per applicable game |
| Mathematical area evidence | 3 valid responses per applicable area |

Coverage-first selection reduces excessive repetition of already assessed games.

**Enforced in:** `coverage.py`

## 5. XGBoost-Assisted Recommendation

The deployed **transferable XGBoost model** estimates the probability that a learner will answer a candidate question correctly using 25 transferable numerical history features.

The policy uses predictions to support decisions:

| Prediction Rule | Policy |
|---|---|
| Target correctness probability | **0.65** for ranking similarly prioritized eligible candidates |
| High predicted correctness | **≥0.75** may support advancement |
| Low predicted correctness | **≤0.40** may support reinforcement |

Predictions never override grade restrictions, prerequisites, Tier B eligibility, or coverage requirements.

**Decision logic:** `coverage.py`, `recommender.py`  
**ML inference:** `../prediction/predictor.py`  
**Feature construction:** `../feature_engineering/feature_adapter.py`  
**Trained artifact:** `../models/transferable/model.json`

## 6. Fallback and Abstention

When the trained model cannot produce a supported prediction, the system can:

- Use a **Beta–Bernoulli history-based estimate**
- Fall back to a deterministic, curriculum-safe recommendation
- Abstain from unsupported ML scoring

Cold-start learners can begin assessment without existing history.

**Enforced in:** `../prediction/predictor.py`, `fallback.py`, `abstention.py`

## 7. Assessment Completion and Safety Limits

| Rule | Threshold |
|---|---|
| Minimum valid first attempts | 16 |
| Maximum valid first attempts | 24 |
| Maximum issued tasks | 36 |
| Maximum session duration | 60 minutes |

Normal completion requires sufficient coverage evidence and the minimum valid-response count. Reaching an operational limit without sufficient evidence can terminate the session as **incomplete**.

Only validated first-attempt responses contribute to scored adaptive history. Incorrect first attempts are valid evidence; uncertain responses and retries are excluded.

**Enforced in:** `coverage.py`, `../core_service.py`, `../contracts/team_adapter.py`  
**Configured in:** `../config/adaptive_assessment_settings.json`

## 8. Policy Justification and Limitations

- **Curriculum eligibility:** Informed by Sri Lankan NIE mathematics teacher guides and the project's grade-skill mapping.
- **Prerequisites:** Represent foundational mathematical skill sequencing.
- **Coverage-first assessment:** Prioritizes evidence across different mathematical skills before repeatedly increasing task difficulty.
- **Tier progression and probability thresholds:** Configurable prototype heuristics, not officially prescribed educational or clinical cutoffs.
- **Session limits and fallback:** Bound assessment length and handle insufficient model evidence safely.
- **Future work:** Expert validation, local learner evaluation, threshold optimization, and longitudinal reassessment with cross-session history.

**Design principle:** Curriculum determines what is allowed; coverage identifies what needs assessment; XGBoost assists ranking; progression rules control difficulty; fallback handles unsupported predictions.

This is an **ML-assisted hybrid recommendation engine**, not an end-to-end learned recommendation policy or a dyscalculia diagnostic model.