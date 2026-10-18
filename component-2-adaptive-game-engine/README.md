
# Component 2 — History-Aware Adaptive Game Recommendation Engine

Part of J26-DS-330: AI-Based Robotic Interactive System for Mathematical Assessment and Early Dyscalculia Screening Support.

## Responsibility

Select mathematical assessment activities and difficulty levels using learner history and assessment coverage.

The live backend now runs one repeated workflow:

1. accept one validated answer from the UI mock or C1 contract;
2. commit it to chronological history;
3. build strictly-prior global and candidate-skill history features;
4. apply grade, prerequisite, tier and hardware eligibility;
5. score eligible candidate skills with `TRANSFER_H2_GB_RT_OFF_v1`, or explicitly fall back to Beta-Bernoulli/rules;
6. choose START, CONTINUE, ADVANCE, REINFORCE or SWITCH while enforcing minimum coverage;
7. stop when every grade-applicable game has at least three valid observations, or at the 48-interaction safeguard;
8. export a descriptive C2_TO_C4_v2.0.0 profile.

The live primary predictor is `TRANSFER_H2_GB_RT_OFF_v1`, a separately named XGBoost model trained on the original learner-disjoint ASSISTments partitions using only 25 strictly-prior numeric features reproducible from local history. The original research winner `H2_GB_RT_OFF_v1` remains frozen and is not used on cube games because its ASSISTments identifiers and platform context cannot be validly mapped. `LOCAL_BETA_BERNOULLI_v1` is the transparent baseline/fallback, not a separately trained supervised model. None of these results is validated for real local children.

Run the standalone Grade 3 demonstration from this directory:

    $env:PYTHONPATH = 'src'
    python tools\demo_trained_ml_assessment.py

The demonstration uses temporary databases and fixed seeds for a stronger learner and a repeated-error learner. It prints model provenance, probabilities, selected game/tier, decisions, completion and the descriptive C4 export boundary. See `docs/TRAINED_MODEL_DEPLOYMENT_AUDIT.md` and `docs/TRAINED_ML_GITHUB_TRANSFER.md` before transferring changes.

## Mathematical Assessment

Four assessment areas and eight provisional game types:

1. Number Sense
   - Counting & Quantity Representation
   - Magnitude Comparison
2. Number Relationships
   - Ordering & Sequencing
   - Place Value & Digit Understanding
3. Arithmetic Calculation
   - Addition
   - Subtraction
4. Multiplicative Reasoning
   - Multiplication
   - Division


