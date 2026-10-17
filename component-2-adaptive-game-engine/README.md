
# Component 2 — History-Aware Adaptive Game Recommendation Engine

Part of J26-DS-330: AI-Based Robotic Interactive System for Mathematical Assessment and Early Dyscalculia Screening Support.

## Responsibility

Select mathematical assessment activities and difficulty levels using learner history and assessment coverage.

The live backend now runs one repeated workflow:

1. accept one validated answer from the UI mock or C1 contract;
2. commit it to chronological history;
3. update the learner's per-skill experimental success estimate when at least two observations exist;
4. apply grade, prerequisite, tier and hardware eligibility;
5. choose START, CONTINUE, ADVANCE, REINFORCE or SWITCH while enforcing minimum coverage;
6. stop when every grade-applicable game has at least three valid observations, or at the 48-interaction safeguard;
7. export a descriptive C2_TO_C4_v2.0.0 profile.

The operational local estimator is LOCAL_BETA_BERNOULLI_v1. It is an online probabilistic learner-state model, not the frozen H2_GB_RT_OFF_v1 XGBoost model. The frozen model is preserved for its ASSISTments research domain because local cube activities lack validated ASSISTments task identities and context. Local model output is explicitly labelled EXPERIMENTAL_MODEL_ASSISTED and is not validated for real children.

Run the standalone Grade 3 demonstration from this directory:

    $env:PYTHONPATH = 'src'
    python tools\demo_adaptive_assessment.py

The demonstration uses temporary databases and fixed seeds for a stronger learner and a repeated-error learner. See docs/ADAPTIVE_ML_FEASIBILITY_AUDIT.md and docs/ADAPTIVE_ML_GITHUB_TRANSFER.md before transferring the changes.

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


