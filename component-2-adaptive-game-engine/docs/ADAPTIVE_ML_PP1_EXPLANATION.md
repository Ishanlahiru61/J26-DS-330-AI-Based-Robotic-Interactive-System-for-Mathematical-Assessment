# PP1 explanation and Q&A

## 60–90-second presentation script

Component 2 now runs one reusable assessment loop independent of Streamlit. After Component 1 or the UI validates an answer, it is stored chronologically. The engine rebuilds the child's skill history, applies grade, prerequisite and robot constraints, and then decides whether to continue, advance, reinforce or switch skills. For local activities, the frozen ASSISTments XGBoost model cannot be used honestly because our cube games do not have validated ASSISTments skill, template and platform-context representations. Instead, the PP1 live prototype uses a clearly labelled experimental Beta-Bernoulli learner-state model. After two answers in a skill it estimates the child's next-response success from strictly prior correct and incorrect counts. That estimate can trigger Tier B advancement or bounded Tier A reinforcement. Coverage rules still ensure every grade-applicable skill receives at least three observations, and the assessment then stops and exports a descriptive C4-compatible summary. In tests, a stronger Grade 3 pattern completed in 29 responses with Tier B advances, while a repeated-error pattern received reinforcement and still covered all eight games in 31 responses. This demonstrates functional adaptation, not real-child predictive accuracy or dyscalculia diagnosis.

## Concise Q&A

1. **Can a child complete one continuous assessment?** Yes. One service owns session, question, answer, adaptation, stopping and export.
2. **Does the frozen XGBoost choose local games?** No. Local candidate representation is scientifically incompatible.
3. **What model affects the live workflow?** LOCAL_BETA_BERNOULLI_v1, an experimental online per-child/per-skill posterior estimator.
4. **Is that the selected research model?** No. H2_GB_RT_OFF_v1 remains preserved and tested only for its ASSISTments domain.
5. **How does history affect a decision?** Only strictly prior valid answers contribute to skill attempts, correct count, posterior mean, recent outcomes and Tier A readiness.
6. **When does the engine advance?** At least three Tier A observations, existing Tier B evidence, and experimental success estimate at or above 0.75.
7. **When does it reinforce?** Estimate at or below 0.40, while respecting the four-consecutive-game bound.
8. **When does it switch?** Once minimum evidence is available and no bounded advance/reinforcement should continue, it selects an under-assessed eligible skill/area.
9. **When does it stop?** All grade-applicable games have at least three valid observations, or the 48-interaction safeguard is reached.
10. **What goes to C4?** Descriptive counts, accuracy, recent/tier evidence, coverage, missing evidence, status and recommendation history—no risk score or diagnosis.
11. **What remains?** Real-child local data, representation and calibration validation, expert threshold review and prospective recommendation-policy evaluation.
