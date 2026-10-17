# Known limitations

- Game taxonomy, Tier boundaries and NIE mappings are provisional engineering definitions pending curriculum/domain-expert approval.
- Robot presentation and C1 answer detection are mocks; text is the working speech-adapter boundary. No physical actuation or voice playback is claimed.
- Local recommendations now use an experimental per-child Beta-Bernoulli skill-state estimate after two valid observations, combined with deterministic curriculum, coverage and stopping rules. This estimator is not validated on real children and is not the frozen XGBoost model. ASSISTments XGBoost probabilities still cannot be transferred to the eight local games without a validated representation and local outcome study.
- Skill indices are transparent descriptive first-attempt percentages. Evidence labels are engineering states, not clinical adequacy or mastery thresholds.
- C2 does not diagnose dyscalculia or produce risk probabilities. C4 owns any combined screening-risk model.
- Single-process SQLite is suitable for PP1, not concurrent multi-site deployment.
- The public-data history is start-time ordered. About 21.0318% of events started before an earlier event finished, so claims of real-time feature availability require that caveat.
- The frozen joblib preprocessor needs the documented `__main__.RTStats` compatibility shim.
