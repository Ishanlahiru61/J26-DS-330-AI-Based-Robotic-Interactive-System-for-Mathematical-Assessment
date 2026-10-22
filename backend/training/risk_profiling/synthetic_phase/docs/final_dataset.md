# Final dataset — data card (v1)

Built by `src/build_final_dataset.py` (seed 42). Parameters are in
`results/generator_parameters.json`. The fidelity check is in `results/synthetic_check.csv`.
Every value below has a row in [evidence_register.md](evidence_register.md) (Steps G and H).

## 1. What the datasets are

| File | Rows | Real or synthetic | In Git | Use |
|---|---|---|---|---|
| `data/synthetic_children_v1.csv` | 10,000 children (500 DD) | **Synthetic** | yes | Model training |
| `data/private/risk_profiling/real_decarli_calibration_z.csv` | 50 children (25 DD) | Real [D] | no | Source of the correlations; model tuning |
| `data/private/risk_profiling/real_decarli_test_z.csv` | 14 children (7 DD) | Real [D] | no | **Final test only.** Never used to build or tune anything |

One row per child. Columns:

| Column | Meaning |
|---|---|
| `child_id` | `SYN-xxxxx` (synthetic) or `DEC-<Sub>` (Decarli participant number) |
| `is_synthetic` | True / False, so synthetic rows can never be mistaken for real children (G5) |
| `z_number_sense_accuracy` … `z_nonsymbolic_comparison_rt` | Features F1–F8 from [feature_evidence.md](feature_evidence.md), as z-scores |
| `dd_label` | Target: 1 = developmental dyscalculia, 0 = typical |

**z-score scale.** 0 = average typical child, 1 = one standard deviation **worse** than
average. The sign is flipped for accuracy, so a higher value is always worse (register E1).
The real children are scored against the control children of the calibration split. The
runtime system will score children against age-band norms (E2).

## 2. How the synthetic children are made

1. **How many children: 10,000.** Riley et al. (2020) give a minimum of 1,425 children
   (71 DD) for a prediction model with 8 predictors at 5% prevalence. We use 7× that,
   because synthetic children cost nothing.
2. **Who has DD: exactly 5% (500 children).** The literature range is 3–7% [Haberstroh &
   Schulte-Körne 2019]. The midpoint, 5%, is also the ≤ 5th-percentile criterion that [D]
   used to define DD.
3. **Typical children:** their 8 feature z-scores are drawn from a multivariate normal
   distribution with mean 0, SD 1, and the correlations in Section 3.
4. **DD children:** same distribution, but each feature's mean is shifted by its published
   effect size (Section 4). Both groups have the same spread. That is what an effect size
   means (mean difference in SD units).

## 3. Correlations between features (G2, G4)

Measured on the 50 real calibration children, after removing the DD/control difference
(pooled within-group correlation), so that the group effect does not inflate them.

| | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 |
|---|---|---|---|---|---|---|---|---|
| F1 number sense acc. | 1.00 | −0.23 | 0 | 0 | −0.07 | 0.07 | 0.15 | −0.07 |
| F2 number sense RT | −0.23 | 1.00 | 0 | 0 | 0.49 | 0.16 | 0.14 | 0.39 |
| F3 arithmetic acc. | 0 | 0 | 1.00 | 0 | 0 | 0 | 0 | 0 |
| F4 arithmetic RT | 0 | 0 | 0 | 1.00 | 0 | 0 | 0 | 0 |
| F5 enumeration RT | −0.07 | 0.49 | 0 | 0 | 1.00 | −0.25 | −0.24 | 0.41 |
| F6 enumeration acc. | 0.07 | 0.16 | 0 | 0 | −0.25 | 1.00 | 0.23 | −0.05 |
| F7 non-symb. acc. | 0.15 | 0.14 | 0 | 0 | −0.24 | 0.23 | 1.00 | −0.39 |
| F8 non-symb. RT | −0.07 | 0.39 | 0 | 0 | 0.41 | −0.05 | −0.39 | 1.00 |

Reading: the response times move together (0.39–0.49, a general speed factor). Within one
task, speed and accuracy trade off (−0.23 to −0.39). Arithmetic (F3, F4) was **not measured
in [D]**, so its correlations are set to 0. This is an assumption (G4), to be replaced once
real system data exists.

## 4. Fidelity check (does the synthetic data match its sources?)

| Feature | Effect size used | Synthetic result | Real calibration children |
|---|---|---|---|
| F1 number sense accuracy | 0.90 [M] | 0.88 | 0.60 |
| F2 number sense RT | 0.75 [S] | 0.73 | 0.71 |
| F3 arithmetic accuracy | 0.87 [M] | 0.84 | not measured |
| F4 arithmetic RT | 0.38 [M] | 0.34 | not measured |
| F5 enumeration 4–7 RT | 0.77 [M] | 0.75 | 0.74 |
| F6 enumeration 4–7 accuracy | 0.69 [M] | 0.66 | 0.44 |
| F7 non-symbolic accuracy | 0.69 [M] | 0.71 | 0.65 |
| F8 non-symbolic RT | 0.24 [S] | 0.26 | 0.03 |

- Synthetic vs target: every difference is ≤ 0.04, inside the sampling error (SE ≈ 0.05 with 500 DD children).
- Synthetic vs target correlations: largest difference 0.019.
- Real vs literature: F2, F5 and F7 agree. F1, F6 and F8 are weaker in the real children. This is the
  expected gap between one 64-child study and a meta-analysis. It is also why the final test uses real children.

## 5. What was left out, and why

| Item | Reason |
|---|---|
| F9 subitizing (1–3) | No DD deficit [D] (D4). Accuracy is at ceiling (skew −4.4), so z-scores are not meaningful. Kept as a runtime data-quality check, not a model feature |
| F10–F13 (error magnitude, attempts, RT variability, learning slope) | No published effect size. Simulating them would only reproduce our own guess. They must be evaluated on real system data |
| Age | Each z-score is already relative to the child's age group (E1). The real [D] data has no age column |
| Per-question records (`CubeAssessmentResult` format) | These need raw-scale age norms (mean accuracy / RT for ages 6, 7, 8 on our tasks), which do not exist yet (E2). They follow once the pilot provides norms |

## 6. Limitations (report these)

1. The synthetic data reproduces the literature by construction. A model that does well on it has only
   shown that it can learn the literature. **Only the real test set gives evidence of performance.**
2. The real test set is small (14 children) and 50% DD, not 5%. AUC, sensitivity and specificity are
   valid there. Predicted probabilities and positive predictive value are not, at that prevalence.
3. [D] children are Italian and aged 7.4–14. Our target is Sri Lankan children aged 6–8.
4. The real test set has no arithmetic features (F3, F4 are empty). Models are tested on F1, F2, F5–F8.
5. The multivariate normal assumption ignores skew in real RTs and ceiling effects in accuracy.
6. The DD label follows research criteria. It is not a clinical diagnosis of any child.
