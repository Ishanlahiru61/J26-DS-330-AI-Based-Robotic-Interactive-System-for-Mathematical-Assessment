# Feature evidence table — dyscalculia risk screening (ages 6–8)

Step 1 of the component. Every feature used for risk scoring is listed here with
(a) which component output it is built from, (b) the published evidence that it
separates children with dyscalculia (DD) from typical children, and (c) the
weight that evidence justifies.

Rule: **this component measures nothing itself.** All features are engineered
from the outputs of the other components.

---

## 1. Inputs received from other components

One record per answered question:

| Field | Meaning |
|---|---|
| `student_id` | Child identifier |
| `session_id` | Session identifier (ordered in time) |
| `question_type` | Skill tested: number sense, arithmetic, numerosity, … |
| `accuracy` | Answer correct (1) or not (0) |
| `error_magnitude` | \|expected answer − given answer\| (called "error rate" by the team) |
| `attempts` | Number of attempts before the final answer |
| `response_time` | Time to the first response (seconds) |

> **To confirm with the team:** the full list of `question_type` values, and
> whether `age` (or date of birth) is sent. Age is required for age-referenced
> scoring (Step 2).

## 2. Mapping question types to research constructs

The research literature does not use the words "number sense / numerosity /
arithmetic" consistently, so each question type is mapped to the task used in
the studies.

| Our `question_type` | Example question for a 6–8-year-old | Research task it corresponds to |
|---|---|---|
| Number sense | "Which is bigger, 7 or 4?" | Symbolic number comparison |
| Numerosity — how many | "How many cubes are there?" (4–7 cubes) | Exact enumeration, counting range |
| Numerosity — how many (small) | "How many cubes?" (1–3 cubes) | Subitizing |
| Numerosity — which more | "Which pile has more cubes?" | Non-symbolic comparison (ANS) |
| Arithmetic | "3 + 4 = ?" | Calculation / arithmetic fact retrieval |

If the team uses one "numerosity" type for both *how many* and *which more*,
ask them to add a sub-type field. The evidence is different for each.

## 3. Evidence sources

| Key | Source | Sample | Role |
|---|---|---|---|
| **[M]** | Haberstroh & Schulte-Körne (2022). *The cognitive profile of math difficulties: A meta-analysis based on clinical criteria.* Frontiers in Psychology, 13, 842391. https://doi.org/10.3389/fpsyg.2022.842391 | 34 studies; 680 MD, 1,565 controls; ages 8–12; DSM/ICD criteria | **Primary** weight source (pooled Hedges' g) |
| **[S]** | Schwenk, Sasanguie, Kuhn, Kempe, Doebler & Holling (2017). *(Non-)symbolic magnitude processing in children with mathematical difficulties: A meta-analysis.* Research in Developmental Disabilities. | 19 studies; N = 1,630; ages 6–14 | Response-time evidence for comparison tasks |
| **[D]** | Decarli et al. (2020). PLoS ONE 15(12): e0244578. https://doi.org/10.1371/journal.pone.0244578 — **our raw data** | 32 DD + 32 controls; ages 7.4–14 | Independent check; recomputed by `src/compute_effect_sizes.py` |
| **[P]** | Piazza et al. (2010). *Developmental trajectory of number acuity reveals a severe impairment in developmental dyscalculia.* Cognition. | — | Age argument: 10-year-old DD ≈ typical 5-year-old on number acuity |

Before the final report, check the volume and page details of [S] and [P] against the original papers.

## 4. Feature evidence table

Effect sizes: positive = the DD group performs worse. g = Hedges' g, d = Cohen's d.
Significance flags in [M] follow its Table 2 footnote (`***` p < .001, `**` p < .01, `*` p < .05).

| # | Engineered feature | Built from | [M] meta-analysis g | [S] g | [D] our data d | Grade | Evidence weight |
|---|---|---|---|---|---|---|---|
| F1 | Number sense **accuracy rate** | `accuracy`, type = number sense | 0.90 ** | — | 0.69 | A | **0.90** |
| F2 | Number sense **median RT** (correct answers) | `response_time` | 1.03 (n.s.) | 0.75 (CI 0.51–0.99) | 0.77 | A | **0.75** |
| F3 | Arithmetic **accuracy rate** | `accuracy`, type = arithmetic | 0.87 * (fact retrieval); 1.15 (calc., n.s.) | — | — | A | **0.87** |
| F4 | Arithmetic **median RT** | `response_time` | 0.38 (n.s.) | — | — | B | 0.19 |
| F5 | Enumeration 4–7 **median RT** | `response_time`, type = numerosity (how many) | 0.77 ** | — | 0.68 | A | **0.77** |
| F6 | Enumeration 4–7 **accuracy rate** | `accuracy` | 0.69 (n.s.) | — | 0.49 | B | 0.35 |
| F7 | Non-symbolic comparison **accuracy rate** | `accuracy`, type = numerosity (which more) | 0.69 ** | — | 0.54 | A | **0.69** |
| F8 | Non-symbolic comparison **median RT** | `response_time` | 0.27 (n.s.) | 0.24 (CI 0.13–0.36) | −0.06 | B | 0.12 (0.24 × 0.5, conservative [S]) |
| F9 | Subitizing (1–3) accuracy / RT | `accuracy`, `response_time` | — | — | 0.33 / −0.21 | — | **0** (intact in DD, [D]) |
| F10 | Relative **error magnitude** | `error_magnitude` ÷ expected answer | no pooled estimate | — | — | C | 0 (exploratory) |
| F11 | **First-attempt success** rate | `attempts` | no pooled estimate | — | — | C | 0 (exploratory) |
| F12 | RT **variability** (coefficient of variation) | `response_time` | no pooled estimate | — | CoV_47: 0.21 | C | 0 (exploratory) |
| F13 | **Learning slope** across sessions | `accuracy` by `session_id` | no pooled estimate | — | — | C | 0 (exploratory) |

### Grading rule (how weights are set)

- **Grade A**: a significant pooled effect in a meta-analysis, and confirmed in our data where it was measured. **Weight = the most conservative significant effect size.** For F2 we use 0.75 from [S], because 1.03 in [M] was not significant.
- **Grade B**: an effect in the expected direction, but not significant or with high heterogeneity. **Weight = effect size × 0.5.**
- **Grade C**: no pooled evidence for this exact measure. **Weight = 0** in the formula. The feature is still computed and tested in the machine learning stage (Step 4); it enters the formula only if it adds predictive value there.

This rule is our own design decision (a precautionary down-weighting), not taken
from a paper. State this explicitly in the presentation.

### Normalised weights (sum = 1)

| Feature | Raw weight | Normalised |
|---|---|---|
| F1 Number sense accuracy | 0.90 | 0.194 |
| F3 Arithmetic accuracy | 0.87 | 0.187 |
| F5 Enumeration 4–7 RT | 0.77 | 0.166 |
| F2 Number sense RT | 0.75 | 0.162 |
| F7 Non-symbolic comparison accuracy | 0.69 | 0.149 |
| F6 Enumeration 4–7 accuracy | 0.35 | 0.075 |
| F4 Arithmetic RT | 0.19 | 0.041 |
| F8 Non-symbolic comparison RT | 0.12 | 0.026 |
| **Total** | **4.64** | **1.000** |

Reading: accuracy on number-sense and arithmetic questions, and the speed of
counting and of comparing digits, carry about 70% of the risk score.

## 5. Why some things are excluded or kept at zero

| Item | Decision | Reason |
|---|---|---|
| Visuo-spatial memory (Corsi, Cowan's K) | Not used | Strongest effect in [D] (d = 1.20) and significant in [M] (g = 0.66), but **no component outputs it**. Recommend that the team add a "repeat the cube sequence" activity. |
| IQ subtests | Not used | Not a component output. In [D] they hardly differ between groups. |
| Subitizing (1–3 items) | Weight 0 | [D] found no DD deficit; the authors call subitizing "intact". Still useful as a control item: a child who fails it may have an attention or vision problem rather than dyscalculia. |
| Error magnitude, attempts, RT variability, learning slope | Weight 0 for now | Plausible, but no pooled effect size. Tested in Step 4. |
| Response time in general | Used with care | [M] warns that RT effects are more heterogeneous than accuracy effects. That is one reason RT weights are taken from the conservative estimates. |

## 6. Known limitations

1. **Age gap.** [M] covers ages 8–12, [S] 6–14 and [D] 7.4–14. No pooled evidence is specific to ages 6–8. The weights assume the *relative* importance of skills is similar at 6–8; this must be checked once real system data exists.
2. **Different tasks.** Our robot/cube questions are not identical to lab tasks (for example, response timing differs). The mapping in Section 2 is an assumption.
3. **Clinical vs screening groups.** [M] and [D] use clinically diagnosed children (≤ 16th or ≤ 5th percentile). Our component screens for *risk*; it does not diagnose.
4. **Not a diagnosis.** Output is a risk indicator to refer a child for professional assessment.

## 7. Next step (Step 2)

Turn these weights into an **age-referenced risk formula**:

- For each feature, compute an age-normed z-score: z = (child value − age-band mean) / age-band SD, signed so that worse = positive.
- Risk index R = Σ wᵢ · zᵢ, using the normalised weights above.
- Map R to Low / Moderate / High risk bands with percentile cut-offs.
