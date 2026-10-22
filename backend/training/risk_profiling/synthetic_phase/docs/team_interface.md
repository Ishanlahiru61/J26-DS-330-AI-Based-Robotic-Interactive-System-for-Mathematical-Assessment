# Team interface — what the other three components send to this component

Source documents: the three teammates' project proposals (J26-DS-330, September 2026).
Page numbers are the printed page numbers in each proposal.

| Key | Proposal | Owner |
|---|---|---|
| **[P-C1]** | AI-Based Cube Detection & Real-Time Answer Validation System with Robotic Arm Demonstration | Naurunna L.A.D.P. (IT23176942) |
| **[P-C2]** | History-Aware Adaptive Game Recommendation Engine | Ariyarathna M.L. (IT23178304) |
| **[P-C3]** | Vision-Based Detection of Finger-Counting Behaviour | Dissanayake D.M.I.L. (IT23208926) |

The proposals number the components differently: [P-C1] calls this component "Component 3",
while [P-C2] and [P-C3] call it C4. All three describe it as performance analysis / mastery
classification / profiling (P-C1 p. 11; P-C2 pp. 11, 35; P-C3 pp. 12, 26).

---

## 1. Fields this component can receive

### From C1: answer validation ([P-C1] pp. 11, 29, 37)

| Field | Proposal wording | Page |
|---|---|---|
| `student_id` | "learner ID", randomised subject ID e.g. `STU_2026_XXXX` | 31, 37 |
| `task_id` | "task ID" | 37 |
| `operation` | one of six: counting, addition, subtraction, grouping, division, place value | 29, 36 |
| `accuracy` | "validation outcome" | 37 |
| `response_time_ms` | "procedural reaction latency in milliseconds" | 37 |
| `error_type` | "off-by-one undercounting, inverted operand placement, base-10 transcoding errors" | 32, 36–37 |
| `error_magnitude` | "calculate error magnitude" | 37 |
| `attempts` | "attempt counts" | 31 |
| `robot_demo_triggered` | robot demonstrates after an incorrect answer **or hesitation > 15 s** | 29, 37 |

### From C2: adaptive task selection ([P-C2] pp. 23–24, 38)

| Field | Proposal wording | Page |
|---|---|---|
| `skill` | seven skills: counting/subitizing, quantity comparison, addition, subtraction, multiplication/grouping, division/sharing, place value/building | 23–24 |
| `difficulty_tier` | Tier A foundational / Tier B extended (project operationalisation, not MoE levels) | 24 |
| `selection_status` | model recommendation vs rule-based fallback; eligible and selected activities logged | 23, 38 |
| `interaction_position` | chronological order within the learner | 23 |

### From C2: update from the C2 owner, 2026-10-20 ([C2-msg])

Message sent by the C2 owner to this component's owner (supersedes the proposal where they differ):

| Item | C2 statement | Change from [P-C2] |
|---|---|---|
| Activity types | **8**: counting, comparison ("comparing numbers: greater than, less than, equal"), ordering, place value, addition, subtraction, multiplication, division | Was 7 skills. Subitizing no longer separate; "quantity comparison" now described as comparing **numbers**; **ordering** added |
| Assessment areas | 4: Number Sense, Number Relationships, Arithmetic Calculation, Multiplicative Reasoning | New |
| Grades | 1–5; "not all eight activities are assigned to every grade" (grade-appropriate curriculum) | Grade range stated |
| Selection | Next activity and difficulty tier from learner history, predicted performance (**XGBoost**), curriculum constraints; coverage of required areas; order differs between students | Model named |
| Length | **16–24 valid questions** per assessment, maximum 24 | New |
| Output to C4 | correctness, activities attempted, difficulty levels, area coverage | Response time, attempts, error value are **not** in C2's output → must come from C1 (`CubeAssessmentResult`) |

Consequences for this component (see [task_taxonomy.md](task_taxonomy.md), Section 4):
- 16–24 questions over up to 8 activities ≈ **2–3 questions per activity per session**. One session gives a noisy
  per-activity score → risk must pool **several sessions** (also required by DSM-5 persistence, register A4).
- "Comparison" of **numbers** is not Grade 1 content ([NIE-G1]); non-symbolic comparison (cube piles), which
  has the strongest Grade 1 evidence, is not named.
- Subitizing (1–3) and counting (4+) have different evidence ([D]); they can only be separated if the set size
  (`expected_answer`) is sent.

### From C3: finger-counting behaviour ([P-C3] pp. 24–25, 28)

| Field | Proposal wording | Page |
|---|---|---|
| `behaviour_class` | finger counting / other hand activity / unknown | 19, 28 |
| `behaviour_confidence` | confidence score | 28 |
| `behaviour_duration` | duration of the finger-counting episode | 28 |
| `tracking_quality` | tracking coverage; "unknown" when evidence is insufficient | 25, 28 |

C3 results are linked to C1 results by the same `task_id` and timestamps ([P-C3] p. 25).

## 2. Constraints the proposals impose on this component

| # | Constraint | Source | Consequence for this component |
|---|---|---|---|
| I1 | Finger counting "is not independently sufficient to indicate dyscalculia" and must be combined with age, accuracy and RT | [P-C3] pp. 3, 14 | Finger counting may only enter as an age-dependent, accuracy-conditioned feature |
| I2 | "unknown" ≠ "no finger counting" | [P-C3] pp. 21, 25 | Treat `unknown` as missing, never as 0 |
| I3 | Robot demo fires after hesitation > 15 s | [P-C1] p. 29 | RT is effectively censored near 15 s; answers after a demo are not independent → flag and exclude from RT features |
| I4 | Tasks are chosen adaptively by C2 | [P-C2] pp. 21–22 | Raw % correct is not comparable between children who got different tiers → features per skill × tier |
| I5 | C2 does not do "final dyscalculia-risk classification" | [P-C2] pp. 23, 39 | Risk output is this component's responsibility alone |
| I6 | Local pilot: ≈ 30–40 children aged 6–8, ≈ 2 sessions of 10–15 min; C2 targets ≥ 20 interactions per learner | [P-C3] pp. 18–19; [P-C2] p. 23 | Size of the real local dataset; no diagnosis labels → norms only, not risk validation |
| I8 | Robot demo events may not be logged | C1 to confirm | `robot_demo_triggered` is optional; missing values are derived by C1's rule and marked `robot_demo_source = derived` (register B1c). **Request to C1:** confirm whether each demo event is logged per question |
| I9 | Practice suggestions in the child report come from C2's activities | C2 to confirm | Report suggests C2 activities per weak skill and grade (child_report.md). **Request to C2:** can the engine receive a per-child priority list (e.g. 'prioritise comparison'), or is it shown to the teacher only? Until confirmed: teacher only |
| I7 | Age or date of birth is **not** listed as a field in any proposal | all three | **Decided (2026-10-20):** this component needs `age_months` and `grade`. Request to the team: add both to the shared contract, sourced from `student_profiles` (C2's service already uses `age_group`). Evidence register B2–B2c |

## 3. Evidence cited by teammates that this component can reuse, after checking the original

| Ref (as cited) | Use here | Status |
|---|---|---|
| Haberstroh & Schulte-Körne (2019), *Dtsch Ärztebl Int* 116:107–114, doi 10.3238/arztebl.2019.0107 ([P-C2] [1], [P-C3] [1]) | Prevalence for the synthetic population (register G3) | ⚠️ read the original for the exact range |
| Szardenings et al. (2018), *Front. Psychol.* 8:1615 ([P-C3] [4]) | RT must be read together with accuracy; 81 DD vs 272 controls, Grades 2–4 | ⚠️ check effect sizes |
| Hewapathirana et al. (2021), ICIAfS, doi 10.1109/ICIAfS52090.2021.9605998 ([P-C2] [5]) | Local Sri Lankan baseline: SVM on score + mean time per quiz type, ages 6–8 | ⚠️ baseline to compare against |
| Wijesekara et al. (2026), *J. Multidiscip. Transl. Res.* 11(1):59–73 ([P-C2] [4]) | Sinhala-medium SLD screening items, ages 5–8 | ⚠️ content source |
| Kißler & Kuhn (2025), *Front. Psychol.* 16:1590581 ([P-C2] [2]) | Heterogeneous DD profiles → report per-skill profile, not one score only | ⚠️ |
| Lafay et al. (2013); Poletti et al. (2025) ([P-C3] [2], [3]) | Finger use is normal and helpful at ages 4–7 | ⚠️ |
| Pelánek (2024), *UMUAI* 34:729–752 ([P-C2] [10]) | RT depends on task and interface | ⚠️ |

Citations in [P-C1] are **not** reused without checking first: several of them could not be matched to a known
publication (for example a DOI ending in `eabk1234`).
