# Evidence register — dyscalculia risk screening component (ages 6–8)

Our external supervisor requires evidence for **every step and every parameter**,
because this is a clinical risk prediction. This register is the single place
where each decision is traced to its source. Nothing enters the code unless it
has a row here.

**Status codes**

| Code | Meaning |
|---|---|
| ✅ Verified | Source checked; value taken directly from it |
| 📘 NIE needed | Source will be the NIE textbook / teacher's guide; page reference pending |
| ⚠️ Assumption | Our design decision; justified in the row and must be validated with real data |
| ❌ Open | Not decided yet |

**Reporting framework.** The component is reported following **TRIPOD+AI**
(Collins et al., 2024, *BMJ* 385:e078378), the current guideline for clinical
prediction models built with regression or machine learning. It replaces TRIPOD 2015.

---

## Step A — Clinical definition (what "risk" means)

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| A1 | Condition screened | Specific learning disorder with impairment in mathematics (developmental dyscalculia) | DSM-5 (APA, 2013), Specific Learning Disorder, specifier 315.1: number sense, memorisation of arithmetic facts, accurate/fluent calculation, mathematical reasoning | ✅ |
| A1b | Core deficit targeted | Numerical magnitude processing (naming, ordering, comparing, estimating, place value) plus arithmetic fluency | [SASC-2025] UK definition of a specific learning difficulty in mathematics, published by the British Dyslexia Association | ✅ |
| A2 | Output is risk, **not** a diagnosis | Screening flag for referral to a qualified professional | DSM-5 requires individually administered standardised tests **and** comprehensive clinical assessment for diagnosis | ✅ |
| A3 | Skills must be below **age** expectations | Scores are compared with children of the same age/grade, not with the whole sample | DSM-5 Criterion B: "substantially and quantifiably below those expected for the individual's chronological age" | ✅ |
| A4 | Difficulty must **persist** | Risk is computed across sessions; a single session can only give a provisional flag | DSM-5 Criterion A: difficulties persist ≥ 6 months despite targeted intervention | ✅ |
| A5 | Use of session history | Learning slope across sessions is computed (F13) | Follows from A4 (persistence despite instruction) | ⚠️ weight 0 until validated |

## Step B — Inputs (what we receive)

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| B1 | All inputs come from other components | `student_id`, `session_id`, `question_type`, `accuracy`, `error_magnitude`, `attempts`, `response_time` | Team system design (agreed interface) | ✅ (team) |
| B2 | Child age **and** grade are required inputs | `age_months` at the session date and `grade` (1–3) on every trial and summary row. Age → DSM-5 comparison with same-age peers (A3). Grade → what has been taught (C2–C3), which differs between Grades 1, 2, 3 | DSM-5 Criterion B ("chronological age"); [NIE-G1/G2/G3]; decision of the component owner, 2026-10-20 | ✅ decided / ❌ source: no shared contract carries it yet |
| B2b | Where age comes from | Date of birth is stored **only** in the secure `student_profiles` module; datasets receive the derived `age_months`, never the date of birth or name | Leader's privacy rules (data minimisation, pseudonymous IDs: `docs/privacy-and-ethics.md`); C2 service already takes `age_group` (`adaptive_questions/service.py`) | ⚠️ to agree with team: add `age_months` and `grade` to the shared contract |
| B2c | Age–grade mismatch is flagged, not hidden | `age_grade_mismatch` = 1 when the age is outside the usual band for the grade (repeating, early or late entry) | Follows from B2; usual band needs C1 | ❌ Open until C1 is sourced |
| B3 | Full list of `question_type` values | C2's 8 activities (counting, comparison, ordering, place value, addition, subtraction, multiplication, division) in 4 areas, mapped to task types T01–T16 | [C2-msg]; [task_taxonomy.md](task_taxonomy.md) Section 4 | ✅ list / ⚠️ grade × activity table and representation still to confirm with C2 |
| B1b | `hint_count` removed from schema v1 | Hints will not be supplied to this component | Team decision reported by the component owner, 2026-10-20 (the shared contract `CubeAssessmentResult` still lists it: inform leader/C1) | ✅ |
| B1c | `robot_demo_triggered` optional, with fallback | C1 value has priority; if missing, derived as wrong answer or RT ≥ 15 s; `robot_demo_source` = component1 / derived; logged and derived rates reported separately | C1 rule [P-C1] p. 29 | ✅ rule / ❌ confirm with C1 that demo events are logged |
| B4 | Questions per session | 16–24 valid questions (≈ 2–3 per activity) → per-activity scores pooled over several sessions | [C2-msg]; DSM-5 Criterion A (A4) | ✅ source / ⚠️ number of sessions to decide |

## Step C — Content and age expectations (what a 6-, 7-, 8-year-old should do)

Sri Lankan children aged 6–8 are in Grades 1–3. Age expectations come from the
national curriculum, so the screening judges children against what they have
been taught.

Full detail: [nie_curriculum_map.md](nie_curriculum_map.md).

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| C1 | Age ↔ grade mapping | Grade is sent as an input together with age | Ministry of Education school-entry circular (to cite) | ❌ Open |
| C2 | Number range per grade | G1: count to 20, numerals 1–9. G2: count to 100 (and in 2s), numbers to 100, place value. G3: to 999–1000 | [NIE-G1] 2.1–3.5, pp. xii–xiii; [NIE-G2] 2.1–3.7, pp. x–xi; [NIE-G3] 1.1, 2.1–2.4, pp. xii–xv | ✅ (G1 guide is a DRAFT copy) |
| C3 | Operations per grade | G1: + and − within 9, with objects (− objects only). G2: + and − within 20, then ≤ 99 without carry/borrow. G3: 3-digit + (one carry), × 2, 5, 10, ÷ 2 | [NIE-G1] 4.1–5.2, p. xiv; [NIE-G2] 5.1–6.3, p. xi; [NIE-G3] 5.1–8.3, pp. xii–xiii | ✅ |
| C3b | Symbolic number comparison is Grade 2+ content | Not a Grade 1 item; Grade 1 number sense is tested non-symbolically | [NIE-G1] (no comparison of numerals; 1.5–1.6 compares sets by matching); [NIE-G2] 3.7 | ✅ |
| C3c | × and ÷ items only in Grade 3 | Component 2's multiplication and division skills disabled for Grades 1–2 | [NIE-G1], [NIE-G2] (absent); [NIE-G3] 7.1–8.3 | ✅ |
| C4 | Fact retrieval is expected by Grade 3 | Basic facts computed mentally without counting aids | [NIE-G3] p. xi | ✅ |
| C5 | Questions only inside the taught range; each question tagged with its NIE competency code | e.g. `2.4` | Follows from C2–C3 + A3 | ⚠️ agree with question component |
| C6 | Curriculum mastery vocabulary (teacher report only, not risk) | Does it successfully / Shows difficulty / Needs further help | [NIE-G3] pp. 121–122; same scale in [NIE-G1] p. 80 and [NIE-G2] p. 113 | ✅ |
| C7 | Task taxonomy (T01–T16) with domain, form, operation, representation, NIE code, C2 skill | A task type is used for risk only if taught at the grade + measured by a published dyscalculia test + supported by [M]/[S] | [task_taxonomy.md](task_taxonomy.md); [BDS], [ZAREKI-R], [TEDI-MATH] | ✅ rule / ⚠️ construct mapping cube task ↔ test task |
| M1 | Mastery-level cut-offs | ≥ 0.80 Does it successfully; ≥ 0.50 Shows difficulty; below Needs further help; at least 2 valid questions | **No source**: NIE gives the levels without percentages. Stored as configuration (`config/mastery_thresholds.json`, `validated: false`), as the leader's README requires | ⚠️ placeholder, to agree with teachers/supervisor |

## Step D — Features and weights

Full detail in [feature_evidence.md](feature_evidence.md).

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| D1 | Question types ↔ research tasks | Replaced by the task taxonomy (C7): each T-code names its [M] category | [M] Table 2 categories; [task_taxonomy.md](task_taxonomy.md) | ⚠️ mapping by construct |
| D1b | Feature availability depends on grade | Grade 1 has no F1/F2 (symbolic comparison not taught); weights renormalised over available features | C3b | ⚠️ to implement in Step 2 formula |
| D2 | Feature effect sizes | Hedges' g per feature | [M] Haberstroh & Schulte-Körne (2022), *Front. Psychol.* 13:842391; [S] Schwenk et al. (2017); [D] Decarli et al. (2020), recomputed by `src/compute_effect_sizes.py` | ✅ |
| D3 | Grading rule → weights | A: conservative significant g; B: g × 0.5; C: 0 | Our precautionary rule | ⚠️ |
| D4 | Subitizing weight 0 | — | [D]: no DD deficit in the 1–3 range | ✅ |
| D5 | Visuo-spatial memory not used | Not a component output | Constraint B1 | ✅ |

## Step E — Age-referenced scoring formula

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| E1 | Per-feature z-score within age band | z = (child − age-band mean) / age-band SD, signed so worse = positive | Standard norm-referenced scoring; required by A3 | ✅ method |
| E2 | Age-band means and SDs for 6, 7, 8 | Values per feature | No published norms for **our** robot tasks. Plan: provisional norms from literature + NIE expectations, then replaced by norms estimated from real system data | ❌ Open, the main evidence gap |
| E1b | Feature engineering implemented | Area + task features, regression-based norming on `age_months` within grade, train split only, positive = worse, capped ±3, missing kept missing | [feature_engineering.md](feature_engineering.md) | ✅ method / ⚠️ norms on synthetic data only |
| E1c | F10/F11 not used for risk | They repeat accuracy in the simulation and have no published effect size | feature_check.csv; Grade C | ✅ |
| E1d | Area number sense = F5/F6 | Use one or the other in a model, not both | feature_check.csv | ✅ |
| E3 | Composite risk index | R = Σ wᵢ zᵢ ÷ Σ wᵢ over available features (weights from D3); implemented in `src/model_formula.py` | Weighted composite; weights from D2 | ✅ implemented / ⚠️ validated on synthetic + 50 real children only |
| E3c | SVM result | Validation AUC 0.70, real Decarli 0.79: same as the formula (overlapping CIs) | [modelling.md](modelling.md) 4b | ⚠️ finding |
| E3e | Random Forest result | Validation AUC 0.66, real Decarli 0.81: no better than formula/SVM (overlapping CIs); train metrics out-of-fold | [modelling.md](modelling.md) 4c | ⚠️ finding |
| E3f | XGBoost result | Validation AUC 0.65, real Decarli 0.83: no better than the others (overlapping CIs); CV chose depth 1 (no interactions) | [modelling.md](modelling.md) 4d | ⚠️ finding |
| E3g | SHAP explanations | TreeExplainer (exact for trees); grouped by skill; ranking agrees with evidence weights (Spearman ρ 0.67); skills not assessed are left out of a child's explanation | Lundberg et al. (2020) *Nat Mach Intell* 2:56–67; [modelling.md](modelling.md) 4d | ✅ method / ⚠️ synthetic |
| E3h | Logistic regression result | Validation AUC 0.69, real Decarli 0.78: same as all other models; learned weights agree with evidence weights (Spearman ρ 0.71) | [modelling.md](modelling.md) 4e; Christodoulou et al. (2019) | ⚠️ finding |
| E3i | Model selected | Evidence-weighted formula: no model reliably better (paired bootstrap); simplest and exactly explainable wins | Rule written before running; [modelling.md](modelling.md) 4f; `config/selected_model.json` | ⚠️ synthetic, confirm in Step 6 |
| E3d | Model list | Proposal: SVM, Random Forest, XGBoost (+ SHAP); logistic regression added as reference | Proposal; TRIPOD+AI; Christodoulou et al. (2019) J Clin Epidemiol 110:12–22 | ✅ |
| E3b | Formula result | AUC 0.81 (0.69–0.93) on 50 real Decarli children without fitting; 0.69 on synthetic validation; sensitivity at moderate band 0.30–0.47 (too low for screening) | [modelling.md](modelling.md) 4a | ⚠️ finding |

## Step F — Risk thresholds (cut-offs)

DSM-5 sets no numeric cut-off. We use the cut-offs that the evidence sources
themselves used to define their groups.

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| F1 | "At risk" threshold | ≤ 16th percentile of the age band (z ≥ 1.0) | Inclusion criterion of [M] (percentile rank ≤ 16) | ✅ source / ⚠️ applied to composite |
| F2 | "High risk" threshold | ≤ 5th percentile of the age band (z ≥ 1.645) | DD criterion of [D] (≤ 5th percentile on AC-MT / BDE-2) | ✅ source / ⚠️ applied to composite |
| F3 | Bands | Low (> 16th pct), Moderate (5th–16th), High (≤ 5th) | F1 + F2 | ⚠️ |
| F4 | Screening favours sensitivity | Missing a child is worse than a false alarm, because a false alarm only leads to a professional check | Screening principle; A2 | ⚠️ state explicitly |
| F5 | Screening cut-off | Smallest flag share whose **train** sensitivity ≥ 0.70: formula worst 35% (validation sens. 0.57, spec. 0.71), SVM worst 45% (0.70 / 0.60). Reported as "needs a closer look" | Target = lower bound of the 70–80% standard for screening tests [Glascoe-2005]; [modelling.md](modelling.md) 4-S | ⚠️ synthetic, unvalidated |
| F6 | No "flagged in two sessions" rule | Requiring a flag in both short sessions drops sensitivity to 0.41; the pooled score of all sessions is used | [modelling.md](modelling.md) 4-S B | ⚠️ finding |
| F7 | Assessment length before a band is reported | About 4 sessions (≤ 96 questions): AUC 0.74 → 0.80, screening sensitivity 0.70 → 0.80; no gain beyond 4 | [modelling.md](modelling.md) 4-S C; G11 | ⚠️ finding; to agree with C2 (I6) |

## Step G — Synthetic data (for development before real data exists)

Full detail: [final_dataset.md](final_dataset.md).

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| G1 | Group differences in generator | DD mean shifted by the most precise **significant** pooled effect size. If none is significant, the [M] point estimate: F1 0.90, F2 0.75, F3 0.87, F4 0.38, F5 0.77, F6 0.69, F7 0.69, F8 0.24 | [M], [S] | ✅ |
| G2 | Correlation between features | Pooled within-group correlation of the 50 calibration children (test children excluded) | [D] | ✅ |
| G3 | Prevalence in simulated population | 5% (500 of 10,000) | Midpoint of 3–7% [HSK-2019]; equals the ≤ 5th-percentile DD criterion of [D] | ✅ |
| G4 | Correlation of arithmetic (F3, F4) with other features | 0 | Not measured in [D]; no sourced value | ⚠️ assumption |
| G5 | Synthetic data is labelled as synthetic everywhere | `SYN-` ids and `is_synthetic` column | TRIPOD+AI transparency | ✅ |
| G6 | Distribution | Multivariate normal, equal variance in both groups | Definition of a standardised mean difference | ⚠️ ignores RT skew and accuracy ceilings |
| G7 | Synthetic sample size | 10,000 children | ≥ Riley minimum of 1,425 (8 predictors, 5% prevalence, R²cs = 15% of max) [Riley-2020] | ✅ |
| G8 | Features in the dataset | F1–F8 only. F9 = data-quality check. F10–F13 need real system data | D4; no effect sizes for F10–F13 | ✅ |
| G9 | Age-band raw distributions (for per-question records) | Placeholders in `config/simulation_parameters.json` (trials-v1) | Needs E2 / pilot data | ⚠️ placeholder |
| G10 | Synthetic trials | 3,000 children × 2 sessions × 16–24 questions in the operational schema; DD effects from the children's z-scores; no effect for task types without evidence | G1–G8; [C2-msg]; [P-C3] | ✅ design / ⚠️ absolute levels are placeholders |
| G11 | Reliability with the planned assessment length | 3–5 questions per task type per child (2 sessions) shrink effects from g ≈ 0.7–0.9 to ≈ 0.21–0.47 | `results/synthetic_trials_check.csv` | ⚠️ finding: more sessions or pooling by area needed |
| G12 | Counting set size independent of difficulty tier | 30% subitizing (1–3), 70% counting (4+) at every grade | Counting to 20 is Grade 1 content ([NIE-G1] 2.1–2.2); share is a placeholder | ⚠️ (bug fix: v1 first tied set size to tier, so Grade 1 had no counting 4+) |

## Step P — Preprocessing

Full detail: [preprocessing.md](preprocessing.md).

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| P1–P4 | Drop invalid, untaught and unassigned trials | `valid_trial = 0`, `in_curriculum = 0`, no task type | B1, C3b, C7 | ✅ |
| P5 | Response time | correct answers, `rt_valid`, log(1 + RT), median | I3; [D] skew; [P-C2] p. 24 | ✅ |
| P7 | Pool all sessions per child | per task type and per C2 area | A4, G11 | ✅ |
| P8 | Accuracy smoothing | (correct + 1) / (n + 2) | Laplace correction for small counts | ✅ method |
| P10 | Data sufficiency | flag children missing a grade's risk task types or areas | Leader's `docs/screening-boundary.md` | ✅ |
| P11 | Split | 70/15/15 by child, stratified by grade and label | [P-C2] p. 23 | ✅ |

## Step R — Child report (explainability)

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| R1 | Explanation of a child's score | Exact formula contributions wᵢzᵢ ÷ Σw, grouped into 4 skills | E3; [child_report.md](child_report.md) | ✅ |
| R2 | Skill levels | ≥ 1.645 needs extra help, ≥ 1.0 needs practice, ≤ −1.0 strength; mapped to the NIE 3-level mastery scale | F1, F2; [NIE-G1/G2] | ⚠️ synthetic norms |
| R3 | Skills not assessed | Never used as a reason; not shown to parents | 4d SHAP finding; C3b | ✅ |
| R4 | Practice suggestions | Component 2 activities per skill, only those offered in the child's grade | [C2-msg]; NIE grade tables; team_interface I9 | ⚠️ C2 to confirm |
| R5 | Report wording | Language files; strengths first; no diagnostic words; teacher vs parent versions; English now, Sinhala/Tamil later | A2, F4; [child_report.md](child_report.md) | ⚠️ not yet reviewed by teachers |

## Step S — Difficulty profiles (clustering)

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| S1 | Who gets a profile | Children flagged at the screening cut-off (worst 35%) | F5 | ⚠️ synthetic |
| S2 | Area score | Mean of accuracy z and speed z per C2 area (equal weights) | No published area-level weights | ⚠️ assumption |
| S3 | Profile method | Rule-based (weak area if z ≥ 1.0); GMM rejected: BIC k = 1, silhouette 0.09, bootstrap ARI 0.19 | F1; [clustering.md](clustering.md); decision rule fixed before running | ⚠️ synthetic data have no subtypes; re-run on pilot data |
| S4 | Report level "worth monitoring" | Flagged at screening cut-off but in the low band | F5; 4-S | ⚠️ synthetic |

## Step K — Mastery classification (proposal)

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| K1 | Mastery tiers | High Performer / Progressing / Needs Urgent Assistance from true ability at ±1.0 SD | Proposal 3.5; NIE 3-level mastery scale; F1 (16th percentile) applied symmetrically | ⚠️ design decision; expert review needed |
| K2 | Label independence | Labels from hidden true ability (synthetic) / teacher rating (pilot), never from model inputs | Proposal 3.5.1 | ✅ |
| K3 | Models | Accuracy rule, RF, SVM, XGBoost, stacking RF+SVM→XGBoost (out-of-fold) | Proposal 3.5.2–3.5.4 | ✅ |
| K4 | Selection | Simplest model not reliably worse on macro-F1 and Needs-Urgent recall: SVM | Proposal 2.2, 3.6.2; [mastery_classification.md](mastery_classification.md) | ⚠️ recall criterion added after first run (documented) |
| K5 | Stacking result | Nested CV 0.48 vs single models ~0.54: no benefit | Proposal 3.5.4 | ⚠️ finding |
| K6 | Ablation | Multimodal +0.07–0.11 macro-F1 over accuracy only; history adds nothing on synthetic data | Proposal 3.4 | ⚠️ finding |
| K7 | Explanations | SHAP + LIME (agreement 0.70–0.84), fold stability 0.88 | Proposal 3.8, 3.8.1; Lundberg & Lee (2017); Ribeiro et al. (2016) | ✅ method / ⚠️ synthetic |
| K8 | Final mastery test | SVM macro-F1 0.48, Needs-Urgent recall 0.68 (0.56–0.78), baseline recall 0.41 | `results/mastery_final_evaluation.json` | ⚠️ synthetic |

## Step H — Validation

| ID | Decision | Value | Source | Status |
|---|---|---|---|---|
| H1 | Train on synthetic, test on real | 50 [D] children calibrate (G2) and tune; the 14 held-out [D] children (7 DD) are the final test and are never used before it | Prevents evaluating on data that shaped the generator | ✅ |
| H1b | Metrics on the real test set | AUC, sensitivity, specificity only. Not calibration or PPV, because the test set is 50% DD, not 5% | Prevalence dependence of calibration and PPV | ✅ |
| H2 | Metrics | Sensitivity, specificity, AUC, calibration | TRIPOD+AI | ✅ |
| H4 | Final evaluation (run once, 2026-10-21, frozen config) | Synthetic test AUC 0.66 (0.55–0.77); screening sens. 0.59 / spec. 0.69; flag rates calibrated (32% vs 35%); real test AUC 0.73 (0.45–0.98, n = 14); Grade 3 AUC 0.43 (8 DD); 900 reports, 0 rule violations | `results/final_evaluation.json`; [tripod_ai_report.md](tripod_ai_report.md) | ⚠️ synthetic + small real test |
| H3 | Clinical validity claim | **None** until tested on real Sri Lankan children with professional diagnosis and ethics approval | TRIPOD+AI; A2 | ✅ |

---

## What is sourced, what is our design, what needs validation

| Part | Status |
|---|---|
| Dyscalculia definition, persistence, below-age criterion | Sourced: DSM-5 |
| Which skills differ in dyscalculia and how strongly | Sourced: meta-analyses [M], [S]; real data [D] |
| Worst 16% / 5% cut-offs | Sourced: the criteria the published studies used to define their groups |
| Prevalence 5%, grade content, task types | Sourced: [HSK-2019], NIE guides, BDS / ZAREKI-R / TEDI-MATH |
| Methods (z-scores, smoothing, CV, SHAP, LIME, TRIPOD+AI) | Sourced: standard published methods |
| Weight grading rule (non-significant effect × 0.5) | **Own design decision** |
| Risk formula (weighted average of z-scores) | **Newly created** from published components; unvalidated |
| Mastery tiers at ±1.0 SD, skill levels, area equal weights, 35% screening cut-off | **Own design decisions** derived from sourced criteria |
| Typical accuracy / speed levels in the simulation | **Placeholders** |
| Clinical or educational validity of any output | **Not established**: needs expert review and a pilot against professional assessment (H3) |

## Sources

- **[DSM-5]** American Psychiatric Association (2013). *Diagnostic and Statistical Manual of Mental Disorders* (5th ed.), Specific Learning Disorder. Verify the criterion wording against the DSM-5 / DSM-5-TR text before final submission.
- **[TRIPOD+AI]** Collins, G. S., et al. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*, 385, e078378.
- **[M]** Haberstroh, S., & Schulte-Körne, G. (2022). The cognitive profile of math difficulties: A meta-analysis based on clinical criteria. *Frontiers in Psychology*, 13, 842391. https://doi.org/10.3389/fpsyg.2022.842391
- **[S]** Schwenk, C., Sasanguie, D., Kuhn, J.-T., Kempe, S., Doebler, P., & Holling, H. (2017). (Non-)symbolic magnitude processing in children with mathematical difficulties: A meta-analysis. *Research in Developmental Disabilities*. Verify volume and pages.
- **[D]** Decarli, G., et al. (2020). Impaired large numerosity estimation and intact subitizing in developmental dyscalculia. *PLoS ONE*, 15(12), e0244578. https://doi.org/10.1371/journal.pone.0244578
- **[HSK-2019]** Haberstroh, S., & Schulte-Körne, G. (2019). The diagnosis and treatment of dyscalculia. *Deutsches Ärzteblatt International*, 116(7), 107–114. https://doi.org/10.3238/arztebl.2019.0107 ("3–7% of all children, adolescents, and adults suffer from dyscalculia").
- **[Glascoe-2005]** Glascoe, F. P. (2005). Screening for developmental and behavioral problems. *Mental Retardation and Developmental Disabilities Research Reviews*, 11(3), 173–179. (Standard: sensitivity and specificity of 70–80% for screening tests. Verify the page before final submission.)
- **[Lundberg-Lee-2017]** Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *NeurIPS 30*, 4765–4774.
- **[Ribeiro-2016]** Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?": Explaining the predictions of any classifier. *KDD '16*, 1135–1144.
- **[Riley-2020]** Riley, R. D., Ensor, J., Snell, K. I. E., et al. (2020). Calculating the sample size required for developing a clinical prediction model. *BMJ*, 368, m441.
- **[NIE-G1]** National Institute of Education, Sri Lanka. *ගණිතය — ගුරු මාර්ගෝපදේශය — පළමුවන ශ්‍රේණිය* (Mathematics Teacher's Guide, Grade 1). First print 2015, in use from 2016. Copy received is marked DRAFT.
- **[NIE-G2]** National Institute of Education, Sri Lanka. *ගණිතය — ගුරු මාර්ගෝපදේශය — දෙවන ශ්‍රේණිය* (Mathematics Teacher's Guide, Grade 2). First print 2015, in use from 2017.
- **[BDS] [ZAREKI-R] [TEDI-MATH]** Published dyscalculia tests; full references and the sources used to verify their subtests in [task_taxonomy.md](task_taxonomy.md), Section 5.
- **[NIE-G3]** National Institute of Education, Sri Lanka (2018). *ගණිතය — ගුරු මාර්ගෝපදේශය — තුන් වන ශ්‍රේණිය* (Mathematics Teacher's Guide, Grade 3). Maharagama: NIE. Grade 1 and 2 guides still to be added.
- **[SASC-2025]** SpLD Assessment Standards Committee (2025). *SASC definition of a Specific Learning Difficulty in Mathematics.* Published by the British Dyslexia Association: https://www.bdadyslexia.org.uk/dyscalculia/sasc-definition-of-a-specific-learning-difficulty-in-mathematics-2025 (it replaces the 2019 SASC/BDA working definition).
