# Task taxonomy — what kinds of maths questions the system asks, and why

Every question the robot/cube system asks gets a **task type** (T01–T16). Each task type is described by
several columns, so the data says *what* was tested, not only a broad label:

| Column | Values |
|---|---|
| `math_domain` | `number_sense`, `counting`, `arithmetic`, `place_value`, `pre_number` |
| `task_form` | `enumeration`, `comparison`, `ordering`, `sequence`, `transcoding`, `calculation`, `word_problem` |
| `operation` | `none`, `addition`, `subtraction`, `multiplication`, `division` |
| `representation` | `non_symbolic` (cubes, dots), `symbolic` (digits), `verbal` (spoken), `mixed` (digit → build with cubes) |
| `nie_competency` | grade + NIE code, e.g. `G2-3.7` |
| `c2_skill` + `difficulty_tier` | link to what Component 2 sends ([P-C2] pp. 23–24) |

## 1. Rule for including a task type

A task type is **used for risk scoring** only if it passes all three tests:

1. **Curriculum:** it is taught at the child's grade ([NIE-G1], [NIE-G2], [NIE-G3]). Testing what was never
   taught is not evidence of difficulty (DSM-5 Criterion B compares with what is *expected*).
2. **Clinical practice:** at least one published dyscalculia test measures it ([BDS], [ZAREKI-R], [TEDI-MATH]).
3. **Research evidence:** a meta-analysis shows a DD-vs-typical difference for it ([M], [S]).

A task type that passes 1 and 2 but not 3 is still **recorded** (for the teacher report and mastery), but its
risk weight stays 0 until real data supports it (same rule as Grade C in [feature_evidence.md](feature_evidence.md)).

## 2. The taxonomy

✅ = passes, ❌ = fails, — = not applicable. Grade columns: is it taught in that grade?

| ID | Task (example with cubes) | domain · form · representation | NIE G1 | NIE G2 | NIE G3 | Dyscalculia tests that measure it | [M] evidence (g) | C2 skill · tier | Risk use |
|---|---|---|---|---|---|---|---|---|---|
| T01 | Subitizing: "How many?" (1–3 cubes, shown briefly) | counting · enumeration · non_symbolic | ✅ 3.1 | ✅ (dice patterns, L6) | ✅ | BDS dot enumeration; ZAREKI-R counting dots | part of quantity–number linking: acc 0.69 n.s., RT 0.77 ** | counting_subitizing · A | **Control item, weight 0** ([D]: subitizing intact in DD) |
| T02 | Enumeration: "How many cubes?" (4–20) | counting · enumeration · non_symbolic | ✅ 2.1–2.3 (≤ 20) | ✅ 2.1–2.2 (≤ 100) | ✅ 1.1 | BDS dot enumeration; ZAREKI-R counting dots; TEDI-MATH enumeration | quantity–number linking RT 0.77 ** | counting_subitizing · B | ✅ F5, F6 |
| T03 | Counting sequence: count on, count back, count in 2s | counting · sequence · verbal | ✅ (to 20) | ✅ 2.3 (2s) | ✅ 1.1 (2s, 5s, 10s) | ZAREKI-R counting backwards; TEDI-MATH counting | number word knowledge/counting 0.84 (n.s.) | counting_subitizing · B | record; weight 0 |
| T04 | Non-symbolic comparison: "Which pile has more?" | number_sense · comparison · non_symbolic | ✅ 1.5–1.6 | (assumed) | (assumed) | ZAREKI-R perceptual estimation (related); TEDI-MATH magnitude estimation (related) | quantity processing acc 0.69 **; RT 0.24 [S] | quantity_comparison · A | ✅ F7, F8 |
| T05 | Symbolic comparison: "Which is bigger, 7 or 4?" | number_sense · comparison · symbolic | ❌ **not taught** | ✅ 3.7 (< 100) | ✅ 2.4 (≤ 999) | BDS number comparison; ZAREKI-R written and spoken number comparison | numerical relations acc 0.90 **; RT 0.75 [S] | quantity_comparison · B | ✅ F1, F2 (**Grade 2+ only**) |
| T06 | Ordering three numbers | number_sense · ordering · symbolic | ❌ | ✅ 3.7 | ✅ 2.4 | — | same family as T05 (assumption) | quantity_comparison · B | record; weight 0 |
| T07 | Number before / after | number_sense · sequence · symbolic | ✅ 3.5 (2–8) | ✅ | ✅ | — | not reported | counting_subitizing · A | record; weight 0 |
| T08 | Build a number: "Show 7 with cubes" | number_sense · transcoding · mixed | ✅ 3.1–3.4 (1–9) | ✅ 3.1–3.4 (≤ 100) | ✅ | ZAREKI-R number reading/dictation; TEDI-MATH transcoding | not reported in [M] | place_value_building · A | record; weight 0 |
| T09 | Place value: "Build 34 with tens and ones" | place_value · transcoding · mixed | ❌ | ✅ 3.6 | ✅ 2.1–2.3 | TEDI-MATH base-10 system | not reported in [M] | place_value_building · B | record; weight 0 |
| T10 | Addition facts (sum ≤ 9, then ≤ 20) | arithmetic · calculation · symbolic or non_symbolic | ✅ 4.1–4.2 (≤ 9, objects) | ✅ 5.1–5.2 (≤ 20) | ✅ | BDS addition; ZAREKI-R mental calculation; TEDI-MATH operations | fact retrieval acc 0.87 *; computation RT 0.38 n.s. | addition · A | ✅ F3, F4 |
| T11 | Two-digit addition (≤ 99, no carry; Grade 3 one carry) | arithmetic · calculation · symbolic | ❌ | ✅ 5.3 | ✅ 5.1–5.3 | ZAREKI-R mental calculation | calculation 1.15 (n.s.) | addition · B | ✅ F3, F4 |
| T12 | Subtraction facts (from ≤ 9, then ≤ 20) | arithmetic · calculation · non_symbolic (G1) or symbolic | ✅ 5.1–5.2 (**objects only**) | ✅ 6.1–6.2 | ✅ 6.1 | ZAREKI-R mental calculation; TEDI-MATH operations | fact retrieval 0.87 * | subtraction · A | ✅ F3, F4 |
| T13 | Two-digit subtraction (≤ 99, no borrow) | arithmetic · calculation · symbolic | ❌ | ✅ 6.3 | ✅ 6.2–6.3 | ZAREKI-R mental calculation | calculation 1.15 (n.s.) | subtraction · B | ✅ F3, F4 |
| T14 | Multiplication × 2, 5, 10 (equal groups of cubes) | arithmetic · calculation · mixed | ❌ | ❌ | ✅ 7.1–7.3 | ZAREKI-R (3 × 2 type items); BDS multiplication only age ≥ 10 | fact retrieval 0.87 * (extension) | multiplication_grouping · A/B | ✅ F3, F4 (**Grade 3 only**) |
| T15 | Division ÷ 2 (sharing cubes equally) | arithmetic · calculation · mixed | ❌ | ❌ | ✅ 8.1–8.3 | none of the three | not reported | division_sharing · A/B | record; weight 0 (**Grade 3 only**) |
| T16 | Word problem ("Peter has 12 marbles, gives 5 …") | arithmetic · word_problem · verbal | ❌ | ✅ 5.3, 6.3 | ✅ | ZAREKI-R problem solving; TEDI-MATH verbal-statement operations | math reasoning 1.29 (n.s.) | addition / subtraction · B | record; weight 0 |

**Not in the taxonomy:** measurement, time, money, shapes (taught, but not part of the dyscalculia evidence
base, SASC-2025 / DSM-5); conservation, seriation and one-to-one matching as stand-alone tasks
(G1 pre-number content `G1-1.x`; the TEDI-MATH "logical operations" domain covers related content, but
no meta-analytic effect size → can be added later as `pre_number`); number-line positioning (ZAREKI-R has it,
but it is not a NIE competency in Grades 1–3, where the number line is only a teaching aid).

## 3. Which task types each grade gets

| Grade | Task types used for risk | Task types recorded only |
|---|---|---|
| 1 | T02, T04, T10, T12 (T01 as control) | T03, T07, T08 |
| 2 | T02, T04, **T05**, T10, T11, T12, T13 (T01 as control) | T03, T06, T07, T08, T09, T16 |
| 3 | as Grade 2 + **T14** | as Grade 2 + T15 |

Consequence for the feature table: in Grade 1, **F1/F2 (symbolic number sense) are not available**, so
number sense is carried by F7/F8 (non-symbolic comparison). The risk formula must renormalise the weights
over the features that exist for the child's grade.

## 4. How this maps to Component 2

Updated to C2's current design: **8 activity types in 4 assessment areas** ([C2-msg], see
[team_interface.md](team_interface.md)). The proposal's 7 skills ([P-C2] pp. 23–24) are superseded.

| C2 activity | C2 area (assumed, to confirm) | Taxonomy task types | NIE grades | Note |
|---|---|---|---|---|
| counting | Number Sense | T01 (1–3), T02 (4+), T03 | 1, 2, 3 | T01 vs T02 separable only via set size (`expected_answer`) |
| comparison | Number Relationships | **T05** (symbolic); T04 only if piles of cubes are also used | T05: 2, 3 | Comparing **numbers** is **not Grade 1 content**; T04 (non-symbolic) is the Grade 1 number-sense measure |
| ordering | Number Relationships | T06 | 2, 3 | weight 0 (no [M] estimate) |
| place value | Number Sense or Number Relationships | T08, T09 | T09: 2, 3 | weight 0 |
| addition | Arithmetic Calculation | T10, T11 (T16 if word problems) | 1, 2, 3 | Grade 1 with objects |
| subtraction | Arithmetic Calculation | T12, T13 | 1, 2, 3 | Grade 1 with objects only |
| multiplication | Multiplicative Reasoning | T14 | **3 only** | disable Grades 1–2 |
| division | Multiplicative Reasoning | T15 | **3 only** | disable Grades 1–2; weight 0 |

C2 covers Grades 1–5; this component covers ages 6–8 (Grades 1–3), so Grade 4–5 activities are out of scope.

**Requests to the C2 owner** (needed to assign each trial its task type):
1. Send `representation` (cubes vs digits) with each comparison item, or add a non-symbolic comparison
   activity. Without it, Grade 1 has no valid number-sense comparison.
2. Send the grade × activity table ("not all eight activities are assigned to every grade"), so it can be
   checked against [nie_curriculum_map.md](nie_curriculum_map.md).
3. Confirm which activity belongs to which of the 4 areas (place value is ambiguous).
4. Send `expected_answer` (set size) for counting, so subitizing (1–3) can be kept apart from counting.
5. Confirm C1 supplies response time, attempts and error value per question (not in C2's output).

## 5. Sources

- **[NIE-G1], [NIE-G2], [NIE-G3]** National Institute of Education, Sri Lanka. Mathematics Teacher's Guides,
  Grades 1, 2, 3. Details and page numbers in [nie_curriculum_map.md](nie_curriculum_map.md).
- **[BDS]** Butterworth, B. (2003). *Dyscalculia Screener.* London: nferNelson (now GL Assessment). Ages 6–14.
  Subtests: simple reaction time; dot enumeration; number comparison (numerical Stroop); arithmetic
  achievement (addition; multiplication only from age 10). Sources: GL Assessment support page
  (https://support.gl-assessment.co.uk/knowledge-base/assessments/dyscalculia-screener-support/administration)
  and the screener manual (https://mathematicalbrain.com/pdf/Dyscalculia_Screener_Manual.PDF).
- **[ZAREKI-R]** von Aster, M., & Dellatolas, G. (2006). *ZAREKI-R: Batterie pour l'évaluation du traitement
  des nombres et du calcul chez l'enfant.* Paris: ECPA (German version: von Aster, Weinhold Zulauf & Horn, 2006).
  11 scored subtests: counting dots, counting backwards, number dictation, number reading, positioning
  numbers on a scale, spoken number comparison, perceptual estimation, contextual estimation, written number
  comparison, mental calculation (+, −, ×), problem solving (plus digit span, not scored). Subtest list
  verified from Dos Santos, F. H., et al. (2012). Number processing and calculation in Brazilian children
  aged 7–12 years. *The Spanish Journal of Psychology*, 15(2), p. 517. Verify volume/pages before submission.
- **[TEDI-MATH]** Van Nieuwenhoven, C., Grégoire, J., & Noël, M.-P. (2001). *TEDI-MATH: Test diagnostique des
  compétences de base en mathématiques.* Paris: ECPA. From end of 2nd kindergarten year to start of Grade 3
  (≈ ages 5–8). Subtests: counting, enumeration, Arabic-numeral system, oral number system, base-10 system,
  transcoding, logical operations, operations with pictures, arithmetic statements, verbal statements,
  conceptual knowledge, magnitude estimation. Source: INAMI/RIZIV list of approved dyscalculia tests
  (https://www.inami.fgov.be/SiteCollectionDocuments/liste-logopede-tests-dyscalculie.pdf); German
  adaptation Kaufmann et al. (2009), Hogrefe. INAMI requires ≤ 16th percentile on subtests, the same
  threshold as register F1.
- **[M]**, **[S]**, **[D]**: see [feature_evidence.md](feature_evidence.md). [M] category names follow its Table 2
  (Haberstroh & Schulte-Körne, 2022, Front. Psychol. 13:842391; ** p < .01, * p < .05).
- **[P-C2]** Component 2 proposal (Ariyarathna, IT23178304), pp. 23–24.
- **[C2-msg]** Message from the C2 owner describing the current C2 design (8 activities, 4 areas, 16–24
  questions), received 2026-10-20. Text recorded in [team_interface.md](team_interface.md).

## 6. Limitations

1. [M] covers ages 8–12; Grade 1 children are younger. Effect sizes for T02/T04/T10/T12 at Grade 1 are an
   extrapolation.
2. The cube tasks are not identical to the test tasks (e.g. BDS uses a computer screen and key presses).
   The mapping in Section 2 is by **construct**, not by identical procedure.
3. The Grade 1 guide received is marked DRAFT.
4. Test manuals (BDS, ZAREKI-R, TEDI-MATH) were not read directly; subtest lists come from the sources cited.
