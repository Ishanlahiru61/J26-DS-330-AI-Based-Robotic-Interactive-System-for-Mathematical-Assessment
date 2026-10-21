# Step 4g — Explainable child report

The selected model (evidence formula, 4f) gives each child a risk band. This step explains the band to a
teacher and a parent in plain language: which skills are fine, which need support, and which Component 2
activities to practise next. Every rule below names its evidence-register row.

## 1. From score to skills (`src/explain_formula.py`)

The formula is R = Σ wᵢzᵢ ÷ Σ wᵢ over the features the child has (E3), so each feature's contribution
cᵢ = wᵢzᵢ ÷ Σ w is **exact** and the contributions add up to R. Features are grouped into four skills
(accuracy and speed of one task share credit, as with SHAP in 4d):

| Skill | Features | Task types |
|---|---|---|
| Comparing written numbers | F1, F2 | T05 |
| Comparing amounts of objects | F7, F8 | T04 |
| Counting objects | F5, F6 | T02 |
| Arithmetic (+, −; ×, ÷ in Grade 3) | F3, F4 | T10–T14 |

For each skill: z = evidence-weighted mean of its available feature z-scores (positive = worse), and

| Level | Rule | Source | NIE mastery scale |
|---|---|---|---|
| Needs extra help | z ≥ 1.645 (≤ 5th percentile) | F2 | Needs further help |
| Needs more practice | z ≥ 1.0 (≤ 16th percentile) | F1 | Shows difficulty |
| As expected | −1.0 < z < 1.0 | F1 | Does it successfully |
| Strength | z ≤ −1.0 (≥ 84th percentile) | F1, same distance on the good side | Does it successfully |

- **Not assessed:** a skill with no data (e.g. symbolic comparison in Grade 1, not taught, C3b) is labelled
  "not assessed". It is never a reason and never appears in the parent text (finding from 4d SHAP).
- **Main reasons:** at most two skills that need support, ordered by contribution.
- **Practice suggestions:** Component 2 activities for the skill, only if C2 offers them in the child's grade:

| Skill | C2 activities | Grades |
|---|---|---|
| Comparing written numbers | comparison, ordering | 2–3 |
| Comparing amounts of objects | comparison | 1–3 |
| Counting objects | counting, place_value | 1–3 |
| Arithmetic | addition, subtraction (+ multiplication, division in Grade 3) | 1–3 |

The output contains **keys only** (skill, level, activity), no sentences.

## 2. From keys to sentences (`src/child_report.py`)

Wording lives in `config/report_templates/<language>.json`; the code never contains report sentences.
English (`en.json`) is built; Sinhala (`si.json`) and Tamil (`ta.json`) can be added later by translating a
copy. A test fails if a language file misses any English key. Skill names in Sinhala should follow the NIE
Sinhala teacher's guides, and translations should be checked by a teacher or native speaker.

| Rule | Why |
|---|---|
| Strengths and on-track skills first | Report starts from what the child can do |
| Never "dyscalculia", "disorder", "disability" (tested) | Screening, not diagnosis (A2, F4) |
| Teacher sees band, reasons, activities, not-assessed skills, data note | Teacher acts on the result |
| Parent sees no band word, no not-assessed skills, a home activity and a reassurance note | Avoid alarm; parent can help |
| High band: "re-check; if difficulties continue, consider referral" | Persistence before referral (DSM-5 A, 4-S) |
| Flagged at the screening cut-off but low band: "worth monitoring" | Sensitivity-oriented cut-off (4-S, F5) |
| Flagged children get a difficulty profile (teacher only) | Step 5, [clustering.md](clustering.md) |
| Insufficient data: no result, "complete another session" | Leader's screening boundary |
| Comparison only "with children of the same age and grade" | No comparison with named classmates |
| Child names come from `student_profiles` at run time, never stored here | Privacy rule |

## 3. Examples (synthetic children, English)

**SYN-00034, Grade 2, high band**

> *Teacher:* SYN-00034, Grade 2: needs a closer look (priority)
> As expected for age and grade: counting objects and addition/subtraction.
> Needs support: comparing written numbers (needs extra help) and comparing amounts of objects (needs extra help).
> Main reasons for this result: comparing written numbers and comparing amounts of objects.
> Suggested robot activities: Comparison and Ordering.
> Give more practice in the suggested activities and re-check after further sessions. If the difficulties continue, consider referral for a specialist assessment.
> Based on 2 sessions (35 questions), compared with children of the same age and grade. This is a screening result, not a diagnosis.

> *Parent:* Your child is doing as expected with counting and sums (adding, taking away).
> Your child finds comparing numbers and comparing groups of objects harder than most children of the same age at the moment.
> You can help at home with short, playful games about comparing numbers and comparing groups of objects, for example using small objects such as beads or stones.
> The teacher will practise this in class and check progress again soon.
> This is a short check of number skills, not a diagnosis. Children develop at different speeds.

**SYN-00006, Grade 1, low band** (one weaker skill; symbolic comparison not assessed)

> *Teacher:* SYN-00006, Grade 1: no concern found
> As expected for age and grade: counting objects and addition/subtraction.
> Needs support: comparing amounts of objects (needs more practice).
> Suggested robot activities: Comparison.
> No overall concern. Continue normal classroom activities; extra practice in any suggested activity may help.
> Not assessed at this grade: comparing written numbers.
> Based on 2 sessions (37 questions), compared with children of the same age and grade. This is a screening result, not a diagnosis.

> *Parent:* Your child is doing as expected with counting and sums (adding, taking away).
> This is a short check of number skills, not a diagnosis. Children develop at different speeds.

**SYN-00081, Grade 3, insufficient data**

> *Teacher:* SYN-00081, Grade 3: not enough data yet. There is not enough data to give a result yet. Please complete another session with the robot.
> *Parent:* Your child needs to play a few more number games before we can share results.

Run `python src/child_report.py` to print these examples (moderate band included).

## 4. Limitations and open items

- Levels use the same synthetic, unvalidated norms as the bands; re-estimate with pilot data (H1).
- Wording has not yet been reviewed by teachers or parents; a short review with 2–3 teachers is recommended
  before the pilot.
- Arithmetic pools +, −, ×, ÷ (F3/F4), so the report cannot yet say *which* operation is weak.
- Whether Component 2 can **receive** these suggestions (to prioritise activities) is open (team_interface I9);
  until confirmed, suggestions are shown to the teacher only.
