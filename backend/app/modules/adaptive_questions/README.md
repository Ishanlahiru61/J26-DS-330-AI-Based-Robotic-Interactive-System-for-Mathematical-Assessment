# C2 — History-Aware Adaptive Game Recommendation Engine

This package is the authoritative Component 2 (C2) backend runtime. It registers
pseudonymous learners, maintains durable assessment sessions, generates
grade-appropriate mathematical tasks, recommends the next game from strictly
prior history, records C1-validated responses, and exports descriptive evidence
for downstream C4 use.

C2 implements an educational assessment workflow. It does **not** diagnose
dyscalculia, claim clinical validation with children, interpret C3 behaviour,
perform C4 screening inference, read cameras, or control robot hardware.

## Architecture and responsibilities

| Path | Responsibility |
| --- | --- |
| `api.py` | FastAPI routes under `/adaptive-questions` |
| `schemas.py` | HTTP request and response models |
| `core_service.py` | Framework-independent `Component2Service` orchestration |
| `service.py` | Async application boundary and retained legacy protocol |
| `contracts/team_adapter.py` | Strict mapping and validation of C1 team events |
| `domain/` | Grade curriculum, prerequisites, task eligibility, and question generation |
| `feature_engineering/` | Strictly prior learner-history features |
| `prediction/` | Manifest verification, XGBoost inference, and fallback estimates |
| `recommendation/` | Coverage, ranking, difficulty, progression, and stopping policy |
| `infrastructure/` | SQLite repositories, migrations, errors, events, and resource paths |
| `config/`, `migrations/`, `models/` | Versioned runtime configuration, schema, and model artifacts |

`Component2Service` is the portable core. FastAPI and shared team contracts are
confined to the boundary layers. The supported imports are:

```python
from backend.app.modules.adaptive_questions import Component2Service, IntegrationError
```

If this package directory is copied into another application, the core remains
importable as:

```python
from adaptive_questions import Component2Service, IntegrationError
```

## Stateful assessment flow

1. Register a pseudonymous learner with a grade from 1 to 5.
2. Start a session. C2 resumes the most recent `ACTIVE` or `PAUSED` session for
   that learner; otherwise it creates one.
3. Request the next task. If a question is already active, C2 returns that same
   question rather than issuing a second one.
4. Give the returned declarative `robot_task` to the robot/orchestration layer.
5. Submit the matching C1 `AssessmentEvent` and `CubeAssessmentResult`.
6. C2 stores every accepted quality event, but only a valid, completed first
   attempt updates adaptive history.
7. Repeat until `assessment_status.terminal` is true, then export C2 evidence.

Event ingestion is idempotent by `event_id`. An identical duplicate is ignored;
a duplicate with different content is rejected. Learner, session, task, game,
tier, and expected-answer mismatches are also rejected.

## Grade 1–5 curriculum

C2 reasons over eight games and four mathematical areas:

| Area | Games |
| --- | --- |
| Number sense | `COUNTING`, `COMPARISON`, `ORDERING` |
| Number relationships | `PLACE_VALUE` |
| Arithmetic calculation | `ADDITION`, `SUBTRACTION` |
| Multiplicative reasoning | `MULTIPLICATION`, `DIVISION` |

Eligibility is grade- and tier-specific:

| Grade | Tier A | Additional Tier B eligibility |
| --- | --- | --- |
| 1 | Counting, comparison, ordering, addition, subtraction | Counting, comparison |
| 2 | Counting, comparison, ordering, place value, addition, subtraction | The same six games |
| 3 | All eight games | All except multiplication and division |
| 4 | All eight games | All except multiplication |
| 5 | All eight games | All eight games |

Prerequisites require at least two prior exposures: comparison, ordering,
addition, subtraction, and multiplication depend on counting; division depends
on multiplication. Counting and place value have no prerequisite. Eligibility
is grade based; C2 does not infer age or clinical status.

## History features and inference

The runtime model is `TRANSFER_H2_GB_RT_OFF_v1`, an XGBoost classifier using 25
numeric features derived only from responses completed before the current
recommendation. Features summarize learner performance, skill/game exposure,
recent correctness, lagged outcomes, and candidate-skill context. Identifiers,
the current response, and all response-time features are excluded.

Before inference, the loader verifies every declared artifact against
`models/transferable/manifest.json`, uses the feature order declared in
`models/transferable/metadata.json`,
and checks the model output. `model.json` is the predictor. The calibrated
artifact is also integrity checked; it is loaded only when
`calibration_applied_at_runtime` is enabled in metadata (currently `false`).

If model loading or inference cannot provide a valid probability, C2 uses a
Beta–Bernoulli history estimate where evidence exists. With no defensible
probability, it returns `decision_status: "abstained"` and a curriculum-safe
recommendation; it does not invent a neutral probability.

## Recommendation and difficulty policy

- Missing curriculum coverage is selected before performance optimisation.
- Otherwise eligible games are ranked toward a target predicted success
  probability of `0.65`, subject to prerequisites and repetition controls.
- Probabilities at or below `0.40` support reinforcement decisions; values at
  or above `0.75` support advancement decisions when all other gates pass.
- Tier B requires at least three Tier-A observations for the same skill, a
  Tier-A accumulated correct rate of at least `0.75`, and at least two correct
  answers in the most recent three Tier-A observations, in addition to grade
  and prerequisite eligibility.
- Tier A is used unless the decision is `ADVANCE` and Tier B is admissible.
- The same game may not be issued more than four consecutive times.

Model history spans the learner's stored record. Completion evidence is scoped
to the current session.

## Stopping rules

The current assessment policy uses:

- minimum 16 valid first-attempt responses;
- maximum 24 valid first-attempt responses;
- at least two valid observations for every eligible game, increased to three
  for a game whose observed outcomes are mixed;
- at least three valid observations per eligible mathematical area;
- at least two observations before relying on a Beta estimate;
- operational safeguards of 36 task issuances or 60 session minutes.

Once the minimum and all evidence requirements are satisfied, the session is
`COMPLETE`. Reaching 24 valid responses without adequate coverage produces
`STOPPED_INCOMPLETE_EVIDENCE`. Reaching an operational safeguard produces
`STOPPED_OPERATIONAL_LIMIT`. Non-valid and retry events do not count toward
these valid-response thresholds.

## SQLite persistence

By default C2 creates `.c2_runtime/component2.sqlite3` below the process working
directory. Set `C2_DATABASE_PATH` to an explicit file location for a deployment
or test environment. Migrations maintain these C2-owned tables:

- `learners` and `sessions`;
- `recommendations` and `question_instances`;
- `interactions` and `skill_profiles`;
- `profile_exports`.

The database contains C2's durable projection and provenance. A team-wide event
ledger can remain canonical outside this package.

## Environment and commands

Run commands from the repository root:

```powershell
python -m pip install -r backend/requirements.txt
python -m pytest backend/tests/adaptive_questions -q
uvicorn backend.app.main:app --reload
```

Use a disposable database during local integration when required:

```powershell
$env:C2_DATABASE_PATH = Join-Path $PWD ".c2_runtime/c2-integration.sqlite3"
uvicorn backend.app.main:app --reload
```

The C2 test suite covers contracts, curriculum rules, history leakage,
recommendation and stopping policies, model identity, persistence, portability,
and stateful API integration.

## Model and research assets

Runtime assets are stored in `models/transferable/`: `model.json`,
`calibrator.joblib`, `manifest.json`, `metadata.json`, and `evaluation.json`.
Both binary artifacts are repository-managed team artifacts; routine startup
does not require the installer.

Offline training, six research notebooks, simulation tooling, and the two frozen
eight-model experiment CSVs are documented in
`backend/training/adaptive_questions/README.md`. Training is never run during
application startup.

See `INTEGRATION_GUIDE.md` for the executable HTTP workflow, ownership boundaries,
and troubleshooting.
