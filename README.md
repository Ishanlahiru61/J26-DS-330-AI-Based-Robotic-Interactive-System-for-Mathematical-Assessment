# J26-DS-330 — AI-Based Robotic Interactive System

This monorepo contains the shared foundation for an AI-assisted, cube-based
mathematics assessment and early dyscalculia screening support system.

> This system provides educational screening support only. Its outputs are not
> a clinical diagnosis and must not be presented as one.

## Architecture

The project uses a **Hybrid Edge-Enabled Client–Server Modular Monolith
Architecture**:

- one monorepo for mobile, dashboard, backend, edge, training, and documentation;
- a Python/FastAPI modular-monolith backend with one shared database initially;
- local edge processing for camera input, computer vision, behaviour tracking,
  and robotic-arm control;
- REST APIs for immediate operations and internal events/background tasks for
  long-running processing and profile updates;
- Android MVVM for the child-facing mobile app; and
- a component-based teacher dashboard.

Each backend research module follows `API -> Service -> Domain -> Infrastructure`.
Model training is isolated under `backend/training/` and is never performed by
the runtime API.

## Research components

1. **Cube Vision and Answer Validation** (`cube_vision`) detects and interprets
   cube arrangements, validates answers, records assessment measures, and can
   produce inputs for robot demonstrations.
2. **Adaptive Question Recommendation** (`adaptive_questions`) estimates
   first-attempt success and recommends curriculum- and age-appropriate tasks
   from chronological assessment history.
3. **Child Behaviour and Motor-Control Analysis** (`behavior_analysis`) analyses
   hand landmarks, finger states, movement, head pose, and coarse gaze, returning
   an uncertain state when visibility is insufficient.
4. **Learning-Risk Profiling** (`risk_profiling`) combines performance,
   recommendation, behaviour, and history into explainable educational
   screening indicators and next-action guidance.

## Repository layout

```text
apps/                  Android and teacher-dashboard applications
backend/app/           FastAPI modular-monolith runtime
backend/tests/         Unit, integration, and end-to-end tests
backend/training/      Offline model training workspaces
data/                  Versioned schemas and safe synthetic samples
docs/                  Architecture, contracts, workflow, and governance
scripts/               Development, training, and evaluation automation
```

Detailed component ownership and data flow are documented in
[`docs/architecture.md`](docs/architecture.md) and
[`docs/data-flow.md`](docs/data-flow.md).

## Backend setup

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r backend/requirements.txt
cp .env.example .env
uvicorn backend.app.main:app --reload
```

Check the service at `http://127.0.0.1:8000/health`.

Run tests from the repository root:

```bash
python -m pytest backend/tests
```

## Branch workflow

`develop` is the integration branch. Create task branches from it, keep changes
local until reviewed, and merge through the team's agreed review workflow.

```bash
git switch develop
git switch -c <type>/<short-description>
```

Never commit directly to `main`. This scaffold does not require a remote push.

## Privacy and research safety

- Use pseudonymous child identifiers; never store names or identifying details.
- Do not commit raw child video, consent forms, medical records, secrets, private
  datasets, or trained model artifacts.
- Keep camera, MediaPipe, cube detection, behaviour classification, and robot
  control on the hardware-connected computer whenever possible.
- Send derived observations instead of continuous raw video.
- Risk thresholds are configuration, not hardcoded clinical cutoffs. They remain
  unvalidated until the research validation process is completed.
- Apply data minimisation, access controls, retention limits, and documented
  consent/ethics approval before collecting participant data.

See [`docs/privacy-and-ethics.md`](docs/privacy-and-ethics.md) and
[`docs/screening-boundary.md`](docs/screening-boundary.md).
