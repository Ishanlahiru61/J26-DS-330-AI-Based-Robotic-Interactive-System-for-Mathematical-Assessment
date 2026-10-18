# C2 — History-Aware Adaptive Game Recommendation

This directory is the single authoritative C2 runtime in the team repository.
It contains the finalized Version 10 engine: learner/session persistence,
strict-prior XGBoost inference, Beta–Bernoulli fallback, curriculum filtering,
question generation, robot-task output, and the 16–24-valid-response policy.

The module provides educational assessment evidence. It does not diagnose
dyscalculia, consume C3 behavior classifications, run C4 risk inference, or
control robot motors.

## Runtime layout

- `core_service.py`: framework-independent stateful `Component2Service`.
- `service.py`: async team application boundary and retained legacy protocol.
- `contracts/team_adapter.py`: strict C1/team-contract mapping.
- `domain/`: curriculum and generated mathematical questions.
- `feature_engineering/`: strictly prior learner-state features.
- `prediction/`: transferable XGBoost loader, integrity checks, and fallback.
- `recommendation/`: coverage, ranking, progression, tier, and stopping policy.
- `infrastructure/`: SQLite persistence, events, errors, and resource paths.
- `config/`, `migrations/`, `models/`: versioned runtime resources.

## Setup

Install `backend/requirements.txt`, then install the approved model binary with
`backend/training/adaptive_questions/install_runtime_model.py`. The expected
model SHA-256 is
`dc31acb1307568b9f7886d98cf163650ba2eb86ed59c938f28af183cc93403f2`.
Set `C2_DATABASE_PATH` to choose the SQLite file; otherwise C2 uses
`.c2_runtime/component2.sqlite3` below the process working directory.

```powershell
python -m pytest backend/tests/adaptive_questions
uvicorn backend.app.main:app --reload
```

The public portable import is:

```python
from backend.app.modules.adaptive_questions import Component2Service, IntegrationError
```

When the package directory is copied into another Python application, the same
core is importable as `from adaptive_questions import Component2Service`.
`api.py`, `service.py`, and `contracts/team_adapter.py` are the only team-facing
layers; the core itself does not import FastAPI or shared team contracts.

See `INTEGRATION_GUIDE.md` for the stateful workflow and contract semantics.
