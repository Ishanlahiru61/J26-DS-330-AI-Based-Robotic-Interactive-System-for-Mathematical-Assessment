# Development workflow

## Branching

`develop` is the integration branch. Create a focused local task branch from an
up-to-date `develop`; do not work directly on `main`.

```bash
git switch develop
git pull origin develop
git switch -c <type>/<short-description>
```

Do not commit secrets, participant data, recordings, private datasets, or model
artifacts. Review `git status` and the complete diff before every commit or push.

## Local checks

From the repository root:

```bash
python -m pip install -r backend/requirements.txt
python -m compileall -q backend/app
python -m pytest backend/tests
```

Future apps should add their own lint, unit-test, and build commands to their
application README and to repeatable scripts under `scripts/development/`.

## Module contributions

Keep API declarations thin, application coordination in services,
framework-independent decisions in domain code, and vendor/hardware/database
details in adapters. Share data through reviewed contracts, not imports from
another module's infrastructure. Put offline training work under
`backend/training/`.
