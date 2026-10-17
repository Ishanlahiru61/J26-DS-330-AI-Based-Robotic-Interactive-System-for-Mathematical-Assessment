# Backend

The backend is a FastAPI modular monolith. Research modules own their domain
logic and infrastructure adapters while sharing contracts, configuration,
security, logging, utilities, and one database boundary.

From the repository root:

```bash
python -m pip install -r backend/requirements.txt
uvicorn backend.app.main:app --reload
python -m pytest backend/tests
```

The runtime API does not train models. Offline training belongs in `training/`.
