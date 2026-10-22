# React dashboard (prototype)

Colourful, interactive teacher / parent dashboard for Component 4 (React + TypeScript + Tailwind +
Recharts + Framer Motion). It shows the component's real outputs for **synthetic demo children** from the
held-out test split. "Demo class A–D" are not real classes: each grade's test children are split into
groups of about 37 to simulate classrooms (proposal: 30–40 students). Every page carries the
non-diagnostic disclaimer.

## Run

```
cd backend/training/risk_profiling/synthetic_phase/dashboard-react
npm install          # first time only
npm run dev          # opens on http://localhost:5173
```

## Refresh the data (after re-running the models)

```
cd backend/training/risk_profiling/synthetic_phase
python dashboard-react/export_data.py     # writes public/data/dashboard.json (~4 min: SHAP for every child)
```

## Pages

| Page | Shows |
|---|---|
| Class overview | Animated KPI cards, mastery-tier donut, risk levels, skill levels, clickable skill heat-map, early-warning cards |
| Child view | Profile banner with "needs help" gauge, skill radar vs peers, session trend, exact risk contributions, SHAP reasons, teacher / parent report with copy button, Component 2 activity tiles, search |
| Practice groups | Flagged children grouped by difficulty profile with activities to practise together |
| Model evidence | Mastery and risk model comparisons, confusion matrix, ablation, final test by grade |

Dynamic profile updates after every session
and access control are not yet included.
