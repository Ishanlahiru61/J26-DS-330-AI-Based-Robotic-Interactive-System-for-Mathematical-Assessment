"""Step 4-S - sensitivity analysis: how can more DD children be caught?

A. Cut-off sweep: sensitivity/specificity when flagging the worst 5%...50% of each grade.
   The screening cut-off is chosen on TRAIN children only: the smallest flag share whose
   sensitivity reaches the screening target (register E4b). SVM train scores are
   out-of-fold (cross_val_predict) so the choice is not optimistic.
B. Two-stage rule (DSM-5 Criterion A, persistence): score session 1 and session 2
   separately; refer only children flagged in both.
C. More sessions: re-simulate the same children with 1, 2, 4, 6 sessions (each session
   keeps C2's 16-24 questions) and score with the formula.

The test split and the real Decarli test children are not used. Formula scores need no
label fitting, so B and C report train + validation together (more DD children).

Usage: python sensitivity_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_synthetic_trials as gen  # noqa: E402
from build_features import apply_norms, fit_norms, raw_features  # noqa: E402
from model_formula import ARTIFACTS, RESULTS, SEED, risk_index, screening_metrics  # noqa: E402
from model_svm import FEATURES, make_pipeline  # noqa: E402
from preprocess import preprocess  # noqa: E402

TARGET_SENSITIVITY = 0.70          # lower bound of the screening standard (register E4b)
FLAG_PERCENTILES = [95, 90, 84, 80, 75, 70, 65, 60, 55, 50]  # flag children at or above this percentile
SESSIONS = [1, 2, 4, 6]


def flag(scored: pd.DataFrame, percentile: float, ref: pd.DataFrame | None = None) -> pd.Series:
    """Flag children at or above the same-grade percentile of the reference (train) children."""
    ref = scored[scored["split"] == "train"] if ref is None else ref
    cut = {g: np.nanpercentile(d["risk_index"], percentile) for g, d in ref.groupby("grade")}
    return scored["risk_index"] >= scored["grade"].map(cut)


def sweep(scored: pd.DataFrame, splits=("train", "validation")) -> list[dict]:
    rows = []
    for p in FLAG_PERCENTILES:
        flagged = flag(scored, p)
        row = {"flag_worst_pct": 100 - p}
        for s in splits:
            m = scored["split"] == s
            row[s] = screening_metrics(scored.loc[m, "dd_label"].to_numpy(), flagged[m].to_numpy())
        rows.append(row)
    return rows


def choose_cutoff(rows: list[dict], target: float = TARGET_SENSITIVITY) -> dict | None:
    """Smallest flag share whose TRAIN sensitivity reaches the target."""
    for row in sorted(rows, key=lambda r: r["flag_worst_pct"]):
        if (row["train"]["sensitivity"] or 0) >= target:
            return row
    return None


def load_features(path: Path = ARTIFACTS / "features_v1.csv") -> pd.DataFrame:
    features = pd.read_csv(path)
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    features["dd_label"] = features["student_id"].map(labels)
    return features[(features["split"] != "test") & (features["sufficient_data_area"] == 1)].reset_index(drop=True)


def part_a(features: pd.DataFrame) -> dict:
    formula = pd.concat([features, risk_index(features)], axis=1)

    params = json.loads((RESULTS / "model_svm_metrics.json").read_text(encoding="utf-8"))["best_params"]
    train = features["split"] == "train"
    svm = make_pipeline().set_params(**params)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    score = np.empty(len(features))
    score[train.to_numpy()] = cross_val_predict(svm, features.loc[train, FEATURES], features.loc[train, "dd_label"],
                                                cv=cv, method="decision_function")
    svm.fit(features.loc[train, FEATURES], features.loc[train, "dd_label"])
    score[~train.to_numpy()] = svm.decision_function(features.loc[~train, FEATURES])
    svm_scored = features.assign(risk_index=score)

    out = {}
    for name, scored in (("formula", formula), ("svm", svm_scored)):
        rows = sweep(scored)
        out[name] = {"sweep": rows, "screening_cutoff": choose_cutoff(rows)}
    return out


def features_from_trials(trials: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    children, by_task, by_area, _ = preprocess(trials, labels)
    raw = raw_features(children, by_task, by_area)
    features = apply_norms(raw, fit_norms(raw))
    features["dd_label"] = features["student_id"].map(labels)
    features = pd.concat([features, risk_index(features)], axis=1)
    return features[(features["split"] != "test") & (features["sufficient_data_area"] == 1)].reset_index(drop=True)


def summarise(scored: pd.DataFrame, flagged: pd.Series) -> dict:
    y = scored["dd_label"].to_numpy()
    return {"auc": round(float(roc_auc_score(y, scored["risk_index"])), 3),
            **screening_metrics(y, flagged.to_numpy())}


def part_b(screen_pct: float) -> dict:
    trials = pd.read_csv(ARTIFACTS / "synthetic_trials_v1.csv")
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    s1 = features_from_trials(trials[trials["session_index"] == 1], labels).set_index("student_id")
    s2 = features_from_trials(trials[trials["session_index"] == 2], labels).set_index("student_id")
    both = features_from_trials(trials, labels).set_index("student_id")
    ids = s1.index.intersection(s2.index).intersection(both.index)
    s1, s2, both = s1.loc[ids], s2.loc[ids], both.loc[ids]

    f1, f2, fb = flag(s1, screen_pct), flag(s2, screen_pct), flag(both, screen_pct)
    return {
        "children": int(len(ids)), "dd": int(both["dd_label"].sum()),
        "session_1_only": summarise(s1, f1),
        "both_sessions_pooled": summarise(both, fb),
        "two_stage_flag_s1_and_s2": summarise(both, f1 & f2),
        "flag_s1_or_s2": summarise(both, f1 | f2),
    }


def part_c(screen_pct: float) -> list[dict]:
    rows = []
    for k in SESSIONS:
        gen.P["sessions_per_child"]["value"] = k
        rng = np.random.default_rng(gen.P["seed"])
        children = gen.select_children(rng)
        trials = gen.simulate_answers(gen.build_questions(children, rng), children, rng)
        labels = children.set_index("child_id")["dd_label"]
        scored = features_from_trials(trials, labels)
        row = {"sessions": k, "max_questions_per_child": 24 * k,
               "at_84th_percentile": summarise(scored, flag(scored, 84)),
               "at_screening_cutoff": summarise(scored, flag(scored, screen_pct))}
        rows.append(row)
        print(f"sessions={k}: {row}")
    return rows


def main() -> None:
    a = part_a(load_features())
    chosen = a["formula"]["screening_cutoff"]
    screen_pct = 100 - chosen["flag_worst_pct"] if chosen else 84
    report = {
        "target_sensitivity": TARGET_SENSITIVITY,
        "screening_percentile_formula": screen_pct,
        "A_cutoff_sweep": a,
        "B_two_stage": part_b(screen_pct),
        "C_more_sessions": part_c(screen_pct),
        "note": "Synthetic data; test split and real Decarli test children not used. Cut-offs unvalidated.",
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "sensitivity_analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "A_cutoff_sweep"}, indent=2, default=str))
    for name in ("formula", "svm"):
        print(name, "chosen:", a[name]["screening_cutoff"])


if __name__ == "__main__":
    main()
