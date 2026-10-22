"""Export the component's outputs for the React dashboard (synthetic demo children only).

Uses the data layer in dashboard_data.py, which reuses the component's own code:
risk level and exact skill contributions (evidence formula), difficulty profile, mastery tier and
P(Needs Urgent Assistance) from the selected SVM, SHAP reasons for that prediction, teacher and
parent reports, Component 2 practice suggestions, per-session accuracy and response time, and the
model-evidence summaries.

Usage (from synthetic_phase/):  python dashboard-react/export_data.py
Writes dashboard-react/public/data/dashboard.json
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dashboard_data as data  # noqa: E402
from dashboard_data import ARTIFACTS, MASTERY_FEATURES, SEED, shap_values  # noqa: E402

OUT = HERE / "public" / "data" / "dashboard.json"


def clean(value):
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else round(float(value), 4)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


def mastery_shap(ids: list[str]) -> dict[str, dict[str, float]]:
    import joblib
    mastery = pd.read_csv(ARTIFACTS / "mastery_features_v1.csv")
    model = joblib.load(ARTIFACTS / "mastery_svm_v1.joblib")
    background = mastery[mastery["split"] == "train"][MASTERY_FEATURES].sample(60, random_state=SEED)
    rows = mastery[mastery["student_id"].isin(ids)].set_index("student_id")
    values = shap_values(model, background, rows[MASTERY_FEATURES].astype(float))
    out = {}
    for sid, v in values.iterrows():
        top = v.reindex(v.abs().sort_values(ascending=False).index[:7])
        out[sid] = {data.FEATURE_NAMES[k]: round(float(x), 4) for k, x in top.items()}
    return out


def evidence() -> dict:
    m, f = data.results("mastery_models.json"), data.results("mastery_final_evaluation.json")
    r, a = data.results("final_evaluation.json"), data.results("mastery_ablation.json")
    sel = data.results("model_selection.json")
    return {
        "mastery_validation": [{"model": k, "macro_f1": v["macro_f1"],
                                "urgent_recall": v["recall_per_class"]["Needs Urgent Assistance"]}
                               for k, v in m["validation"].items()],
        "mastery_selected": m["selected"],
        "mastery_test": {k: f["selected_model"][k] for k in ("accuracy", "macro_f1", "recall_per_class",
                                                             "needs_urgent_recall_ci95", "roc_auc_ovr_macro")},
        "mastery_baseline_test": {k: f["accuracy_rule_baseline"][k] for k in ("macro_f1", "recall_per_class")},
        "confusion_matrix": f["selected_model"]["confusion_matrix"]["values"],
        "ablation": [{"model": mn, "set": fs, "macro_f1": x["validation_macro_f1"]}
                     for mn, sets in a["models"].items() for fs, x in sets.items()],
        "risk_models": [{"model": k, "validation_auc": v["auc"], "real_auc": sel["real_decarli_calibration"][k]["auc"]}
                        for k, v in sel["validation"]["models"].items()],
        "risk_test": {"auc": r["synthetic_test"]["auc"], "real_auc": r["real_decarli_test"]["auc"],
                      "screening": r["synthetic_test"]["rules"]["screening_flag"],
                      "by_grade": {g: v["auc"]["auc"] for g, v in r["synthetic_test"]["by_grade"].items()}},
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    df = data.load_children()
    sessions = data.load_sessions(df["student_id"].tolist())
    reasons = mastery_shap(df["student_id"].tolist())
    kids = []
    for _, c in df.iterrows():
        e = c["explanation"]
        teacher, parent = data.reports(e)
        s = sessions[sessions["student_id"] == c["student_id"]].sort_values("session_index")
        kids.append({
            "id": c["student_id"], "grade": int(c["grade"]), "age_months": int(c["age_months"]),
            "class_name": c["class_name"], "tier": c["tier"], "p_urgent": c["p_urgent"],
            "level": c["level"], "level_name": data.LEVEL_NAMES[c["level"]], "flagged": bool(c["flagged"]),
            "profile": data.PROFILE_NAMES[c["profile"]] if c["flagged"] else None,
            "skills": {data.SKILL_NAMES[s_]: {"percentile": c[f"pct_{s_}"], "level": c[f"lvl_{s_}"],
                                              "contribution": e.get("skills", {}).get(s_, {}).get("contribution")}
                       for s_ in data.SKILLS},
            "not_assessed": [data.SKILL_NAMES[x] for x in e.get("not_assessed", [])],
            "risk_index": e.get("risk_index"), "practise": data.practise_for(e),
            "sessions": [{"session": int(r.session_index), "accuracy": r.accuracy, "median_rt": r.median_rt}
                         for r in s.itertuples()],
            "reasons": reasons.get(c["student_id"]),
            "teacher_report": teacher.splitlines(), "parent_report": parent.splitlines(),
            "n_sessions": e["data"]["n_sessions"], "n_questions": e["data"]["n_questions"],
        })
    payload = {
        "meta": {"note": "Synthetic demo children (held-out test split). Screening support, not a diagnosis.",
                 "classes": sorted(df["class_name"].unique()), "skills": list(data.SKILL_NAMES.values())},
        "children": kids, "evidence": evidence(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(clean(payload), ensure_ascii=False), encoding="utf-8")
    print(f"{len(kids)} children -> {OUT} ({OUT.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
