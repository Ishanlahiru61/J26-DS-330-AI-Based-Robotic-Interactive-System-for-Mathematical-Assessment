"""Figures for presentations and the thesis, drawn only from saved results (no new analysis).

Writes PNG files to results/figures/:
 1 model_comparison.png      AUC (95% CI) of the five models, validation and real Decarli
 2 roc_final.png             ROC curves of the selected formula on the held-out test data
 3 sessions_effect.png       AUC and sensitivity by number of sessions (4-S C)
 4 cutoff_tradeoff.png       sensitivity / specificity by flag share (4-S A, formula)
 5 weights_evidence_vs_learned.png   evidence weights vs logistic-regression weights (shares)
 6 shap_skills.png           mean |SHAP| per skill (XGBoost, validation)
 7 final_by_grade.png        test AUC (95% CI) by grade
 8 profiles_dd_share.png     rule-based profiles: children and DD share

Usage: python make_figures.py   (run after final_evaluation.py)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_formula import ARTIFACTS, DECARLI, REPO, RESULTS, risk_index  # noqa: E402
from profiles import rule_profiles, screening_flag  # noqa: E402

OUT = RESULTS / "figures"
BLUE, ORANGE, GREY, DARK = "#2f6db5", "#e07b28", "#9a9a9a", "#333333"
MODELS = [("formula", "Evidence formula\n(selected)"), ("svm", "SVM"), ("random_forest", "Random Forest"),
          ("xgboost", "XGBoost"), ("logistic", "Logistic\nregression")]
SKILL_NAMES = {"arithmetic": "Arithmetic", "counting": "Counting",
               "number_comparison": "Comparing written numbers", "quantity_comparison": "Comparing amounts"}
NOTE = "Synthetic development data; not validated on Sri Lankan children."

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 13, "figure.dpi": 150})


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def save(fig, name: str, note: str = NOTE) -> None:
    fig.text(0.01, 0.01, note, fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)


def model_comparison() -> None:
    sel = load("model_selection.json")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, (key, label) in enumerate(MODELS):
        v = sel["validation"]["models"][key]
        r = sel["real_decarli_calibration"][key]
        for offset, auc, ci, colour in ((-0.12, v["auc"], v["auc_ci95"], BLUE), (0.12, r["auc"], r["ci95"], ORANGE)):
            ax.errorbar(i + offset, auc, yerr=[[auc - ci[0]], [ci[1] - auc]], fmt="o", color=colour,
                        capsize=4, markersize=7, lw=1.5)
    ax.set_xticks(range(len(MODELS)), [m[1] for m in MODELS])
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.text(4.45, 0.51, "chance", color=GREY, fontsize=9, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("AUC (95% CI)")
    ax.plot([], [], "o", color=BLUE, label="Synthetic validation (445 children, 23 DD)")
    ax.plot([], [], "o", color=ORANGE, label="Real Decarli children (50, 25 DD)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("All five models perform alike; the explainable formula was selected")
    save(fig, "model_comparison.png")


def roc_final() -> None:
    final = load("final_evaluation.json")
    preds = pd.read_csv(ARTIFACTS / "predictions_formula_v1.csv")
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    test = preds[(preds["split"] == "test") & (preds["band"] != "insufficient")]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    y = test["student_id"].map(labels)
    fpr, tpr, _ = roc_curve(y, test["risk_index"])
    auc = final["synthetic_test"]["auc"]
    ax.plot(fpr, tpr, color=BLUE, lw=2, label=f"Synthetic test (n=436): AUC {roc_auc_score(y, test['risk_index']):.2f} "
                                                f"({auc['ci95'][0]:.2f}-{auc['ci95'][1]:.2f})")
    real_path = REPO / "data" / "private" / "risk_profiling" / "real_decarli_test_z.csv"
    if real_path.exists():
        real = pd.read_csv(real_path).rename(columns=DECARLI)
        real[list(DECARLI.values())] = real[list(DECARLI.values())].clip(-3, 3)
        score = risk_index(real)["risk_index"]
        fpr, tpr, _ = roc_curve(real["dd_label"], score)
        ra = final["real_decarli_test"]["auc"]
        ax.plot(fpr, tpr, color=ORANGE, lw=2, drawstyle="steps-post",
                label=f"Real Decarli test (n=14): AUC {roc_auc_score(real['dd_label'], score):.2f} "
                      f"({ra['ci95'][0]:.2f}-{ra['ci95'][1]:.2f})")
    s = final["synthetic_test"]["rules"]["screening_flag"]
    ax.plot(1 - s["specificity"], s["sensitivity"], "s", color=DARK, markersize=8,
            label=f"Screening cut-off: sens {s['sensitivity']:.2f}, spec {s['specificity']:.2f}")
    ax.plot([0, 1], [0, 1], color=GREY, ls=":", lw=1)
    ax.set_xlabel("False-alarm rate (1 - specificity)")
    ax.set_ylabel("Sensitivity (DD children caught)")
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    ax.set_title("Final test on unseen children")
    save(fig, "roc_final.png")


def sessions_effect() -> None:
    rows = load("sensitivity_analysis.json")["C_more_sessions"]
    k = [r["sessions"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(k, [r["at_screening_cutoff"]["auc"] for r in rows], "o-", color=BLUE, lw=2, label="AUC")
    ax.plot(k, [r["at_screening_cutoff"]["sensitivity"] for r in rows], "s-", color=ORANGE, lw=2,
            label="Sensitivity at screening cut-off")
    ax.axvline(2, color=GREY, ls=":", lw=1)
    ax.text(2.08, 0.62, "current plan", color=GREY, fontsize=9)
    ax.set_xticks(k)
    ax.set_xlabel("Robot sessions per child (each 16-24 questions)")
    ax.set_ylim(0.6, 0.85)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("About 4 sessions give the largest gain")
    save(fig, "sessions_effect.png")


def cutoff_tradeoff() -> None:
    sweep = load("sensitivity_analysis.json")["A_cutoff_sweep"]["formula"]["sweep"]
    share = [r["flag_worst_pct"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.plot(share, [r["validation"]["sensitivity"] for r in sweep], "o-", color=ORANGE, lw=2,
            label="Sensitivity (DD caught)")
    ax.plot(share, [r["validation"]["specificity"] for r in sweep], "s-", color=BLUE, lw=2,
            label="Specificity (typical not flagged)")
    for x, name in ((5, "high"), (16, "moderate"), (35, "screening")):
        ax.axvline(x, color=GREY, ls=":", lw=1)
        ax.text(x + 0.6, 0.05, name, color=GREY, fontsize=9, rotation=90, va="bottom")
    ax.set_xlabel("Children flagged: worst ...% of their grade")
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=9)
    ax.set_title("Flagging more children catches more DD children", pad=28)
    save(fig, "cutoff_tradeoff.png", NOTE + " Validation children (23 DD), evidence formula.")


def weights() -> None:
    w = load("model_logistic_metrics.json")["coefficients"]["weights"]
    names = {"z_f1_symbolic_comparison_acc": "F1\nnumber comp.\naccuracy",
             "z_f2_symbolic_comparison_rt": "F2\nnumber comp.\nspeed", "z_f3_arithmetic_acc": "F3\narithmetic\naccuracy",
             "z_f4_arithmetic_rt": "F4\narithmetic\nspeed", "z_f5_counting_rt": "F5\ncounting\nspeed",
             "z_f6_counting_acc": "F6\ncounting\naccuracy", "z_f7_nonsymbolic_comparison_acc": "F7\namount comp.\naccuracy",
             "z_f8_nonsymbolic_comparison_rt": "F8\namount comp.\nspeed"}
    order = sorted(w, key=lambda f: -w[f]["evidence_share"])
    x = range(len(order))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.bar([i - 0.2 for i in x], [w[f]["evidence_share"] * 100 for f in order], 0.4, color=BLUE,
           label="Evidence weight (meta-analysis)")
    ax.bar([i + 0.2 for i in x], [w[f]["learned_share"] * 100 for f in order], 0.4, color=ORANGE,
           label="Learned by logistic regression")
    ax.set_xticks(list(x), [names[f] for f in order], fontsize=9)
    ax.set_ylabel("Share of total weight (%)")
    ax.legend(frameon=False)
    rho = load("model_logistic_metrics.json")["coefficients"]["spearman_learned_vs_evidence"]["rho"]
    ax.set_title(f"Learned weights agree with the published evidence (Spearman rho = {rho:.2f})")
    save(fig, "weights_evidence_vs_learned.png")


def shap_skills() -> None:
    s = load("shap_xgboost_summary.json")["global_importance_validation"]["per_skill"]
    items = sorted(s.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    ax.barh([SKILL_NAMES[k] for k, _ in items], [v for _, v in items], color=BLUE)
    for i, (_, v) in enumerate(items):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("Mean |SHAP| contribution (log-odds), validation children")
    ax.set_title("Which skills drive the risk score (XGBoost + SHAP)")
    save(fig, "shap_skills.png")


def final_by_grade() -> None:
    g = load("final_evaluation.json")["synthetic_test"]["by_grade"]
    grades = sorted(g, key=int)
    preds = pd.read_csv(ARTIFACTS / "predictions_formula_v1.csv")
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    test = preds[(preds["split"] == "test") & (preds["band"] != "insufficient")]
    exact = {str(k): roc_auc_score(d["student_id"].map(labels), d["risk_index"]) for k, d in test.groupby("grade")}
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for i, gr in enumerate(grades):
        a = g[gr]["auc"]
        ax.errorbar(i, a["auc"], yerr=[[a["auc"] - a["ci95"][0]], [a["ci95"][1] - a["auc"]]], fmt="o",
                    color=BLUE if a["auc"] >= 0.6 else ORANGE, capsize=5, markersize=8, lw=1.5)
        ax.text(i + 0.08, a["auc"], f"{exact[gr]:.2f}\n({g[gr]['n_dd']} DD)", fontsize=9, va="center")
    ax.axhline(0.5, color=GREY, ls=":", lw=1)
    ax.set_xticks(range(len(grades)), [f"Grade {gr}" for gr in grades])
    ax.set_ylim(0.2, 1.05)
    ax.set_ylabel("Test AUC (95% CI)")
    ax.set_title("Final test by grade: Grade 3 needs attention in the pilot")
    save(fig, "final_by_grade.png")


def profiles() -> None:
    feats = pd.read_csv(ARTIFACTS / "features_v1.csv").merge(
        pd.read_csv(ARTIFACTS / "predictions_formula_v1.csv")[["student_id", "risk_index"]], on="student_id")
    labels = pd.read_csv(ARTIFACTS / "synthetic_trials_v1_children.csv").set_index("child_id")["dd_label"]
    flagged = feats[screening_flag(feats, 35) & (feats["split"] == "train")]
    stats = flagged.groupby(rule_profiles(flagged))["student_id"].agg(lambda ids: (ids.map(labels).mean(), len(ids)))
    names = {"no_single_area": "No single area", "number_relationships": "Number relationships",
             "number_sense": "Number sense", "arithmetic": "Calculation", "multiple_areas": "Several areas"}
    keys = ["no_single_area", "number_relationships", "number_sense", "arithmetic", "multiple_areas"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    shares = [stats[k][0] * 100 for k in keys]
    ax.bar([names[k] for k in keys], shares, color=[ORANGE if k == "multiple_areas" else BLUE for k in keys])
    for i, k in enumerate(keys):
        ax.text(i, shares[i] + 1, f"{shares[i]:.0f}%\n({stats[k][1]} children)", ha="center", fontsize=9)
    ax.set_ylabel("Share of children who are DD (%)")
    ax.set_ylim(0, 55)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_title("Difficulty profiles: weakness in several areas signals DD most")
    save(fig, "profiles_dd_share.png", NOTE + " 725 flagged training children.")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for make in (model_comparison, roc_final, sessions_effect, cutoff_tradeoff, weights, shap_skills,
                 final_by_grade, profiles):
        make()
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
