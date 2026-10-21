"""Reproduce DD-vs-control effect sizes from the Decarli et al. (2020) dataset.

Output: results/decarli2020_effect_sizes.csv, one row per measure, used
as evidence in docs/feature_evidence.md. Sign convention: positive d means the
DD group performs WORSE (lower accuracy / slower / higher Weber fraction).
"""

from pathlib import Path

import numpy as np
import pandas as pd

MODULE = Path(__file__).resolve().parents[1]
REPO = Path(__file__).resolve().parents[5]
SOURCE = REPO / "data" / "raw" / "decarli2020" / "Complete_Dataframe_def.xlsx"
OUTPUT = MODULE / "results" / "decarli2020_effect_sizes.csv"

# column -> (readable name, True if a higher value means worse performance)
MEASURES = {
    "Mean_ACC_13": ("Enumeration 1-3 accuracy (subitizing)", False),
    "MeanRT_13": ("Enumeration 1-3 response time (subitizing)", True),
    "Mean_ACC_47": ("Enumeration 4-7 accuracy (counting range)", False),
    "MeanRT_47": ("Enumeration 4-7 response time (counting range)", True),
    "Mean_ACC_ANS": ("Non-symbolic comparison accuracy (ANS)", False),
    "Mean_RTs_ANS": ("Non-symbolic comparison response time (ANS)", True),
    "w_fraction_def": ("Weber fraction (ANS acuity)", True),
    "Accuracy_SymbolicComp": ("Symbolic digit comparison accuracy", False),
    "RTs_SymbolicComp": ("Symbolic digit comparison response time", True),
    "wm_K": ("Visuo-spatial short-term memory (Cowan's K)", False),
    "Average_Corsi": ("Corsi block span, average", False),
    "Similarities_IQ": ("WISC Similarities (verbal IQ)", False),
    "MatrixReas_IQ": ("WISC Matrix Reasoning (non-verbal IQ)", False),
}


def cohens_d(worse: pd.Series, better: pd.Series) -> float:
    worse, better = worse.dropna(), better.dropna()
    pooled_var = ((len(worse) - 1) * worse.var() + (len(better) - 1) * better.var()) / (
        len(worse) + len(better) - 2
    )
    return (worse.mean() - better.mean()) / np.sqrt(pooled_var)


def main() -> None:
    frame = pd.read_excel(SOURCE)
    frame["w_fraction_def"] = pd.to_numeric(frame["w_fraction_def"], errors="coerce")
    dd = frame[frame["group"] == "DD"]
    control = frame[frame["group"] == "contr"]

    rows = []
    for column, (name, higher_is_worse) in MEASURES.items():
        d = cohens_d(dd[column], control[column])
        rows.append(
            {
                "column": column,
                "measure": name,
                "dd_mean": round(dd[column].mean(), 3),
                "control_mean": round(control[column].mean(), 3),
                "n_dd": int(dd[column].notna().sum()),
                "n_control": int(control[column].notna().sum()),
                "d_dd_worse": round(d if higher_is_worse else -d, 2),
            }
        )

    result = pd.DataFrame(rows).sort_values("d_dd_worse", ascending=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
