"""Step 5 (part 1) - rule-based difficulty profiles.

The risk band says how much concern there is; a profile says what kind. Profiles are given only to
children flagged at the screening cut-off (worst 35% of their grade, register F5).

Inputs: the C2 area features (Step 3), each as accuracy z and speed z (positive = worse):
number_sense (counting, place value), number_relationships (comparison, ordering),
arithmetic (addition, subtraction). multiplicative_reasoning is Grade 3 only and is not used
for profiles (it would be missing for Grades 1-2).

Area z = mean of the area's available accuracy and speed z (equal weights: no published weights
exist at area level; assumption, register S2). An area is weak if area z >= 1.0 (16th percentile,
register F1). Profile:
    one weak area      -> that area
    two or more        -> "multiple_areas"
    none               -> "no_single_area" (flagged by the overall score, no single area below the cut-off)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

AREAS = ["number_sense", "number_relationships", "arithmetic"]
AREA_COLUMNS = [f"z_area_{a}_{m}" for a in AREAS for m in ("acc", "rt")]
WEAK_Z = 1.0  # register F1


def area_z(features: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({a: features[[f"z_area_{a}_acc", f"z_area_{a}_rt"]].mean(axis=1) for a in AREAS},
                        index=features.index)


def rule_profile(zs: pd.Series) -> str:
    weak = [a for a in AREAS if pd.notna(zs[a]) and zs[a] >= WEAK_Z]
    if len(weak) == 1:
        return weak[0]
    return "multiple_areas" if weak else "no_single_area"


def rule_profiles(features: pd.DataFrame) -> pd.Series:
    return area_z(features).apply(rule_profile, axis=1)


def screening_flag(scored: pd.DataFrame, flag_worst_pct: float) -> pd.Series:
    """Flag children at or above the same-grade TRAIN percentile of the formula score (4-S)."""
    train = scored[scored["split"] == "train"]
    cut = {g: np.nanpercentile(d["risk_index"], 100 - flag_worst_pct) for g, d in train.groupby("grade")}
    return (scored["risk_index"] >= scored["grade"].map(cut)) & (scored["sufficient_data_area"] == 1)
