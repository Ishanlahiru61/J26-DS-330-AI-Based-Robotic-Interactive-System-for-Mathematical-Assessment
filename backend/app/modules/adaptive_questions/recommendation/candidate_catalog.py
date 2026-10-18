from __future__ import annotations

import csv
from pathlib import Path


REQUIRED_COLUMNS = {
    "activity_id", "activity_name", "project_skill", "grade_min", "grade_max", "tier",
    "curriculum_trace_id", "prerequisite_skill", "prerequisite_level", "representation_type",
    "number_range_min", "number_range_max", "operation_type", "step_complexity",
    "requires_regrouping", "allows_remainder", "physical_cube_suitability",
    "robot_suitability", "active", "notes",
}


def _bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def load_catalog(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not REQUIRED_COLUMNS.issubset(rows[0]):
        raise ValueError(f"Catalog missing required columns: {sorted(REQUIRED_COLUMNS - set(rows[0] if rows else []))}")
    seen: set[str] = set()
    for row in rows:
        if row["activity_id"] in seen:
            raise ValueError(f"Duplicate activity_id: {row['activity_id']}")
        seen.add(row["activity_id"])
        row["grade_min"] = int(row["grade_min"])
        row["grade_max"] = int(row["grade_max"])
        row["active"] = _bool(row["active"])
        row["requires_regrouping"] = _bool(row["requires_regrouping"])
        row["allows_remainder"] = _bool(row["allows_remainder"])
    return rows
