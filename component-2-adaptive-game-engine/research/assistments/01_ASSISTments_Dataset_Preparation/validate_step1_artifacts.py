"""Independent post-write validation for Step-1 Parquet artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
BROAD = ROOT / "data" / "processed" / "ASSISTments_2012_2013_cleaned.parquet"
CORE = ROOT / "data" / "processed" / "ASSISTments_2012_2013_core.parquet"

FLAG_COLUMNS = [
    "flag_start_time_parse_failed",
    "flag_end_time_parse_failed",
    "flag_negative_timestamp_duration",
    "flag_nonpositive_response_time",
    "flag_missing_skill_both",
    "flag_problem_log_id_mismatch",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    profile = json.loads((OUTPUT / "ASSISTments_Step1_Full_Profile.json").read_text(encoding="utf-8"))
    expected_rows = profile["dimensions"]["rows"]
    broad_pf, core_pf = pq.ParquetFile(BROAD), pq.ParquetFile(CORE)
    flag_counts = {name: 0 for name in FLAG_COLUMNS}
    core_required_missing = {
        name: 0 for name in [
            "problem_log_id", "user_id", "problem_id", "assistment_id", "correct",
            "attempt_count", "ms_first_response", "start_time_parsed", "end_time_parsed",
        ]
    }
    correct_min = correct_max = None
    response_min = response_max = None

    for batch in core_pf.iter_batches(
        columns=list(core_required_missing) + FLAG_COLUMNS,
        batch_size=200_000,
    ):
        frame = batch.to_pandas()
        for name in FLAG_COLUMNS:
            flag_counts[name] += int(frame[name].fillna(False).sum())
        for name in core_required_missing:
            core_required_missing[name] += int(frame[name].isna().sum())
        cmin, cmax = frame["correct"].min(), frame["correct"].max()
        rmin, rmax = frame["ms_first_response"].min(), frame["ms_first_response"].max()
        correct_min = cmin if correct_min is None else min(correct_min, cmin)
        correct_max = cmax if correct_max is None else max(correct_max, cmax)
        response_min = rmin if response_min is None else min(response_min, rmin)
        response_max = rmax if response_max is None else max(response_max, rmax)

    first_broad = broad_pf.read_row_group(0, columns=["problem_log_id"]).column(0)[0].as_py()
    first_core = core_pf.read_row_group(0, columns=["problem_log_id"]).column(0)[0].as_py()
    last_broad_group = broad_pf.metadata.num_row_groups - 1
    last_core_group = core_pf.metadata.num_row_groups - 1
    last_broad_column = broad_pf.read_row_group(last_broad_group, columns=["problem_log_id"]).column(0)
    last_core_column = core_pf.read_row_group(last_core_group, columns=["problem_log_id"]).column(0)
    last_broad, last_core = last_broad_column[-1].as_py(), last_core_column[-1].as_py()

    checks = [
        ["broad_rows", broad_pf.metadata.num_rows, expected_rows, broad_pf.metadata.num_rows == expected_rows],
        ["core_rows", core_pf.metadata.num_rows, expected_rows, core_pf.metadata.num_rows == expected_rows],
        ["broad_columns", broad_pf.metadata.num_columns, 44, broad_pf.metadata.num_columns == 44],
        ["core_columns", core_pf.metadata.num_columns, 36, core_pf.metadata.num_columns == 36],
        ["first_id_alignment", first_core, first_broad, first_core == first_broad],
        ["last_id_alignment", last_core, last_broad, last_core == last_broad],
        ["correct_min", float(correct_min), 0.0, float(correct_min) == 0.0],
        ["correct_max", float(correct_max), 1.0, float(correct_max) == 1.0],
        ["response_min", int(response_min), profile["response_time"]["min_ms"], int(response_min) == profile["response_time"]["min_ms"]],
        ["response_max", int(response_max), profile["response_time"]["max_ms"], int(response_max) == profile["response_time"]["max_ms"]],
        ["negative_duration_flags", flag_counts["flag_negative_timestamp_duration"], profile["timestamps"]["negative_durations"], flag_counts["flag_negative_timestamp_duration"] == profile["timestamps"]["negative_durations"]],
        ["nonpositive_response_flags", flag_counts["flag_nonpositive_response_time"], profile["response_time"]["negative_count"], flag_counts["flag_nonpositive_response_time"] == profile["response_time"]["negative_count"]],
        ["missing_skill_both_flags", flag_counts["flag_missing_skill_both"], profile["skills"]["relationship"]["skill_missing_skill_id_missing"], flag_counts["flag_missing_skill_both"] == profile["skills"]["relationship"]["skill_missing_skill_id_missing"]],
        ["problem_log_mismatch_flags", flag_counts["flag_problem_log_id_mismatch"], 0, flag_counts["flag_problem_log_id_mismatch"] == 0],
        ["start_parse_failure_flags", flag_counts["flag_start_time_parse_failed"], 0, flag_counts["flag_start_time_parse_failed"] == 0],
        ["end_parse_failure_flags", flag_counts["flag_end_time_parse_failed"], 0, flag_counts["flag_end_time_parse_failed"] == 0],
        ["core_required_missing_total", sum(core_required_missing.values()), 0, sum(core_required_missing.values()) == 0],
    ]
    checks_df = pd.DataFrame(checks, columns=["check", "observed", "expected", "passed"])
    checks_df["status"] = checks_df["passed"].map({True: "PASS", False: "FAIL"})
    checks_df.to_csv(OUTPUT / "ASSISTments_Step1_Independent_Validation.csv", index=False)

    manifest_rows = [
        [BROAD, "Broad cleaned dataset", BROAD.stat().st_size, sha256(BROAD)],
        [CORE, "Core chronology-ready dataset (no history features)", CORE.stat().st_size, sha256(CORE)],
        [ROOT / "01_ASSISTments_Dataset_Preparation.ipynb", "Updated Step-1 research notebook", (ROOT / "01_ASSISTments_Dataset_Preparation.ipynb").stat().st_size, sha256(ROOT / "01_ASSISTments_Dataset_Preparation.ipynb")],
        [ROOT / "run_step1_pipeline.py", "Reproducible chunked pipeline", (ROOT / "run_step1_pipeline.py").stat().st_size, sha256(ROOT / "run_step1_pipeline.py")],
        [ROOT / "generate_step1_report_and_notebook.py", "Supplemental QA/report/notebook generator", (ROOT / "generate_step1_report_and_notebook.py").stat().st_size, sha256(ROOT / "generate_step1_report_and_notebook.py")],
        [ROOT / "validate_step1_artifacts.py", "Independent artifact validator", (ROOT / "validate_step1_artifacts.py").stat().st_size, sha256(ROOT / "validate_step1_artifacts.py")],
        [OUTPUT / "ASSISTments_Step1_Data_Quality_Report.md", "Human-readable Step-1 report", (OUTPUT / "ASSISTments_Step1_Data_Quality_Report.md").stat().st_size, sha256(OUTPUT / "ASSISTments_Step1_Data_Quality_Report.md")],
        [OUTPUT / "ASSISTments_Step1_Data_Dictionary.csv", "35-column data dictionary", (OUTPUT / "ASSISTments_Step1_Data_Dictionary.csv").stat().st_size, sha256(OUTPUT / "ASSISTments_Step1_Data_Dictionary.csv")],
        [OUTPUT / "ASSISTments_Step1_Missingness.csv", "Full-dataset missingness", (OUTPUT / "ASSISTments_Step1_Missingness.csv").stat().st_size, sha256(OUTPUT / "ASSISTments_Step1_Missingness.csv")],
        [OUTPUT / "ASSISTments_Step1_Cleaning_Rules.csv", "Auditable cleaning rules", (OUTPUT / "ASSISTments_Step1_Cleaning_Rules.csv").stat().st_size, sha256(OUTPUT / "ASSISTments_Step1_Cleaning_Rules.csv")],
    ]
    freeze_summary = OUTPUT / "ASSISTments_Step1_Final_Freeze_Summary.md"
    if freeze_summary.exists():
        manifest_rows.append([
            freeze_summary,
            "Final Step-1 freeze decision and audit summary",
            freeze_summary.stat().st_size,
            sha256(freeze_summary),
        ])
    pd.DataFrame(manifest_rows, columns=["path", "role", "bytes", "sha256"]).to_csv(
        OUTPUT / "ASSISTments_Step1_Artifact_Manifest.csv", index=False
    )
    result = {
        "all_checks_passed": bool(checks_df["passed"].all()),
        "flag_counts": flag_counts,
        "core_required_missing": core_required_missing,
        "checks": checks_df.to_dict(orient="records"),
    }
    (OUTPUT / "ASSISTments_Step1_Independent_Validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if not result["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
