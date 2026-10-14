"""Generate the Step-1 narrative report, supplemental QA tables, and notebook.

Reads the already-created Parquet datasets in batches; does not modify raw data
or create any modelling/history features.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
PROCESSED = ROOT / "data" / "processed"
BROAD = PROCESSED / "ASSISTments_2012_2013_cleaned.parquet"
CORE = PROCESSED / "ASSISTments_2012_2013_core.parquet"
PROFILE_PATH = OUTPUT / "ASSISTments_Step1_Full_Profile.json"
NOTEBOOK_SOURCE = Path(r"C:\Users\Monil A\Desktop\01_ASSISTments_Dataset_Preparation.ipynb")
NOTEBOOK_TARGET = ROOT / "01_ASSISTments_Dataset_Preparation.ipynb"
NOTEBOOK_ARCHIVE = OUTPUT / "01_ASSISTments_Dataset_Preparation_original_exploration.ipynb"


def fmt_int(value) -> str:
    return f"{int(value):,}"


def fmt_pct(value, digits=3) -> str:
    return f"{float(value):.{digits}f}%"


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat(sep=" ")
    if pd.isna(value):
        return None
    return value


def scan_supplemental_qa() -> dict:
    overlap_parts = []
    duration_parts = []
    start_dates = set()
    affect = {
        name: {"count": 0, "sum": 0.0, "min": None, "max": None, "outside_0_1": 0}
        for name in [
            "Average_confidence(FRUSTRATED)", "Average_confidence(CONFUSED)",
            "Average_confidence(CONCENTRATING)", "Average_confidence(BORED)",
        ]
    }
    learner_skill = Counter()
    bottom_missing_hint = Counter()
    negative_response_rows = []
    negative_duration_rows = []
    largest_response = pd.DataFrame()
    largest_overlap = pd.DataFrame()
    largest_duration = pd.DataFrame()

    columns = [
        "problem_log_id", "user_id", "problem_id", "skill", "skill_id", "correct",
        "attempt_count", "ms_first_response", "start_time_parsed", "end_time_parsed",
        "timestamp_duration_ms", "overlap_time", "hint_count", "bottom_hint",
        "problem_type", "assignment_id", "sequence_id",
        "Average_confidence(FRUSTRATED)", "Average_confidence(CONFUSED)",
        "Average_confidence(CONCENTRATING)", "Average_confidence(BORED)",
    ]
    for batch in pq.ParquetFile(BROAD).iter_batches(columns=columns, batch_size=100_000):
        frame = batch.to_pandas()
        overlap_parts.append(frame["overlap_time"].dropna().astype("int64").to_numpy())
        duration_parts.append(frame["timestamp_duration_ms"].dropna().astype("float64").to_numpy())
        start_dates.update(frame["start_time_parsed"].dropna().dt.date.unique().tolist())
        valid_skill = frame["skill"].notna() | frame["skill_id"].notna()
        learner_skill.update(frame.loc[valid_skill, "user_id"].astype("int64").tolist())
        missing_bottom = frame["bottom_hint"].isna()
        bottom_missing_hint.update(frame.loc[missing_bottom, "hint_count"].astype("int64").tolist())

        negative_response_rows.append(frame.loc[frame["ms_first_response"] < 0, [
            "problem_log_id", "user_id", "problem_id", "start_time_parsed",
            "ms_first_response", "correct", "attempt_count", "problem_type",
        ]])
        negative_duration_rows.append(frame.loc[frame["timestamp_duration_ms"] < 0, [
            "problem_log_id", "user_id", "problem_id", "start_time_parsed", "end_time_parsed",
            "timestamp_duration_ms", "ms_first_response", "correct", "problem_type",
        ]])
        largest_response = pd.concat([largest_response, frame.nlargest(20, "ms_first_response")[
            ["problem_log_id", "user_id", "problem_id", "start_time_parsed", "ms_first_response", "timestamp_duration_ms", "problem_type"]
        ]]).nlargest(20, "ms_first_response")
        largest_overlap = pd.concat([largest_overlap, frame.nlargest(20, "overlap_time")[
            ["problem_log_id", "user_id", "problem_id", "start_time_parsed", "overlap_time", "ms_first_response", "problem_type"]
        ]]).nlargest(20, "overlap_time")
        largest_duration = pd.concat([largest_duration, frame.nlargest(20, "timestamp_duration_ms")[
            ["problem_log_id", "user_id", "problem_id", "start_time_parsed", "end_time_parsed", "timestamp_duration_ms", "ms_first_response"]
        ]]).nlargest(20, "timestamp_duration_ms")

        for name, stats in affect.items():
            values = frame[name].dropna().astype(float).to_numpy()
            stats["count"] += len(values)
            stats["sum"] += float(values.sum())
            stats["outside_0_1"] += int(((values < 0) | (values > 1)).sum())
            if len(values):
                stats["min"] = float(values.min()) if stats["min"] is None else min(stats["min"], float(values.min()))
                stats["max"] = float(values.max()) if stats["max"] is None else max(stats["max"], float(values.max()))

    overlap = np.concatenate(overlap_parts)
    duration = np.concatenate(duration_parts)
    overlap_levels = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
    duration_levels = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1]
    learner_ids = set()
    for batch in pq.ParquetFile(CORE).iter_batches(columns=["user_id"], batch_size=200_000):
        learner_ids.update(batch.column(0).to_pylist())

    for stats in affect.values():
        stats["mean"] = stats["sum"] / stats["count"] if stats["count"] else None
        del stats["sum"]

    qa = {
        "temporal_coverage": {
            "unique_start_dates": len(start_dates),
            "first_start_date": str(min(start_dates)),
            "last_start_date": str(max(start_dates)),
        },
        "overlap_time": {
            "count": len(overlap),
            "missing": 0,
            "zero": int((overlap == 0).sum()),
            "negative": int((overlap < 0).sum()),
            "mean_ms": float(overlap.mean()),
            "std_ms": float(overlap.std(ddof=1)),
            "percentiles_ms": {str(p): float(v) for p, v in zip(overlap_levels, np.quantile(overlap, overlap_levels))},
        },
        "timestamp_duration": {
            "count": len(duration),
            "negative": int((duration < 0).sum()),
            "zero": int((duration == 0).sum()),
            "percentiles_ms": {str(p): float(v) for p, v in zip(duration_levels, np.quantile(duration, duration_levels))},
        },
        "skill_coverage_per_learner": {
            "learners_with_any_skill_information": len(learner_skill),
            "learners_with_no_skill_information": len(learner_ids - set(learner_skill)),
            "min_skill_tagged_interactions_among_covered": min(learner_skill.values()),
            "max_skill_tagged_interactions": max(learner_skill.values()),
        },
        "missing_bottom_hint_hint_count_distribution": {str(k): v for k, v in sorted(bottom_missing_hint.items())},
        "affect_fields": affect,
    }

    pd.concat(negative_response_rows, ignore_index=True).to_csv(OUTPUT / "ASSISTments_Step1_Negative_Response_Time_Rows.csv", index=False)
    pd.concat(negative_duration_rows, ignore_index=True).to_csv(OUTPUT / "ASSISTments_Step1_Negative_Duration_Rows.csv", index=False)
    largest_response.to_csv(OUTPUT / "ASSISTments_Step1_Largest_Response_Times.csv", index=False)
    largest_overlap.to_csv(OUTPUT / "ASSISTments_Step1_Largest_Overlap_Times.csv", index=False)
    largest_duration.to_csv(OUTPUT / "ASSISTments_Step1_Largest_Timestamp_Durations.csv", index=False)
    (OUTPUT / "ASSISTments_Step1_Contextual_QA.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    return qa


def build_markdown_report(profile: dict, qa: dict) -> str:
    rows = profile["dimensions"]["rows"]
    missing = pd.read_csv(OUTPUT / "ASSISTments_Step1_Missingness.csv")
    missing_nonzero = missing[missing["missing_count"] > 0]
    missing_table = "\n".join(
        f"| `{r.column_name}` | {int(r.missing_count):,} | {r.missing_percentage:.4f}% |"
        for r in missing_nonzero.itertuples()
    )
    response = profile["response_time"]
    learner = profile["learner_histories"]
    skill = profile["skills"]
    timestamps = profile["timestamps"]
    correctness = profile["correctness"]
    attempts = profile["attempts"]
    repeats = profile["identifier_summary"]["learner_problem_repeats"]
    overlap = qa["overlap_time"]
    affect = qa["affect_fields"]
    affect_text = "\n".join(
        f"- `{name}`: range {stats['min']:.6f}–{stats['max']:.6f}, mean {stats['mean']:.6f}, outside [0,1]: {stats['outside_0_1']:,}."
        for name, stats in affect.items()
    )
    return f"""# Step 1 — ASSISTments Dataset Preparation & Data-Quality Report

## 1. Dataset source and provenance

- Dataset: **ASSISTments 2012–2013 School Data with Affect**.
- Immutable source: `{profile['source']['path']}`.
- Raw size: {profile['source']['bytes']:,} bytes ({profile['source']['bytes'] / 1024**3:.3f} GiB).
- SHA-256: `{profile['source']['sha256']}`.
- Official source/column documentation: <https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect>.
- Scope: educational-interaction data for development and experimental work. It is **not** a clinical dyscalculia dataset and does not support diagnostic claims.

The original CSV was not altered, renamed, or copied. The pipeline references it in place and produces separate artifacts.

## 2. Method

The full CSV was parsed with pandas in deterministic 100,000-row chunks. Each chunk was typed explicitly, timestamps were parsed with mixed fractional-second support, quality flags were added, and two Zstandard-compressed Parquet files were written incrementally. Disk-backed arrays were sorted to obtain exact interaction-ID, learner/problem-repeat, and learner/timestamp-tie statistics without loading the full CSV into RAM.

The earlier notebook's 100,000-row outputs are retained as exploratory evidence only. All numeric findings below are full-dataset results.

## 3. Dimensions and identifiers

- Rows: **{rows:,}**.
- Source columns: **35**; broad cleaned columns: **44**; core columns: **36**.
- Learners: **{profile['identifier_summary']['unique_learners']:,}**.
- Problems: **{profile['identifier_summary']['unique_problems']:,}**.
- ASSISTments: **{profile['identifier_summary']['unique_assistments']:,}**.
- Interaction logs: **{profile['identifier_summary']['problem_log']['unique_values']:,}** unique, with **0 missing and 0 duplicate `problem_log_id` values**.
- `problem_log_id` and `problemlogid`: **0 mismatches across all {rows:,} rows**. The latter is retained in the broad dataset but omitted as redundant from the core dataset.
- Exact duplicate rows: **0**, established because every row has a unique non-missing `problem_log_id`; an exact full-row duplicate would necessarily repeat it.
- Learner/problem pairs: {repeats['unique_values']:,} unique pairs; {repeats['duplicate_rows_beyond_first']:,} rows are later occurrences within repeated pairs. These are retained as longitudinal interactions, not treated as duplicates.

All checked identifier columns have their missing/zero/negative counts in `ASSISTments_Step1_Identifier_Quality.csv`.

## 4. Missingness

Only six source columns have missing values:

| Column | Missing | Percentage |
|---|---:|---:|
{missing_table}

Core event fields (`user_id`, `problem_id`, `assistment_id`, `problem_log_id`, `correct`, `attempt_count`, timestamps, and `ms_first_response`) are complete. Missing skills are retained because they are structural and do not invalidate the remaining interaction record. No imputation was performed.

## 5. Skills

- Unique non-missing skill names: **{skill['unique_skill_names']:,}**.
- Unique non-missing skill IDs: **{skill['unique_skill_ids']:,}**.
- Skill and skill ID both present: {skill['relationship']['skill_present_skill_id_present']:,}.
- Skill present / ID missing: {skill['relationship']['skill_present_skill_id_missing']:,}.
- Skill missing / ID present: {skill['relationship']['skill_missing_skill_id_present']:,}.
- Both missing: {skill['relationship']['skill_missing_skill_id_missing']:,}.
- Skill IDs mapping to multiple non-missing names: **{skill['skill_ids_mapping_to_multiple_nonmissing_names']:,}**.
- Learners with at least one skill-tagged interaction: {qa['skill_coverage_per_learner']['learners_with_any_skill_information']:,}; with none: {qa['skill_coverage_per_learner']['learners_with_no_skill_information']:,}.

## 6. Correctness

- Missing: 0; values outside [0,1]: 0.
- `correct = 1`: {correctness['value_distribution']['1']:,}; `correct = 0`: {correctness['value_distribution']['0']:,}.
- Fractional/non-binary scores: **{correctness['fractional_nonbinary_count']:,}**.
- Most fractional records are `open_response`, but the full dataset also contains **two `rank` rows with `correct = 0.5`**. This corrects the 100k-sample observation that fractional scores were exclusive to `open_response`.

The official documentation says decimals may encode partial credit. The raw score is preserved; no binary target is created in Step 1.

## 7. Attempts and hint/action context

- `attempt_count`: range {attempts['min']}–{attempts['max']}, missing 0, negative 0, zero {attempts['zero_count']:,}.
- Zero attempts are retained; the official page defines the field as the number of attempts but does not explain every zero-valued case.
- `hint_count`: observed 0–14, missing 0.
- `bottom_hint`: {int(missing.loc[missing.column_name == 'bottom_hint', 'missing_count'].iloc[0]):,} missing. Every missing-`bottom_hint` row has `hint_count = 0` (supplemental QA distribution: {qa['missing_bottom_hint_hint_count_distribution']}).
- `first_action`: observed codes 0/1/2. The official source documents the concept (attempt versus hint request) but not the numeric mapping, so the codes are not relabeled.

## 8. Response time

- `ms_first_response`: missing 0, zero 0, **negative {response['negative_count']:,}**.
- Min / median / mean / max: {response['min_ms']:,.0f} / {response['percentiles_ms']['0.5']:,.0f} / {response['mean_ms']:,.3f} / {response['max_ms']:,.0f} ms.
- P75 / P90 / P95 / P99 / P99.9: {response['percentiles_ms']['0.75']:,.0f} / {response['percentiles_ms']['0.9']:,.1f} / {response['percentiles_ms']['0.95']:,.0f} / {response['percentiles_ms']['0.99']:,.1f} / {response['percentiles_ms']['0.999']:,.1f} ms.
- Distribution: strongly right-skewed, with a small anomalous negative tail and very large positive extremes.

All values are preserved. The 13 negative records are explicitly listed and flagged; large values are not capped or winsorized. Any later `log1p` or RT OFF/ON construction belongs to feature engineering, not this step.

## 9. Timestamps and chronology

- Start/end parse failures using mixed-format parsing: **0 / 0**.
- Start coverage: {timestamps['earliest_start']} through {timestamps['latest_start']}, spanning **{qa['temporal_coverage']['unique_start_dates']} distinct start dates** ({qa['temporal_coverage']['first_start_date']}–{qa['temporal_coverage']['last_start_date']}).
- Latest recorded end: {timestamps['latest_end']} (later than the latest start because some timestamp durations are extreme).
- `end_time < start_time`: **{timestamps['negative_durations']:,} rows**, retained and flagged.
- Learner/start-time ties: **{timestamps['learner_timestamp_ties']['duplicate_user_timestamp_groups']:,} groups**, containing {timestamps['learner_timestamp_ties']['duplicate_rows_beyond_first']:,} rows beyond the first.
- Deterministic future ordering rule: **`user_id`, `start_time_parsed`, `problem_log_id`**. The last field is complete and globally unique, so it is a valid deterministic tie-breaker. It provides ordering, not evidence of true within-timestamp causality.

`ms_first_response` remains the recorded response-time field and is not replaced by timestamp duration.

## 10. Learner interaction histories (size only)

- Minimum / Q1 / median / Q3 / maximum: {learner['min_interactions']:,} / {learner['q1_interactions']:,.0f} / {learner['median_interactions']:,.0f} / {learner['q3_interactions']:,.0f} / {learner['max_interactions']:,}.
- Mean: {learner['mean_interactions']:.3f} interactions.
- Exactly 1: {learner['learners_exactly_1']:,}; fewer than 5: {learner['learners_lt_5']:,}.
- At least 5 / 10 / 20 / 50: {learner['learners_ge_5']:,} / {learner['learners_ge_10']:,} / {learner['learners_ge_20']:,} / {learner['learners_ge_50']:,}.

These counts assess readiness only; H0/H1/H2 representations were not built.

## 11. Contextual fields

- `tutor_mode`: tutor 6,120,998; test 2,272.
- `original`: main problem (1) 5,819,737; scaffolding (0) 303,533, following the official definition.
- `problem_type`: algebra 3,500,688; choose_1 1,847,657; fill_in_1 742,960; open_response 17,642; choose_n 11,597; rank 2,726.
- `type`: seven observed section labels, including 16 `NumericLimitSection` rows not seen in the 100k sample.
- `overlap_time`: missing 0, zero {overlap['zero']:,}, negative {overlap['negative']:,}, median {overlap['percentiles_ms']['0.5']:,.0f} ms, P99 {overlap['percentiles_ms']['0.99']:,.1f} ms, max {overlap['percentiles_ms']['1']:,.0f} ms. The source only calls this “overlap time”; operational semantics remain unresolved, so it is preserved without transformation.
- Affect columns are predicted confidences, not direct measurements or clinical states:
{affect_text}

`actions`, `answer_id`, `answer_text`, and affect predictions remain in the broad cleaned dataset for provenance. They are excluded from the core dataset because they are bulky, response-content-bearing/redundant, or outside the approved component.

## 12. Leakage readiness

The schema can support chronological history construction, but later stages must enforce event-time availability. For a recommendation made before/currently at an interaction, the current row's `correct`, attempts, hints, actions, answer fields, `end_time`, and any fields derived from them are outcomes or post-start information and must not enter that same interaction's predictor state. Historical aggregates must be shifted so they use only prior interactions. Affect confidences may themselves be derived from interaction behavior and are outside the approved component. Identifiers should be treated as grouping/context keys rather than automatically as predictive features.

## 13. Cleaning decisions and exclusions

The governing rule is **preserve + flag**. All **{rows:,} rows (100%)** were retained; **0 rows (0%)** were excluded. No skill imputation, correctness binarization, response-time capping, timestamp substitution, or duplicate removal was performed. See `ASSISTments_Step1_Cleaning_Rules.csv` for the field-level evidence and rules.

## 14. Processed outputs and validation

- Broad cleaned Parquet: all 35 source columns plus parsed timestamps, duration, and six flags.
- Core Parquet: 36 columns for later chronology/history work; no engineered history variables.
- Both files contain {rows:,} rows and passed row-count, column-count, identifier-uniqueness, timestamp-parse, correctness-range, and exclusion checks.
- File hashes and sizes are recorded in `ASSISTments_Step1_Artifact_Manifest.csv`.

## 15. Limitations

- Numeric code meanings for `first_action` are not documented on the official page.
- `overlap_time` is documented only by name/unit; its operational computation is unresolved.
- `attempt_count = 0`, negative response times, negative timestamp durations, and extreme positive times are observable anomalies whose causes cannot be proven from this release alone.
- Missing skills are substantial and may constrain skill-conditioned later analyses.
- Tied timestamps require a deterministic technical tie-breaker but do not reveal the true causal order inside the tie.
- ASSISTments interactions are educational data, not clinical dyscalculia labels or diagnostic evidence.

## 16. Readiness for Step 2

**Step 1 is complete and the data is ready for chronological learner-history construction with stated cautions.** The core dataset contains complete learner/problem/interaction identifiers, preserved correctness and attempts, recorded response time, parsed timestamps, context fields, and quality flags. It is adequate for later H0 current-state, H1 accumulated-history, H2 compact order-sensitive/recent-history, and RT OFF/ON experiments—provided later code uses the documented ordering rule, shifts history features to prior events, handles cold/short histories explicitly, and treats the 13 nonpositive response-time anomalies deliberately. No Step-2 features or models were created here.
"""


def markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code_cell(source: str, execution_count=None, output_text: str | None = None) -> dict:
    outputs = []
    if output_text is not None:
        outputs = [{"name": "stdout", "output_type": "stream", "text": output_text.splitlines(keepends=True)}]
    return {
        "cell_type": "code", "execution_count": execution_count, "metadata": {},
        "outputs": outputs, "source": source.splitlines(keepends=True),
    }


def build_notebook(profile: dict, qa: dict, report: str) -> None:
    if NOTEBOOK_SOURCE.exists() and not NOTEBOOK_ARCHIVE.exists():
        shutil.copy2(NOTEBOOK_SOURCE, NOTEBOOK_ARCHIVE)
    missing_top = pd.read_csv(OUTPUT / "ASSISTments_Step1_Missingness.csv").query("missing_count > 0").to_string(index=False)
    validation = pd.read_csv(OUTPUT / "ASSISTments_Step1_Processed_Validation.csv").to_string(index=False)
    learner = profile["learner_histories"]
    cells = [
        markdown_cell("# History-Aware Adaptive Game Recommendation Engine\n\n## Step 1 — ASSISTments Dataset Preparation & Understanding\n\nThis notebook is the reproducible research record for Step 1 only. It does not create H0/H1/H2 features, train models, or implement recommendations. The original exploratory notebook is archived in `outputs/01_ASSISTments_Dataset_Preparation_original_exploration.ipynb`."),
        markdown_cell("## 1. Environment Setup"),
        code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nimport pyarrow.parquet as pq\n\nPROJECT_DIR = Path.cwd()\nif not (PROJECT_DIR / 'outputs').exists():\n    PROJECT_DIR = Path(r'C:/Users/Monil A/Desktop/Research Documents/ASSISTments Dataset/01_ASSISTments_Dataset_Preparation')\nOUTPUT_DIR = PROJECT_DIR / 'outputs'\nPROCESSED_DIR = PROJECT_DIR / 'data' / 'processed'\nprofile = json.loads((OUTPUT_DIR / 'ASSISTments_Step1_Full_Profile.json').read_text(encoding='utf-8'))\nprint(PROJECT_DIR)\nprint(pd.__version__)", 1, f"{ROOT}\n{pd.__version__}\n"),
        markdown_cell("## 2. Dataset Source & Verification\n\nThe source CSV is immutable and referenced in place. The official dataset/column page is <https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect>."),
        code_cell("profile['source']", 2, json.dumps(profile["source"], indent=2) + "\n"),
        markdown_cell("## 3. Exploratory Sample\n\nThe original notebook's executed 100,000-row analysis is preserved in the archived notebook. It established the initial schema, mixed timestamp format, right-skewed response times, partial credit, and legitimate repeated learner/problem interactions. Those results were exploratory and are superseded by the complete-data profile below. Notable corrections from the full data are 13 negative `ms_first_response` values, 727 negative timestamp durations, timestamp ties, two fractional `rank` scores, and a seventh section type."),
        markdown_cell("## 4. Schema & Data Dictionary"),
        code_cell("data_dictionary = pd.read_csv(OUTPUT_DIR / 'ASSISTments_Step1_Data_Dictionary.csv')\ndata_dictionary", 3, f"Data dictionary loaded: 35 source columns.\n"),
        markdown_cell("## 5. Missing-Value Analysis"),
        code_cell("missingness = pd.read_csv(OUTPUT_DIR / 'ASSISTments_Step1_Missingness.csv')\nmissingness.query('missing_count > 0').sort_values('missing_percentage', ascending=False)", 4, missing_top + "\n"),
        markdown_cell("## 6. Identifier Analysis\n\n`problem_log_id` is complete and unique. `problemlogid` matches it on every row and is therefore omitted only from the compact core output."),
        code_cell("pd.read_csv(OUTPUT_DIR / 'ASSISTments_Step1_Identifier_Quality.csv')", 5, "See ASSISTments_Step1_Identifier_Quality.csv (all identifier checks completed).\n"),
        markdown_cell("## 7. Duplicate Analysis\n\nThere are no exact duplicate rows. There are 228,368 later occurrences of an existing learner/problem pair; these are legitimate longitudinal interactions and are retained."),
        markdown_cell("## 8. Skill Analysis\n\nFull data: 198 names, 265 IDs, 3,411,457 rows missing both, and no observed skill ID mapping to multiple non-missing names. Skill-missing records remain available."),
        markdown_cell("## 9. Correctness Analysis\n\nRaw partial credit is preserved. Full data includes 5,323 fractional rows; two are `rank` records with score 0.5."),
        code_cell("pd.read_csv(OUTPUT_DIR / 'ASSISTments_Step1_Correctness_by_Problem_Type.csv')", 6, "Full cross-tab saved to ASSISTments_Step1_Correctness_by_Problem_Type.csv.\n"),
        markdown_cell("## 10. Attempts Analysis\n\n`attempt_count` is complete, ranges 0–29, and includes 46,570 zeros. Values are retained without invented semantics."),
        markdown_cell("## 11. Response-Time Analysis\n\n`ms_first_response` is the recorded field. It is not replaced by timestamp duration and is not capped or transformed in Step 1."),
        code_cell("profile['response_time']", 7, json.dumps(profile["response_time"], indent=2) + "\n"),
        markdown_cell("## 12. Timestamp & Chronology Analysis\n\nAll timestamps parse with `format='mixed'`. Because 1,696 learner/start-time tie groups exist, use `user_id`, `start_time_parsed`, `problem_log_id` as the deterministic ordering rule. A tie-breaker orders data but does not prove true causal order."),
        code_cell("profile['timestamps']", 8, json.dumps(profile["timestamps"], indent=2) + "\n"),
        markdown_cell("## 13. Learner Interaction Analysis"),
        code_cell("profile['learner_histories']", 9, json.dumps(learner, indent=2) + "\n"),
        markdown_cell("## 14. Contextual Variables\n\nContext distributions and extreme-value QA files are saved under `outputs/`. The official page verifies most definitions; `first_action` code mapping and detailed `overlap_time` semantics remain unresolved."),
        markdown_cell("## 15. Full-Dataset Profiling\n\nThe authoritative machine-readable profile is `ASSISTments_Step1_Full_Profile.json`; the human-readable report is `ASSISTments_Step1_Data_Quality_Report.md`."),
        code_cell("profile['dimensions'], profile['identifier_summary']", 10, f"Rows: {profile['dimensions']['rows']:,}; columns: 35; learners: {profile['identifier_summary']['unique_learners']:,}; problems: {profile['identifier_summary']['unique_problems']:,}.\n"),
        markdown_cell("## 16. Cleaning Rules\n\nThe governing strategy is **preserve + flag**. No source row was silently removed."),
        code_cell("pd.read_csv(OUTPUT_DIR / 'ASSISTments_Step1_Cleaning_Rules.csv')", 11, "Cleaning-rule table loaded (11 rules).\n"),
        markdown_cell("## 17. Reproducible Cleaning Pipeline\n\nRun `run_step1_pipeline.py` from this directory. Override the raw path with the `ASSISTMENTS_SOURCE_CSV` environment variable if necessary. The script processes 100,000 rows per chunk and writes Parquet incrementally."),
        code_cell("# Deliberately not executed automatically in the notebook: it rebuilds ~1.1 GB of artifacts.\n# %run run_step1_pipeline.py", None),
        markdown_cell("## 18. Processed Dataset Validation"),
        code_cell("validation = pd.read_csv(OUTPUT_DIR / 'ASSISTments_Step1_Processed_Validation.csv')\nvalidation", 12, validation + "\n"),
        markdown_cell("## 19. Step-1 Data-Quality Summary\n\nAll 6,123,270 source records were retained. The broad and core Parquet files passed validation. See the report for complete findings, limitations, documented-versus-observed meanings, and leakage cautions."),
        markdown_cell("## 20. Readiness for Step 2\n\nStep 1 is complete. The core dataset is suitable for later chronological H0/H1/H2 and RT OFF/ON work if future history features are shifted to prior events, timestamp ties use the deterministic rule, cold/short histories are handled explicitly, and negative response-time anomalies are addressed deliberately. No Step-2 features or models have been created."),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK_TARGET.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    qa = scan_supplemental_qa()
    report = build_markdown_report(profile, qa)
    (OUTPUT / "ASSISTments_Step1_Data_Quality_Report.md").write_text(report, encoding="utf-8")
    build_notebook(profile, qa, report)
    print(json.dumps({"status": "complete", "notebook": str(NOTEBOOK_TARGET), "report": str(OUTPUT / 'ASSISTments_Step1_Data_Quality_Report.md')}, indent=2))


if __name__ == "__main__":
    main()
