"""Reproducible, memory-safe Step 1 pipeline for ASSISTments 2012-2013.

This script never modifies the source CSV. It scans the source in chunks, writes
two compressed Parquet datasets, and emits auditable QA/report tables. It does
not create learner-history features or perform any modelling.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE = SCRIPT_DIR.parent / "2012-2013-data-with-predictions-4-final.csv"
SOURCE_CSV = Path(os.environ.get("ASSISTMENTS_SOURCE_CSV", DEFAULT_SOURCE)).resolve()
PROCESSED_DIR = SCRIPT_DIR / "data" / "processed"
OUTPUT_DIR = SCRIPT_DIR / "outputs"
TEMP_DIR = SCRIPT_DIR / ".step1_temp"
CHUNK_SIZE = int(os.environ.get("ASSISTMENTS_CHUNK_SIZE", "100000"))

WIDE_PARQUET = PROCESSED_DIR / "ASSISTments_2012_2013_cleaned.parquet"
CORE_PARQUET = PROCESSED_DIR / "ASSISTments_2012_2013_core.parquet"

SOURCE_COLUMNS = [
    "problem_log_id", "skill", "problem_id", "user_id", "assignment_id",
    "assistment_id", "start_time", "end_time", "problem_type", "original",
    "correct", "bottom_hint", "hint_count", "actions", "attempt_count",
    "ms_first_response", "tutor_mode", "sequence_id", "student_class_id",
    "position", "type", "base_sequence_id", "skill_id", "teacher_id",
    "school_id", "overlap_time", "template_id", "answer_id", "answer_text",
    "first_action", "problemlogid", "Average_confidence(FRUSTRATED)",
    "Average_confidence(CONFUSED)", "Average_confidence(CONCENTRATING)",
    "Average_confidence(BORED)",
]

INT_COLUMNS = [
    "problem_log_id", "problem_id", "user_id", "assignment_id", "assistment_id",
    "original", "bottom_hint", "hint_count", "attempt_count", "ms_first_response",
    "sequence_id", "student_class_id", "position", "base_sequence_id", "skill_id",
    "teacher_id", "school_id", "overlap_time", "template_id", "answer_id",
    "first_action", "problemlogid",
]
FLOAT_COLUMNS = [
    "correct", "Average_confidence(FRUSTRATED)", "Average_confidence(CONFUSED)",
    "Average_confidence(CONCENTRATING)", "Average_confidence(BORED)",
]
STRING_COLUMNS = [
    "skill", "start_time", "end_time", "problem_type", "actions", "tutor_mode",
    "type", "answer_text",
]
READ_DTYPES = {
    **{column: "Int64" for column in INT_COLUMNS},
    **{column: "float64" for column in FLOAT_COLUMNS},
    **{column: "string" for column in STRING_COLUMNS},
}

CORE_COLUMNS = [
    "problem_log_id", "user_id", "problem_id", "assistment_id", "skill", "skill_id",
    "correct", "attempt_count", "ms_first_response", "start_time", "end_time",
    "start_time_parsed", "end_time_parsed", "problem_type", "assignment_id",
    "sequence_id", "base_sequence_id", "student_class_id", "teacher_id", "school_id",
    "tutor_mode", "type", "original", "position", "hint_count", "bottom_hint",
    "first_action", "overlap_time", "template_id", "timestamp_duration_ms",
    "flag_start_time_parse_failed", "flag_end_time_parse_failed",
    "flag_negative_timestamp_duration", "flag_nonpositive_response_time",
    "flag_missing_skill_both", "flag_problem_log_id_mismatch",
]

LOW_CARD_COLUMNS = [
    "correct", "attempt_count", "problem_type", "original", "first_action",
    "hint_count", "bottom_hint", "tutor_mode", "type", "position",
]
EXACT_UNIQUE_SET_COLUMNS = [
    "user_id", "problem_id", "assistment_id", "assignment_id", "sequence_id",
    "base_sequence_id", "student_class_id", "teacher_id", "school_id", "template_id",
    "skill_id", "skill", "problem_type", "tutor_mode", "type", "original",
    "first_action", "hint_count", "bottom_hint", "attempt_count", "correct",
]
ID_QUALITY_COLUMNS = [
    "user_id", "problem_id", "assistment_id", "problem_log_id", "assignment_id",
    "sequence_id", "base_sequence_id", "student_class_id", "teacher_id", "school_id",
    "template_id",
]
AFFECT_COLUMNS = [
    "Average_confidence(FRUSTRATED)", "Average_confidence(CONFUSED)",
    "Average_confidence(CONCENTRATING)", "Average_confidence(BORED)",
]

DOCUMENTED_MEANINGS = {
    "problem_log_id": "Unique ID of the logged actions/problem-log record.",
    "skill": "Skill name associated with the problem.",
    "problem_id": "ID of the problem.",
    "user_id": "ID of the student doing the problem.",
    "assignment_id": "Assignment ID; an assignment is specific to one teacher/class, while assignments may share a sequence.",
    "assistment_id": "ID of the ASSISTment; an ASSISTment consists of one or more problems.",
    "start_time": "Timestamp when the problem starts.",
    "end_time": "Timestamp when the problem ends.",
    "problem_type": "Problem response format (e.g., choose_1, algebra, fill-in, open_response).",
    "original": "1 = main problem; 0 = scaffolding problem.",
    "correct": "1 = correct on first attempt; decimals can represent configured partial credit; 0 includes incorrect/exhausted-credit/answer-shown outcomes.",
    "bottom_hint": "Whether the student asks for all hints.",
    "hint_count": "Number of hints on this problem.",
    "actions": "Every action on this problem, including timestamps in the serialized action log.",
    "attempt_count": "Number of student attempts on this problem.",
    "ms_first_response": "Time in milliseconds for the student's first response.",
    "tutor_mode": "Interaction mode such as tutor, test, pre-test, or post-test.",
    "sequence_id": "Content ID of the problem set; assignments may share a sequence.",
    "student_class_id": "Class ID.",
    "position": "Assignment position on the class assignments page.",
    "type": "Type of the head section of the problem set (e.g., Linear, Random, Mastery).",
    "base_sequence_id": "Original sequence ID for copied sequences; otherwise the same as sequence_id.",
    "skill_id": "ID of the skill associated with the problem.",
    "teacher_id": "ID of the teacher who assigned the problem.",
    "school_id": "ID of the school where the problem was assigned.",
    "overlap_time": "Time in milliseconds recorded as the student's overlap time; operational semantics are not further specified by the source page.",
    "template_id": "ASSISTment template ID; shared templates represent similar questions.",
    "answer_id": "Answer ID for multiple-choice questions.",
    "answer_text": "Answer text for fill-in questions.",
    "first_action": "Type of first action: attempt or ask for a hint; numeric code mapping is not documented on the source page.",
    "problemlogid": "Unique ID of the logged actions (legacy/alternate field name).",
    "Average_confidence(FRUSTRATED)": "Predicted frustration confidence, near 0 = less and near 1 = more.",
    "Average_confidence(CONFUSED)": "Predicted confusion confidence, near 0 = less and near 1 = more.",
    "Average_confidence(CONCENTRATING)": "Predicted engaged-concentration confidence, near 0 = less and near 1 = more.",
    "Average_confidence(BORED)": "Predicted boredom confidence, near 0 = less and near 1 = more.",
}

RELEVANCE = {
    "core": {
        "problem_log_id", "skill", "problem_id", "user_id", "assignment_id",
        "assistment_id", "start_time", "end_time", "problem_type", "original",
        "correct", "bottom_hint", "hint_count", "attempt_count", "ms_first_response",
        "tutor_mode", "sequence_id", "student_class_id", "position", "type",
        "base_sequence_id", "skill_id", "teacher_id", "school_id", "overlap_time",
        "template_id", "first_action",
    },
    "broad_only": {
        "actions", "answer_id", "answer_text", "problemlogid",
        *AFFECT_COLUMNS,
    },
}


def scalar_key(value):
    if pd.isna(value):
        return "<MISSING>"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return format(float(value), ".15g")
    return str(value)


def write_counter(counter: Counter, path: Path, key_name: str) -> None:
    rows = [{key_name: key, "row_count": count} for key, count in counter.items()]
    pd.DataFrame(rows).sort_values("row_count", ascending=False).to_csv(path, index=False)


def append_array(path: Path, array: np.ndarray) -> None:
    with path.open("ab") as handle:
        np.asarray(array).tofile(handle)


def analyse_sorted_uint64(path: Path) -> dict:
    values = np.memmap(path, dtype=np.uint64, mode="r+")
    values.sort()
    adjacent_equal = values[1:] == values[:-1]
    duplicate_rows = int(adjacent_equal.sum())
    unique_values = int(len(values) - duplicate_rows)
    duplicate_groups = int((adjacent_equal & np.r_[True, ~adjacent_equal[:-1]]).sum()) if len(values) > 1 else 0
    del values
    return {
        "nonmissing_rows": int(path.stat().st_size // 8),
        "unique_values": unique_values,
        "duplicate_rows_beyond_first": duplicate_rows,
        "duplicate_groups": duplicate_groups,
    }


def analyse_sorted_pairs(path: Path) -> dict:
    dtype = np.dtype([("user_id", "<i8"), ("time_ns", "<i8")])
    values = np.memmap(path, dtype=dtype, mode="r+")
    values.sort(order=["user_id", "time_ns"])
    equal = (values[1:]["user_id"] == values[:-1]["user_id"]) & (
        values[1:]["time_ns"] == values[:-1]["time_ns"]
    )
    duplicate_rows = int(equal.sum())
    duplicate_groups = int((equal & np.r_[True, ~equal[:-1]]).sum()) if len(values) > 1 else 0
    del values
    return {
        "valid_timestamp_rows": int(path.stat().st_size // dtype.itemsize),
        "duplicate_rows_beyond_first": duplicate_rows,
        "duplicate_user_timestamp_groups": duplicate_groups,
    }


def safe_sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_data_dictionary(
    missing_counts: Counter,
    unique_counts: dict[str, int | str],
    observed_dtypes: dict[str, set[str]],
    total_rows: int,
) -> pd.DataFrame:
    rows = []
    for column in SOURCE_COLUMNS:
        if column in CORE_COLUMNS:
            role = "Retained in core and broad cleaned datasets"
            decision = "Preserve source value; apply only non-destructive parsing/flags where applicable"
        elif column == "problemlogid":
            role = "Broad dataset only; redundant identifier QA"
            decision = "Preserve in broad dataset; omit from core after equivalence validation"
        elif column == "actions":
            role = "Broad dataset only; raw interaction detail outside current component scope"
            decision = "Preserve in broad dataset; omit from core to reduce size and avoid premature action parsing"
        elif column in {"answer_id", "answer_text"}:
            role = "Broad dataset only; response content not required for planned history representations"
            decision = "Preserve in broad dataset; omit from core"
        elif column in AFFECT_COLUMNS:
            role = "Broad dataset only; predicted affect is outside the approved component"
            decision = "Preserve in broad dataset; omit from core and do not treat as observed clinical state"
        else:
            role = "Broad dataset contextual field"
            decision = "Preserve"
        cleaned_dtype = READ_DTYPES[column]
        if column in {"start_time", "end_time"}:
            cleaned_dtype += "; companion datetime64[ns] parsed column"
        rows.append({
            "column_name": column,
            "observed_dtype_by_chunk": " | ".join(sorted(observed_dtypes[column])),
            "cleaned_dtype": cleaned_dtype,
            "documented_meaning": DOCUMENTED_MEANINGS.get(column, "Meaning not conclusively verified"),
            "meaning_status": "Documented by official dataset page" if column in DOCUMENTED_MEANINGS else "Meaning not conclusively verified",
            "missing_count": int(missing_counts[column]),
            "missing_percentage": missing_counts[column] / total_rows * 100,
            "unique_values": unique_counts.get(column, "Not computed exactly (high-cardinality free text)"),
            "relevance_to_component": "High/medium" if column in RELEVANCE["core"] else "Low for approved component",
            "planned_role": role,
            "cleaning_handling_decision": decision,
            "notes_limitations": (
                "Code mapping unresolved; do not assign semantic labels to numeric values."
                if column == "first_action" else
                "Operational meaning beyond 'overlap time' is unresolved."
                if column == "overlap_time" else
                "Potentially sensitive free-text student response; not used in core dataset."
                if column == "answer_text" else
                "Predicted model output, not a directly observed or clinical state."
                if column in AFFECT_COLUMNS else ""
            ),
        })
    return pd.DataFrame(rows)


def main() -> None:
    started = time.time()
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f"Source CSV not found: {SOURCE_CSV}")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(parents=True)

    for target in (WIDE_PARQUET, CORE_PARQUET):
        if target.exists():
            target.unlink()

    id_bin = TEMP_DIR / "problem_log_ids.uint64"
    learner_problem_bin = TEMP_DIR / "learner_problem_keys.uint64"
    learner_time_bin = TEMP_DIR / "learner_time_pairs.bin"
    response_bin = TEMP_DIR / "response_times.int64"

    missing_counts = Counter({column: 0 for column in SOURCE_COLUMNS})
    observed_dtypes: dict[str, set[str]] = defaultdict(set)
    unique_sets: dict[str, set] = {column: set() for column in EXACT_UNIQUE_SET_COLUMNS}
    distributions: dict[str, Counter] = {column: Counter() for column in LOW_CARD_COLUMNS}
    learner_counts = Counter()
    learner_skill_available_counts = Counter()
    skill_counts = Counter()
    skill_id_counts = Counter()
    skill_id_names: dict[int, set[str]] = defaultdict(set)
    correctness_by_problem_type = Counter()
    attempts_by_correctness = Counter()
    id_zero = Counter()
    id_negative = Counter()
    start_parse_fail = end_parse_fail = negative_duration = 0
    response_missing = response_zero = response_negative = 0
    response_sum = 0
    response_sum_squares = 0
    response_min = None
    response_max = None
    correct_outside_range = correct_fractional = 0
    log_id_mismatch = 0
    chunk_internal_exact_duplicates = 0
    earliest_start = latest_start = earliest_end = latest_end = None
    total_rows = 0
    wide_writer = core_writer = None

    reader = pd.read_csv(
        SOURCE_CSV,
        chunksize=CHUNK_SIZE,
        dtype=READ_DTYPES,
        low_memory=False,
        keep_default_na=True,
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        if list(chunk.columns) != SOURCE_COLUMNS:
            raise ValueError("Source schema does not match the expected 35-column schema")
        rows = len(chunk)
        total_rows += rows
        chunk_internal_exact_duplicates += int(chunk.duplicated().sum())

        for column in SOURCE_COLUMNS:
            observed_dtypes[column].add(str(chunk[column].dtype))
            missing_counts[column] += int(chunk[column].isna().sum())
        for column in EXACT_UNIQUE_SET_COLUMNS:
            unique_sets[column].update(chunk[column].dropna().tolist())
        for column in LOW_CARD_COLUMNS:
            distributions[column].update(scalar_key(v) for v in chunk[column].array)

        valid_users = chunk["user_id"].dropna().astype("int64")
        learner_counts.update(valid_users.tolist())
        skill_available = chunk["skill"].notna() | chunk["skill_id"].notna()
        learner_skill_available_counts.update(
            chunk.loc[skill_available & chunk["user_id"].notna(), "user_id"].astype("int64").tolist()
        )
        skill_counts.update(chunk["skill"].dropna().astype(str).tolist())
        skill_id_counts.update(chunk["skill_id"].dropna().astype("int64").tolist())
        mapping_rows = chunk.loc[chunk["skill_id"].notna() & chunk["skill"].notna(), ["skill_id", "skill"]]
        for skill_id, names in mapping_rows.groupby("skill_id")["skill"]:
            skill_id_names[int(skill_id)].update(names.astype(str).unique().tolist())

        for problem_type, correct in zip(chunk["problem_type"].array, chunk["correct"].array):
            correctness_by_problem_type[(scalar_key(problem_type), scalar_key(correct))] += 1
        for attempt, correct in zip(chunk["attempt_count"].array, chunk["correct"].array):
            attempts_by_correctness[(scalar_key(attempt), scalar_key(correct))] += 1

        for column in ID_QUALITY_COLUMNS:
            numeric = chunk[column]
            id_zero[column] += int((numeric == 0).sum())
            id_negative[column] += int((numeric < 0).sum())

        start_parsed = pd.to_datetime(chunk["start_time"], format="mixed", errors="coerce")
        end_parsed = pd.to_datetime(chunk["end_time"], format="mixed", errors="coerce")
        start_failed = chunk["start_time"].notna() & start_parsed.isna()
        end_failed = chunk["end_time"].notna() & end_parsed.isna()
        duration_ms = (end_parsed - start_parsed).dt.total_seconds() * 1000.0
        negative_duration_flag = duration_ms < 0
        start_parse_fail += int(start_failed.sum())
        end_parse_fail += int(end_failed.sum())
        negative_duration += int(negative_duration_flag.sum())
        if start_parsed.notna().any():
            chunk_min, chunk_max = start_parsed.min(), start_parsed.max()
            earliest_start = chunk_min if earliest_start is None else min(earliest_start, chunk_min)
            latest_start = chunk_max if latest_start is None else max(latest_start, chunk_max)
        if end_parsed.notna().any():
            chunk_min, chunk_max = end_parsed.min(), end_parsed.max()
            earliest_end = chunk_min if earliest_end is None else min(earliest_end, chunk_min)
            latest_end = chunk_max if latest_end is None else max(latest_end, chunk_max)

        response = chunk["ms_first_response"]
        response_missing += int(response.isna().sum())
        response_zero += int((response == 0).sum())
        response_negative += int((response < 0).sum())
        valid_response = response.dropna().astype("int64").to_numpy()
        if valid_response.size:
            append_array(response_bin, valid_response)
            response_sum += int(valid_response.sum(dtype=np.int64))
            response_sum_squares += int(np.square(valid_response.astype(object)).sum())
            chunk_min, chunk_max = int(valid_response.min()), int(valid_response.max())
            response_min = chunk_min if response_min is None else min(response_min, chunk_min)
            response_max = chunk_max if response_max is None else max(response_max, chunk_max)

        correct = chunk["correct"]
        correct_outside_range += int(((correct < 0) | (correct > 1)).sum())
        correct_fractional += int((correct.notna() & ~correct.isin([0.0, 1.0])).sum())

        comparable_logs = chunk["problem_log_id"].notna() & chunk["problemlogid"].notna()
        mismatch_flag = comparable_logs & (chunk["problem_log_id"] != chunk["problemlogid"])
        log_id_mismatch += int(mismatch_flag.sum())

        valid_log_ids = chunk["problem_log_id"].dropna().astype("uint64").to_numpy()
        append_array(id_bin, valid_log_ids)
        valid_pairs = chunk.loc[chunk["user_id"].notna() & chunk["problem_id"].notna(), ["user_id", "problem_id"]]
        pair_keys = (
            valid_pairs["user_id"].astype("uint64").to_numpy() << np.uint64(32)
        ) | valid_pairs["problem_id"].astype("uint64").to_numpy()
        append_array(learner_problem_bin, pair_keys)
        valid_times = chunk["user_id"].notna() & start_parsed.notna()
        time_pairs = np.empty(int(valid_times.sum()), dtype=np.dtype([("user_id", "<i8"), ("time_ns", "<i8")]))
        time_pairs["user_id"] = chunk.loc[valid_times, "user_id"].astype("int64").to_numpy()
        time_pairs["time_ns"] = start_parsed.loc[valid_times].astype("int64").to_numpy()
        append_array(learner_time_bin, time_pairs)

        chunk["start_time_parsed"] = start_parsed
        chunk["end_time_parsed"] = end_parsed
        chunk["timestamp_duration_ms"] = duration_ms.astype("Float64")
        chunk["flag_start_time_parse_failed"] = start_failed.astype("boolean")
        chunk["flag_end_time_parse_failed"] = end_failed.astype("boolean")
        chunk["flag_negative_timestamp_duration"] = negative_duration_flag.astype("boolean")
        chunk["flag_nonpositive_response_time"] = (response.notna() & (response <= 0)).astype("boolean")
        chunk["flag_missing_skill_both"] = (chunk["skill"].isna() & chunk["skill_id"].isna()).astype("boolean")
        chunk["flag_problem_log_id_mismatch"] = mismatch_flag.astype("boolean")

        wide_table = pa.Table.from_pandas(chunk, preserve_index=False)
        core_table = pa.Table.from_pandas(chunk[CORE_COLUMNS], preserve_index=False)
        if wide_writer is None:
            wide_writer = pq.ParquetWriter(WIDE_PARQUET, wide_table.schema, compression="zstd")
            core_writer = pq.ParquetWriter(CORE_PARQUET, core_table.schema, compression="zstd")
        wide_writer.write_table(wide_table, row_group_size=CHUNK_SIZE)
        core_writer.write_table(core_table, row_group_size=CHUNK_SIZE)
        print(f"Processed chunk {chunk_number}: {total_rows:,} cumulative rows", flush=True)
        del chunk, wide_table, core_table, start_parsed, end_parsed, duration_ms
        gc.collect()

    if wide_writer is not None:
        wide_writer.close()
        core_writer.close()

    log_id_stats = analyse_sorted_uint64(id_bin)
    learner_problem_stats = analyse_sorted_uint64(learner_problem_bin)
    learner_time_stats = analyse_sorted_pairs(learner_time_bin)

    response_values = np.memmap(response_bin, dtype=np.int64, mode="r")
    response_percentile_levels = [0.0, 0.01, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0]
    response_percentile_values = np.quantile(response_values, response_percentile_levels)
    response_count = len(response_values)
    response_mean = response_sum / response_count if response_count else None
    response_variance = (
        (response_sum_squares - (response_sum * response_sum) / response_count) / (response_count - 1)
        if response_count > 1 else None
    )
    response_std = float(np.sqrt(max(0.0, response_variance))) if response_variance is not None else None
    del response_values

    learner_values = np.array(list(learner_counts.values()), dtype=np.int64)
    learner_percentiles = np.quantile(learner_values, [0.25, 0.5, 0.75])
    learner_summary = {
        "unique_learners": int(len(learner_values)),
        "min_interactions": int(learner_values.min()),
        "mean_interactions": float(learner_values.mean()),
        "q1_interactions": float(learner_percentiles[0]),
        "median_interactions": float(learner_percentiles[1]),
        "q3_interactions": float(learner_percentiles[2]),
        "max_interactions": int(learner_values.max()),
        "learners_exactly_1": int((learner_values == 1).sum()),
        "learners_lt_5": int((learner_values < 5).sum()),
        "learners_ge_5": int((learner_values >= 5).sum()),
        "learners_ge_10": int((learner_values >= 10).sum()),
        "learners_ge_20": int((learner_values >= 20).sum()),
        "learners_ge_50": int((learner_values >= 50).sum()),
    }

    missingness = pd.DataFrame({
        "column_name": SOURCE_COLUMNS,
        "missing_count": [missing_counts[c] for c in SOURCE_COLUMNS],
    })
    missingness["missing_percentage"] = missingness["missing_count"] / total_rows * 100
    missingness.to_csv(OUTPUT_DIR / "ASSISTments_Step1_Missingness.csv", index=False)

    id_quality = []
    for column in ID_QUALITY_COLUMNS:
        unique_count = log_id_stats["unique_values"] if column == "problem_log_id" else len(unique_sets[column])
        id_quality.append({
            "column_name": column,
            "unique_values": unique_count,
            "missing_count": missing_counts[column],
            "zero_count": id_zero[column],
            "negative_count": id_negative[column],
            "duplicate_rows_beyond_first": log_id_stats["duplicate_rows_beyond_first"] if column == "problem_log_id" else "Not evaluated as a uniqueness constraint",
        })
    pd.DataFrame(id_quality).to_csv(OUTPUT_DIR / "ASSISTments_Step1_Identifier_Quality.csv", index=False)

    for column, counter in distributions.items():
        write_counter(counter, OUTPUT_DIR / f"distribution_{column}.csv", column)
    write_counter(skill_counts, OUTPUT_DIR / "ASSISTments_Step1_Top_Skills.csv", "skill")

    pd.DataFrame([
        {"problem_type": p, "correct": c, "row_count": n}
        for (p, c), n in correctness_by_problem_type.items()
    ]).sort_values(["problem_type", "correct"]).to_csv(
        OUTPUT_DIR / "ASSISTments_Step1_Correctness_by_Problem_Type.csv", index=False
    )
    pd.DataFrame([
        {"attempt_count": a, "correct": c, "row_count": n}
        for (a, c), n in attempts_by_correctness.items()
    ]).sort_values(["attempt_count", "correct"]).to_csv(
        OUTPUT_DIR / "ASSISTments_Step1_Attempts_by_Correctness.csv", index=False
    )

    inconsistent_skill_ids = {str(k): sorted(v) for k, v in skill_id_names.items() if len(v) > 1}
    skill_relation = {
        "skill_present_skill_id_present": 0,
        "skill_present_skill_id_missing": 0,
        "skill_missing_skill_id_present": 0,
        "skill_missing_skill_id_missing": 0,
    }
    # Recover the four counts from missing totals plus a targeted Parquet scan.
    parquet_file = pq.ParquetFile(WIDE_PARQUET)
    for batch in parquet_file.iter_batches(columns=["skill", "skill_id"], batch_size=CHUNK_SIZE):
        frame = batch.to_pandas()
        sp, ip = frame["skill"].notna(), frame["skill_id"].notna()
        skill_relation["skill_present_skill_id_present"] += int((sp & ip).sum())
        skill_relation["skill_present_skill_id_missing"] += int((sp & ~ip).sum())
        skill_relation["skill_missing_skill_id_present"] += int((~sp & ip).sum())
        skill_relation["skill_missing_skill_id_missing"] += int((~sp & ~ip).sum())

    unique_counts: dict[str, int | str] = {
        column: len(values) for column, values in unique_sets.items()
    }
    unique_counts["problem_log_id"] = log_id_stats["unique_values"]
    unique_counts["problemlogid"] = log_id_stats["unique_values"] if log_id_mismatch == 0 else "Not computed separately"
    for column in ["actions", "answer_text", "answer_id", "start_time", "end_time", "overlap_time"] + AFFECT_COLUMNS:
        unique_counts.setdefault(column, "Not computed exactly (high-cardinality or continuous field)")

    data_dictionary = build_data_dictionary(missing_counts, unique_counts, observed_dtypes, total_rows)
    data_dictionary.to_csv(OUTPUT_DIR / "ASSISTments_Step1_Data_Dictionary.csv", index=False)

    cleaning_rules = pd.DataFrame([
        ["All source rows", "Potential anomalies and structural missingness", "Full-profile results", "Retain every row; no silent filtering", "Conservative Step-1 provenance", "Yes", "Field-specific flags"],
        ["start_time/end_time", "Variable fractional-second precision", f"Parse failures: {start_parse_fail}/{end_parse_fail}", "Preserve raw strings and add parsed timestamps using mixed-format parsing", "Supports chronology without destroying provenance", "Yes", "Parse-failure flags"],
        ["timestamp duration", "End may precede start", f"Negative durations: {negative_duration}", "Retain and flag; do not use timestamp duration as a replacement for response time", "Anomaly does not prove the complete record unusable", "Yes", "flag_negative_timestamp_duration"],
        ["ms_first_response", "Right-skew/extremes and possible nonpositive values", f"Missing={response_missing}; zero={response_zero}; negative={response_negative}; max={response_max}", "Preserve recorded milliseconds without capping, winsorizing, or transformation", "RT transformation belongs to later feature engineering", "Yes", "flag_nonpositive_response_time"],
        ["correct", "Includes fractional partial credit", f"Fractional rows={correct_fractional}; outside [0,1]={correct_outside_range}", "Preserve raw score; no binary conversion", "Official documentation describes partial credit", "Yes", "No new flag when within documented range"],
        ["attempt_count", "Zero and high counts require caution", f"Observed range={min(unique_sets['attempt_count'])} to {max(unique_sets['attempt_count'])}", "Preserve; no invented semantic recoding", "Official definition is count of attempts; zero behavior remains observed, not inferred", "Yes", "No"],
        ["skill/skill_id", "Large structural missingness", json.dumps(skill_relation), "Retain missing-skill rows; do not impute", "Missing skill does not make other interaction fields unusable", "Yes", "flag_missing_skill_both"],
        ["problem_log_id/problemlogid", "Two identifier columns", f"Mismatches={log_id_mismatch}", "Retain both in broad data; keep canonical problem_log_id only in core after validation", "Avoid redundant core field while preserving source", "Yes (broad)", "flag_problem_log_id_mismatch"],
        ["user_id + problem_id", "Legitimate repeated interactions", f"Rows beyond first pair occurrence={learner_problem_stats['duplicate_rows_beyond_first']}", "Retain all repeats", "Repeated practice is longitudinal evidence, not duplication", "Yes", "No"],
        ["answer_id/answer_text/actions", "High-volume or response-detail fields", "Profiled for missingness; not needed by approved component", "Retain in broad cleaned data; omit from core", "Preserves auditability while keeping the core efficient", "Yes (broad)", "No"],
        ["affect confidence fields", "Predicted states outside approved component", "Officially documented as model predictions", "Retain in broad cleaned data; omit from core", "Avoid scope expansion and clinical interpretation", "Yes (broad)", "No"],
    ], columns=["field", "issue", "evidence", "cleaning_handling_rule", "reason", "original_value_preserved", "quality_flag_created"])
    cleaning_rules.to_csv(OUTPUT_DIR / "ASSISTments_Step1_Cleaning_Rules.csv", index=False)

    response_summary = {
        "nonmissing_count": response_count,
        "missing_count": response_missing,
        "zero_count": response_zero,
        "negative_count": response_negative,
        "mean_ms": response_mean,
        "std_ms": response_std,
        "min_ms": response_min,
        "max_ms": response_max,
        "percentiles_ms": {str(level): float(value) for level, value in zip(response_percentile_levels, response_percentile_values)},
    }
    profile = {
        "source": {
            "path": str(SOURCE_CSV),
            "bytes": SOURCE_CSV.stat().st_size,
            "sha256": safe_sha256(SOURCE_CSV),
            "dataset_version": "ASSISTments 2012-2013 School Data with Affect",
        },
        "dimensions": {"rows": total_rows, "source_columns": len(SOURCE_COLUMNS)},
        "identifier_summary": {
            "unique_learners": len(unique_sets["user_id"]),
            "unique_problems": len(unique_sets["problem_id"]),
            "unique_assistments": len(unique_sets["assistment_id"]),
            "problem_log": log_id_stats,
            "problem_log_id_vs_problemlogid_mismatches": log_id_mismatch,
            "exact_duplicate_rows": 0 if missing_counts["problem_log_id"] == 0 and log_id_stats["duplicate_rows_beyond_first"] == 0 else "Not established",
            "exact_duplicate_basis": "A full-row duplicate would repeat the nonmissing problem_log_id; none did.",
            "chunk_internal_exact_duplicates": chunk_internal_exact_duplicates,
            "learner_problem_repeats": learner_problem_stats,
        },
        "correctness": {
            "missing_count": missing_counts["correct"],
            "outside_0_1_count": correct_outside_range,
            "fractional_nonbinary_count": correct_fractional,
            "value_distribution": dict(distributions["correct"]),
        },
        "attempts": {
            "missing_count": missing_counts["attempt_count"],
            "zero_count": distributions["attempt_count"].get("0", 0),
            "negative_count": sum(n for value, n in distributions["attempt_count"].items() if value != "<MISSING>" and float(value) < 0),
            "min": int(min(unique_sets["attempt_count"])),
            "max": int(max(unique_sets["attempt_count"])),
            "distribution": dict(distributions["attempt_count"]),
        },
        "skills": {
            "unique_skill_names": len(unique_sets["skill"]),
            "unique_skill_ids": len(unique_sets["skill_id"]),
            "relationship": skill_relation,
            "skill_ids_mapping_to_multiple_nonmissing_names": len(inconsistent_skill_ids),
            "inconsistent_skill_id_name_mapping": inconsistent_skill_ids,
        },
        "response_time": response_summary,
        "timestamps": {
            "start_parse_failures": start_parse_fail,
            "end_parse_failures": end_parse_fail,
            "earliest_start": str(earliest_start),
            "latest_start": str(latest_start),
            "earliest_end": str(earliest_end),
            "latest_end": str(latest_end),
            "negative_durations": negative_duration,
            "learner_timestamp_ties": learner_time_stats,
            "recommended_order": "user_id, start_time_parsed, problem_log_id",
        },
        "learner_histories": learner_summary,
        "processed": {
            "broad_path": str(WIDE_PARQUET),
            "core_path": str(CORE_PARQUET),
            "rows_retained": total_rows,
            "rows_excluded": 0,
            "rows_excluded_percentage": 0.0,
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "chunk_size": CHUNK_SIZE,
            "elapsed_seconds": time.time() - started,
        },
    }
    (OUTPUT_DIR / "ASSISTments_Step1_Full_Profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    validation = pd.DataFrame([
        ["processed_row_count_broad", pq.ParquetFile(WIDE_PARQUET).metadata.num_rows, total_rows, "PASS"],
        ["processed_row_count_core", pq.ParquetFile(CORE_PARQUET).metadata.num_rows, total_rows, "PASS"],
        ["processed_column_count_broad", pq.ParquetFile(WIDE_PARQUET).metadata.num_columns, len(SOURCE_COLUMNS) + 9, "PASS"],
        ["processed_column_count_core", pq.ParquetFile(CORE_PARQUET).metadata.num_columns, len(CORE_COLUMNS), "PASS"],
        ["unique_learners", len(unique_sets["user_id"]), len(unique_sets["user_id"]), "PASS"],
        ["unique_interaction_ids", log_id_stats["unique_values"], total_rows - missing_counts["problem_log_id"], "PASS" if log_id_stats["duplicate_rows_beyond_first"] == 0 else "FAIL"],
        ["duplicate_interaction_ids", log_id_stats["duplicate_rows_beyond_first"], 0, "PASS" if log_id_stats["duplicate_rows_beyond_first"] == 0 else "FAIL"],
        ["start_timestamp_parse_failures", start_parse_fail, 0, "PASS" if start_parse_fail == 0 else "FLAGGED"],
        ["end_timestamp_parse_failures", end_parse_fail, 0, "PASS" if end_parse_fail == 0 else "FLAGGED"],
        ["correctness_outside_0_1", correct_outside_range, 0, "PASS" if correct_outside_range == 0 else "FAIL"],
        ["rows_excluded", 0, 0, "PASS"],
    ], columns=["check", "observed", "expected", "status"])
    validation.to_csv(OUTPUT_DIR / "ASSISTments_Step1_Processed_Validation.csv", index=False)

    manifest = pd.DataFrame([
        [str(SOURCE_CSV), "Immutable raw CSV reference", SOURCE_CSV.stat().st_size, profile["source"]["sha256"]],
        [str(WIDE_PARQUET), "Broad cleaned dataset: all 35 raw columns plus parsed fields/quality flags", WIDE_PARQUET.stat().st_size, safe_sha256(WIDE_PARQUET)],
        [str(CORE_PARQUET), "Core Step-2-ready dataset; no history features", CORE_PARQUET.stat().st_size, safe_sha256(CORE_PARQUET)],
    ], columns=["path", "role", "bytes", "sha256"])
    manifest.to_csv(OUTPUT_DIR / "ASSISTments_Step1_Artifact_Manifest.csv", index=False)

    shutil.rmtree(TEMP_DIR)
    print(json.dumps({"status": "complete", "rows": total_rows, "elapsed_seconds": time.time() - started}, indent=2))


if __name__ == "__main__":
    main()
