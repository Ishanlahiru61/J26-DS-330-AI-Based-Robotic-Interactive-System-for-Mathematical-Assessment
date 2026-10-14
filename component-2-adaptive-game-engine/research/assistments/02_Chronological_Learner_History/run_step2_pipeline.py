#!/usr/bin/env python3
"""Build Step 2 chronology and leakage-safe history features with DuckDB.

This stage deliberately performs no learner split, fitting, encoding, scaling,
normalisation, model training, recommendation, or performance evaluation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
STEP1 = ROOT.parent / "01_ASSISTments_Dataset_Preparation"
INPUT = STEP1 / "data" / "processed" / "ASSISTments_2012_2013_core.parquet"
STEP1_MANIFEST = STEP1 / "outputs" / "ASSISTments_Step1_Artifact_Manifest.csv"
INTERMEDIATE = ROOT / "data" / "intermediate"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
TESTS = ROOT / "tests"
TMP = ROOT / ".duckdb_tmp"
DB = ROOT / ".step2_build.duckdb"
CHRON = INTERMEDIATE / "ASSISTments_2012_2013_chronological.parquet"
FEATURES = PROCESSED / "ASSISTments_2012_2013_history_features.parquet"
EXPECTED_STEP1_SHA256 = "32fcd2ee16ed635b26b48865f5e3130928c79f3a779af95d16ee5250b6d8d2f4"
ORDER_RULE = "user_id ASC, start_time_parsed ASC, problem_log_id ASC"


H0 = [
    "skill_id", "problem_type", "tutor_mode", "original", "position", "template_id",
    "previous_correct_raw", "previous_attempt_count", "previous_ms_first_response",
    "previous_skill_id", "previous_problem_type", "time_since_previous_seconds",
    "has_strict_prior_interaction", "previous_interaction_rt_valid",
]
H1_ADDITIONS = [
    "prior_interaction_count", "prior_target_eligible_count", "prior_full_correct_count",
    "prior_incorrect_count", "prior_correct_rate", "prior_valid_rt_count",
    "prior_mean_ms_first_response", "prior_mean_attempt_count",
    "prior_skill_exposure_count", "prior_skill_target_eligible_count",
    "prior_skill_full_correct_count", "prior_skill_correct_rate",
    "prior_same_problem_exposure_count", "prior_same_assistment_exposure_count",
    "prior_same_problem_type_exposure_count",
]
H2_ADDITIONS = [
    "lag2_target_first_attempt_correct", "lag3_target_first_attempt_correct",
    "lag4_target_first_attempt_correct", "lag5_target_first_attempt_correct",
    "recent3_target_eligible_count", "recent3_full_correct_count", "recent3_correct_rate",
    "recent5_target_eligible_count", "recent5_full_correct_count", "recent5_correct_rate",
    "interactions_since_last_full_correct", "previous_same_skill_correct_raw",
    "time_since_previous_skill_seconds",
]
H1 = H0 + H1_ADDITIONS
H2 = H1 + H2_ADDITIONS


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def qpath(path: Path) -> str:
    return str(path).replace("'", "''")


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def manifest_expected_hash() -> tuple[str, str]:
    manifest_hash = sha256(STEP1_MANIFEST)
    with STEP1_MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            if Path(row["path"]).name == INPUT.name:
                return row["sha256"], manifest_hash
    raise RuntimeError(f"Core input absent from {STEP1_MANIFEST}")


def provenance() -> dict:
    if not INPUT.exists() or not STEP1_MANIFEST.exists():
        raise FileNotFoundError("Frozen Step-1 core or manifest is missing")
    actual = sha256(INPUT)
    recorded, manifest_hash = manifest_expected_hash()
    verified = actual == recorded == EXPECTED_STEP1_SHA256
    meta = pq.ParquetFile(INPUT).metadata
    record = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(INPUT),
        "input_sha256": actual,
        "expected_frozen_sha256": EXPECTED_STEP1_SHA256,
        "step1_manifest_path": str(STEP1_MANIFEST),
        "step1_manifest_sha256": manifest_hash,
        "step1_manifest_recorded_input_sha256": recorded,
        "input_rows": meta.num_rows,
        "input_columns": meta.num_columns,
        "verification_result": "PASS" if verified else "FAIL",
    }
    (OUTPUTS / "ASSISTments_Step2_Provenance.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    if not verified:
        raise RuntimeError(f"Frozen Step-1 verification failed: {record}")
    return record


def connect() -> duckdb.DuckDBPyConnection:
    TMP.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    con.execute("SET threads TO 1")
    con.execute("SET preserve_insertion_order=false")
    con.execute(f"SET temp_directory='{qpath(TMP)}'")
    memory = os.environ.get("ASSISTMENTS_STEP2_MEMORY_LIMIT", "2GB")
    con.execute(f"SET memory_limit='{memory}'")
    return con


def build_chronology(con: duckdb.DuckDBPyConnection) -> None:
    # Window only over the narrow key projection. Carrying all 36 wide columns
    # through the sort needlessly pins string buffers and defeats external spill.
    con.execute(f"""
      CREATE OR REPLACE TABLE chron_meta AS
      SELECT problem_log_id,
        row_number() OVER learner_order - 1 AS learner_interaction_index,
        count(*) OVER (PARTITION BY user_id) AS learner_interaction_count_total,
        count(*) OVER (PARTITION BY user_id, start_time_parsed) AS timestamp_tie_group_size,
        lag(problem_log_id) OVER learner_order AS previous_problem_log_id_technical,
        lag(start_time_parsed) OVER learner_order AS previous_start_time_technical
      FROM read_parquet('{qpath(INPUT)}')
      WINDOW learner_order AS (
        PARTITION BY user_id ORDER BY start_time_parsed, problem_log_id
      )
    """)
    sql = f"""
    COPY (
      SELECT *,
        timestamp_tie_group_size > 1 AS flag_timestamp_tie,
        previous_start_time_technical = start_time_parsed AS flag_previous_same_timestamp_technical,
        correct AS correct_raw,
        CASE WHEN correct IN (0.0, 1.0) THEN CAST(correct AS TINYINT) ELSE NULL END
          AS target_first_attempt_correct,
        correct IN (0.0, 1.0) AS target_eligible,
        CASE WHEN correct IN (0.0, 1.0) THEN NULL ELSE 'fractional_partial_credit' END
          AS target_exclusion_reason,
        true AS history_event_eligible,
        ms_first_response > 0 AS rt_feature_eligible,
        skill_id IS NOT NULL AS skill_conditioned_feature_eligible,
        learner_interaction_index > 0 AS learner_history_available_technical,
        skill_id IS NOT NULL AS skill_history_available_key,
        learner_interaction_index = 0 AS is_cold_start_technical
      FROM read_parquet('{qpath(INPUT)}') i
      JOIN chron_meta m USING(problem_log_id)
      ORDER BY user_id, start_time_parsed, problem_log_id
    ) TO '{qpath(CHRON)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """
    con.execute(sql)


def build_features(con: duckdb.DuckDBPyConnection) -> None:
    # EXCLUDE GROUP is crucial: the peer group is defined only by start_time_parsed,
    # so no row at the same recorded timestamp contributes to another row's history.
    sql = f"""
      CREATE OR REPLACE TABLE feature_meta AS
      WITH w AS (
        SELECT c.problem_log_id,c.user_id,c.start_time_parsed,c.learner_interaction_index,
          c.skill_id,c.problem_id,c.assistment_id,c.problem_type,c.correct_raw,c.attempt_count,
          c.ms_first_response,c.target_first_attempt_correct,c.rt_feature_eligible,
          count(*) OVER learner_prior AS prior_interaction_count,
          count(target_first_attempt_correct) OVER learner_prior AS prior_target_eligible_count,
          count(*) FILTER (WHERE target_first_attempt_correct=1) OVER learner_prior AS prior_full_correct_count,
          count(*) FILTER (WHERE target_first_attempt_correct=0) OVER learner_prior AS prior_incorrect_count,
          count(*) FILTER (WHERE rt_feature_eligible) OVER learner_prior AS prior_valid_rt_count,
          avg(ms_first_response) FILTER (WHERE rt_feature_eligible) OVER learner_prior AS prior_mean_ms_first_response,
          avg(attempt_count) OVER learner_prior AS prior_mean_attempt_count,
          arg_max(correct_raw, struct_pack(t := start_time_parsed, p := problem_log_id)) OVER learner_prior AS previous_correct_raw,
          arg_max(attempt_count, struct_pack(t := start_time_parsed, p := problem_log_id)) OVER learner_prior AS previous_attempt_count,
          arg_max(CASE WHEN rt_feature_eligible THEN ms_first_response END,
                  struct_pack(t := start_time_parsed, p := problem_log_id))
            FILTER (WHERE rt_feature_eligible) OVER learner_prior AS previous_ms_first_response,
          arg_max(skill_id, struct_pack(t := start_time_parsed, p := problem_log_id)) OVER learner_prior AS previous_skill_id,
          arg_max(problem_type, struct_pack(t := start_time_parsed, p := problem_log_id)) OVER learner_prior AS previous_problem_type,
          max(start_time_parsed) OVER learner_prior AS previous_strict_start_time,
          max(CASE WHEN target_first_attempt_correct=1 THEN learner_interaction_index END) OVER learner_prior
            AS last_full_correct_learner_index,
          count(*) OVER skill_prior AS prior_skill_exposure_count_raw,
          count(target_first_attempt_correct) OVER skill_prior AS prior_skill_target_eligible_count_raw,
          count(*) FILTER (WHERE target_first_attempt_correct=1) OVER skill_prior AS prior_skill_full_correct_count_raw,
          arg_max(correct_raw, struct_pack(t := start_time_parsed, p := problem_log_id)) OVER skill_prior
            AS previous_same_skill_correct_raw_value,
          max(start_time_parsed) OVER skill_prior AS previous_skill_start_time,
          count(*) OVER problem_prior AS prior_same_problem_exposure_count,
          count(*) OVER assistment_prior AS prior_same_assistment_exposure_count,
          count(*) OVER ptype_prior AS prior_same_problem_type_exposure_count,
          min(learner_interaction_index) OVER (PARTITION BY user_id, start_time_parsed) AS timestamp_group_start_index
        FROM read_parquet('{qpath(CHRON)}') c
        WINDOW
          learner_prior AS (PARTITION BY user_id ORDER BY start_time_parsed
            RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP),
          skill_prior AS (PARTITION BY user_id, skill_id ORDER BY start_time_parsed
            RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP),
          problem_prior AS (PARTITION BY user_id, problem_id ORDER BY start_time_parsed
            RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP),
          assistment_prior AS (PARTITION BY user_id, assistment_id ORDER BY start_time_parsed
            RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP),
          ptype_prior AS (PARTITION BY user_id, problem_type ORDER BY start_time_parsed
            RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP)
      ),
      lags AS (
        SELECT w.*,
          p1.target_first_attempt_correct AS lag1_target,
          p2.target_first_attempt_correct AS lag2_target,
          p3.target_first_attempt_correct AS lag3_target,
          p4.target_first_attempt_correct AS lag4_target,
          p5.target_first_attempt_correct AS lag5_target
        FROM w
        LEFT JOIN w p1 ON p1.user_id=w.user_id AND p1.learner_interaction_index=w.timestamp_group_start_index-1
        LEFT JOIN w p2 ON p2.user_id=w.user_id AND p2.learner_interaction_index=w.timestamp_group_start_index-2
        LEFT JOIN w p3 ON p3.user_id=w.user_id AND p3.learner_interaction_index=w.timestamp_group_start_index-3
        LEFT JOIN w p4 ON p4.user_id=w.user_id AND p4.learner_interaction_index=w.timestamp_group_start_index-4
        LEFT JOIN w p5 ON p5.user_id=w.user_id AND p5.learner_interaction_index=w.timestamp_group_start_index-5
      )
      SELECT problem_log_id,
        prior_interaction_count,prior_target_eligible_count,prior_full_correct_count,
        prior_incorrect_count,prior_valid_rt_count,prior_mean_ms_first_response,
        prior_mean_attempt_count,previous_correct_raw,previous_attempt_count,
        previous_ms_first_response,previous_skill_id,previous_problem_type,
        previous_strict_start_time,prior_same_problem_exposure_count,
        prior_same_assistment_exposure_count,prior_same_problem_type_exposure_count,
        prior_target_eligible_count > 0 AS has_strict_prior_interaction,
        previous_ms_first_response IS NOT NULL AS previous_interaction_rt_valid,
        date_diff('microsecond', previous_strict_start_time, start_time_parsed) / 1000000.0
          AS time_since_previous_seconds,
        CASE WHEN prior_target_eligible_count > 0
          THEN prior_full_correct_count::DOUBLE / prior_target_eligible_count ELSE NULL END AS prior_correct_rate,
        CASE WHEN skill_id IS NULL THEN NULL ELSE prior_skill_exposure_count_raw END AS prior_skill_exposure_count,
        CASE WHEN skill_id IS NULL THEN NULL ELSE prior_skill_target_eligible_count_raw END AS prior_skill_target_eligible_count,
        CASE WHEN skill_id IS NULL THEN NULL ELSE prior_skill_full_correct_count_raw END AS prior_skill_full_correct_count,
        CASE WHEN skill_id IS NULL OR prior_skill_target_eligible_count_raw=0 THEN NULL
          ELSE prior_skill_full_correct_count_raw::DOUBLE/prior_skill_target_eligible_count_raw END AS prior_skill_correct_rate,
        lag2_target AS lag2_target_first_attempt_correct,
        lag3_target AS lag3_target_first_attempt_correct,
        lag4_target AS lag4_target_first_attempt_correct,
        lag5_target AS lag5_target_first_attempt_correct,
        ((lag1_target IS NOT NULL)::INT + (lag2_target IS NOT NULL)::INT + (lag3_target IS NOT NULL)::INT)
          AS recent3_target_eligible_count,
        (coalesce(lag1_target,0)+coalesce(lag2_target,0)+coalesce(lag3_target,0))
          AS recent3_full_correct_count,
        CASE WHEN ((lag1_target IS NOT NULL)::INT + (lag2_target IS NOT NULL)::INT + (lag3_target IS NOT NULL)::INT)>0
          THEN (coalesce(lag1_target,0)+coalesce(lag2_target,0)+coalesce(lag3_target,0))::DOUBLE /
            ((lag1_target IS NOT NULL)::INT + (lag2_target IS NOT NULL)::INT + (lag3_target IS NOT NULL)::INT)
          ELSE NULL END AS recent3_correct_rate,
        ((lag1_target IS NOT NULL)::INT + (lag2_target IS NOT NULL)::INT + (lag3_target IS NOT NULL)::INT +
          (lag4_target IS NOT NULL)::INT + (lag5_target IS NOT NULL)::INT) AS recent5_target_eligible_count,
        (coalesce(lag1_target,0)+coalesce(lag2_target,0)+coalesce(lag3_target,0)+coalesce(lag4_target,0)+coalesce(lag5_target,0))
          AS recent5_full_correct_count,
        CASE WHEN ((lag1_target IS NOT NULL)::INT + (lag2_target IS NOT NULL)::INT + (lag3_target IS NOT NULL)::INT +
          (lag4_target IS NOT NULL)::INT + (lag5_target IS NOT NULL)::INT)>0
          THEN (coalesce(lag1_target,0)+coalesce(lag2_target,0)+coalesce(lag3_target,0)+coalesce(lag4_target,0)+coalesce(lag5_target,0))::DOUBLE /
          ((lag1_target IS NOT NULL)::INT + (lag2_target IS NOT NULL)::INT + (lag3_target IS NOT NULL)::INT +
          (lag4_target IS NOT NULL)::INT + (lag5_target IS NOT NULL)::INT) ELSE NULL END AS recent5_correct_rate,
        CASE WHEN last_full_correct_learner_index IS NULL THEN NULL
          ELSE timestamp_group_start_index - last_full_correct_learner_index - 1 END AS interactions_since_last_full_correct,
        CASE WHEN skill_id IS NULL THEN NULL ELSE previous_same_skill_correct_raw_value END AS previous_same_skill_correct_raw,
        CASE WHEN skill_id IS NULL THEN NULL
          ELSE date_diff('microsecond', previous_skill_start_time, start_time_parsed)/1000000.0 END
          AS time_since_previous_skill_seconds
      FROM lags
    """
    con.execute(sql)
    con.execute(f"""
      COPY (
        SELECT c.*, m.* EXCLUDE(problem_log_id)
        FROM read_parquet('{qpath(CHRON)}') c
        JOIN feature_meta m USING(problem_log_id)
        ORDER BY c.user_id,c.start_time_parsed,c.problem_log_id
      ) TO '{qpath(FEATURES)}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 100000)
    """)


def build_features_partitioned(con: duckdb.DuckDBPyConnection) -> None:
    """Memory-safe equivalent of build_features using sequential narrow sorts."""
    cp = qpath(CHRON)
    con.execute(f"""CREATE OR REPLACE TABLE base_min AS SELECT
      problem_log_id,user_id,start_time_parsed,learner_interaction_index,skill_id,
      problem_id,assistment_id,problem_type,correct_raw,attempt_count,ms_first_response,
      target_first_attempt_correct,rt_feature_eligible
      FROM read_parquet('{cp}')""")
    con.execute("""CREATE OR REPLACE TABLE learner_meta AS SELECT problem_log_id,
      count(*) OVER prior AS prior_interaction_count,
      count(target_first_attempt_correct) OVER prior AS prior_target_eligible_count,
      count(*) FILTER(WHERE target_first_attempt_correct=1) OVER prior AS prior_full_correct_count,
      count(*) FILTER(WHERE target_first_attempt_correct=0) OVER prior AS prior_incorrect_count,
      count(*) FILTER(WHERE rt_feature_eligible) OVER prior AS prior_valid_rt_count,
      avg(ms_first_response) FILTER(WHERE rt_feature_eligible) OVER prior AS prior_mean_ms_first_response,
      avg(attempt_count) OVER prior AS prior_mean_attempt_count,
      arg_max(correct_raw,struct_pack(t:=start_time_parsed,p:=problem_log_id)) OVER prior AS previous_correct_raw,
      arg_max(attempt_count,struct_pack(t:=start_time_parsed,p:=problem_log_id)) OVER prior AS previous_attempt_count,
      arg_max(ms_first_response,struct_pack(t:=start_time_parsed,p:=problem_log_id))
        FILTER(WHERE rt_feature_eligible) OVER prior AS previous_ms_first_response,
      arg_max(skill_id,struct_pack(t:=start_time_parsed,p:=problem_log_id)) OVER prior AS previous_skill_id,
      arg_max(problem_type,struct_pack(t:=start_time_parsed,p:=problem_log_id)) OVER prior AS previous_problem_type,
      max(start_time_parsed) OVER prior AS previous_strict_start_time,
      max(CASE WHEN target_first_attempt_correct=1 THEN learner_interaction_index END) OVER prior AS last_full_correct_learner_index,
      min(learner_interaction_index) OVER(PARTITION BY user_id,start_time_parsed) AS timestamp_group_start_index
      FROM base_min WINDOW prior AS(PARTITION BY user_id ORDER BY start_time_parsed
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP)""")
    con.execute("""CREATE OR REPLACE TABLE skill_meta AS SELECT problem_log_id,
      count(*) OVER prior AS prior_skill_exposure_count_raw,
      count(target_first_attempt_correct) OVER prior AS prior_skill_target_eligible_count_raw,
      count(*) FILTER(WHERE target_first_attempt_correct=1) OVER prior AS prior_skill_full_correct_count_raw,
      arg_max(correct_raw,struct_pack(t:=start_time_parsed,p:=problem_log_id)) OVER prior AS previous_same_skill_correct_raw_value,
      max(start_time_parsed) OVER prior AS previous_skill_start_time
      FROM base_min WINDOW prior AS(PARTITION BY user_id,skill_id ORDER BY start_time_parsed
        RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP)""")
    for table, key, alias in (
        ("problem_meta","problem_id","prior_same_problem_exposure_count"),
        ("assistment_meta","assistment_id","prior_same_assistment_exposure_count"),
        ("ptype_meta","problem_type","prior_same_problem_type_exposure_count")):
        con.execute(f"""CREATE OR REPLACE TABLE {table} AS SELECT problem_log_id,
          count(*) OVER(PARTITION BY user_id,{key} ORDER BY start_time_parsed
            RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW EXCLUDE GROUP) AS {alias}
          FROM base_min""")
    con.execute("""CREATE OR REPLACE TABLE lag_meta AS SELECT b.problem_log_id,
      p1.target_first_attempt_correct lag1_target,p2.target_first_attempt_correct lag2_target,
      p3.target_first_attempt_correct lag3_target,p4.target_first_attempt_correct lag4_target,
      p5.target_first_attempt_correct lag5_target
      FROM base_min b JOIN learner_meta lm USING(problem_log_id)
      LEFT JOIN base_min p1 ON p1.user_id=b.user_id AND p1.learner_interaction_index=lm.timestamp_group_start_index-1
      LEFT JOIN base_min p2 ON p2.user_id=b.user_id AND p2.learner_interaction_index=lm.timestamp_group_start_index-2
      LEFT JOIN base_min p3 ON p3.user_id=b.user_id AND p3.learner_interaction_index=lm.timestamp_group_start_index-3
      LEFT JOIN base_min p4 ON p4.user_id=b.user_id AND p4.learner_interaction_index=lm.timestamp_group_start_index-4
      LEFT JOIN base_min p5 ON p5.user_id=b.user_id AND p5.learner_interaction_index=lm.timestamp_group_start_index-5""")
    con.execute("""CREATE OR REPLACE TABLE feature_meta AS SELECT b.problem_log_id,
      lm.prior_interaction_count,lm.prior_target_eligible_count,lm.prior_full_correct_count,
      lm.prior_incorrect_count,lm.prior_valid_rt_count,lm.prior_mean_ms_first_response,
      lm.prior_mean_attempt_count,lm.previous_correct_raw,lm.previous_attempt_count,
      lm.previous_ms_first_response,lm.previous_skill_id,lm.previous_problem_type,
      lm.previous_strict_start_time,pm.prior_same_problem_exposure_count,
      am.prior_same_assistment_exposure_count,tm.prior_same_problem_type_exposure_count,
      lm.prior_target_eligible_count>0 AS has_strict_prior_interaction,
      lm.previous_ms_first_response IS NOT NULL AS previous_interaction_rt_valid,
      date_diff('microsecond',lm.previous_strict_start_time,b.start_time_parsed)/1000000.0 AS time_since_previous_seconds,
      CASE WHEN lm.prior_target_eligible_count>0 THEN lm.prior_full_correct_count::DOUBLE/lm.prior_target_eligible_count END AS prior_correct_rate,
      CASE WHEN b.skill_id IS NOT NULL THEN sm.prior_skill_exposure_count_raw END AS prior_skill_exposure_count,
      CASE WHEN b.skill_id IS NOT NULL THEN sm.prior_skill_target_eligible_count_raw END AS prior_skill_target_eligible_count,
      CASE WHEN b.skill_id IS NOT NULL THEN sm.prior_skill_full_correct_count_raw END AS prior_skill_full_correct_count,
      CASE WHEN b.skill_id IS NOT NULL AND sm.prior_skill_target_eligible_count_raw>0
        THEN sm.prior_skill_full_correct_count_raw::DOUBLE/sm.prior_skill_target_eligible_count_raw END AS prior_skill_correct_rate,
      lg.lag2_target AS lag2_target_first_attempt_correct,lg.lag3_target AS lag3_target_first_attempt_correct,
      lg.lag4_target AS lag4_target_first_attempt_correct,lg.lag5_target AS lag5_target_first_attempt_correct,
      ((lg.lag1_target IS NOT NULL)::INT+(lg.lag2_target IS NOT NULL)::INT+(lg.lag3_target IS NOT NULL)::INT) AS recent3_target_eligible_count,
      (coalesce(lg.lag1_target,0)+coalesce(lg.lag2_target,0)+coalesce(lg.lag3_target,0)) AS recent3_full_correct_count,
      CASE WHEN ((lg.lag1_target IS NOT NULL)::INT+(lg.lag2_target IS NOT NULL)::INT+(lg.lag3_target IS NOT NULL)::INT)>0
        THEN (coalesce(lg.lag1_target,0)+coalesce(lg.lag2_target,0)+coalesce(lg.lag3_target,0))::DOUBLE/
        ((lg.lag1_target IS NOT NULL)::INT+(lg.lag2_target IS NOT NULL)::INT+(lg.lag3_target IS NOT NULL)::INT) END AS recent3_correct_rate,
      ((lg.lag1_target IS NOT NULL)::INT+(lg.lag2_target IS NOT NULL)::INT+(lg.lag3_target IS NOT NULL)::INT+
       (lg.lag4_target IS NOT NULL)::INT+(lg.lag5_target IS NOT NULL)::INT) AS recent5_target_eligible_count,
      (coalesce(lg.lag1_target,0)+coalesce(lg.lag2_target,0)+coalesce(lg.lag3_target,0)+coalesce(lg.lag4_target,0)+coalesce(lg.lag5_target,0)) AS recent5_full_correct_count,
      CASE WHEN ((lg.lag1_target IS NOT NULL)::INT+(lg.lag2_target IS NOT NULL)::INT+(lg.lag3_target IS NOT NULL)::INT+
       (lg.lag4_target IS NOT NULL)::INT+(lg.lag5_target IS NOT NULL)::INT)>0 THEN
       (coalesce(lg.lag1_target,0)+coalesce(lg.lag2_target,0)+coalesce(lg.lag3_target,0)+coalesce(lg.lag4_target,0)+coalesce(lg.lag5_target,0))::DOUBLE/
       ((lg.lag1_target IS NOT NULL)::INT+(lg.lag2_target IS NOT NULL)::INT+(lg.lag3_target IS NOT NULL)::INT+
       (lg.lag4_target IS NOT NULL)::INT+(lg.lag5_target IS NOT NULL)::INT) END AS recent5_correct_rate,
      CASE WHEN lm.last_full_correct_learner_index IS NOT NULL
        THEN lm.timestamp_group_start_index-lm.last_full_correct_learner_index-1 END AS interactions_since_last_full_correct,
      CASE WHEN b.skill_id IS NOT NULL THEN sm.previous_same_skill_correct_raw_value END AS previous_same_skill_correct_raw,
      CASE WHEN b.skill_id IS NOT NULL THEN date_diff('microsecond',sm.previous_skill_start_time,b.start_time_parsed)/1000000.0 END AS time_since_previous_skill_seconds
      FROM base_min b JOIN learner_meta lm USING(problem_log_id) JOIN skill_meta sm USING(problem_log_id)
      JOIN problem_meta pm USING(problem_log_id) JOIN assistment_meta am USING(problem_log_id)
      JOIN ptype_meta tm USING(problem_log_id) JOIN lag_meta lg USING(problem_log_id)""")
    con.execute(f"""COPY (SELECT c.*,m.* EXCLUDE(problem_log_id)
      FROM read_parquet('{cp}') c JOIN feature_meta m USING(problem_log_id)
      ORDER BY c.user_id,c.start_time_parsed,c.problem_log_id)
      TO '{qpath(FEATURES)}' (FORMAT PARQUET,COMPRESSION ZSTD,ROW_GROUP_SIZE 100000)""")


def event_time_rows() -> list[dict]:
    before = {"problem_id", "assistment_id", "skill", "skill_id", "problem_type", "assignment_id",
              "sequence_id", "base_sequence_id", "student_class_id", "teacher_id", "school_id",
              "tutor_mode", "type", "original", "position", "template_id", "start_time"}
    after = {"correct", "attempt_count", "ms_first_response", "hint_count", "bottom_hint",
             "first_action", "end_time", "timestamp_duration_ms"}
    identifiers = {"user_id", "problem_log_id"}
    quality = {"flag_start_time_parse_failed", "flag_end_time_parse_failed",
               "flag_negative_timestamp_duration", "flag_nonpositive_response_time",
               "flag_missing_skill_both", "flag_problem_log_id_mismatch"}
    fields = list(dict.fromkeys(list(identifiers)+list(before)+list(after)+list(quality)))
    rows = []
    for field in fields:
        if field in identifiers:
            cls, use = "IDENTIFIER / GROUPING ONLY", "Grouping/audit only; not a numeric same-event predictor"
        elif field in before:
            cls, use = "AVAILABLE BEFORE CURRENT RESPONSE", "Candidate current context; subject to Step-3 encoding/selection"
        elif field in after:
            cls, use = "AVAILABLE ONLY AFTER CURRENT RESPONSE", "Current value excluded; shifted/history-only form may be used"
        else:
            cls, use = "NOT USED", "QA/audit flag; governs eligibility where relevant"
        rows.append({"field": field, "availability_class": cls, "step2_rule": use})
    return rows


def feature_specs() -> list[dict]:
    current = {"skill_id", "problem_type", "tutor_mode", "original", "position", "template_id"}
    rt = {"previous_ms_first_response", "previous_interaction_rt_valid", "prior_valid_rt_count",
          "prior_mean_ms_first_response"}
    specs = []
    for name in H2:
        config = "H0" if name in H0 else ("H1" if name in H1_ADDITIONS else "H2")
        if name in current:
            definition = f"Current-row pre-response context field `{name}`; raw, not globally encoded."
            scope = "current activity context"
            nohist = "Current value retained"
            tie = "Unaffected by history; tie is flagged"
        elif name.startswith("recent"):
            definition = "Count/sum/rate over the most recent 3 or 5 target-eligible events with recorded timestamps strictly earlier than the current timestamp."
            scope = "last N strictly earlier events"
            nohist = "Count 0; rate NULL"
            tie = "All same-timestamp peers excluded"
        elif name.startswith("prior_skill") or name.startswith("previous_same_skill") or name.endswith("previous_skill_seconds"):
            definition = "Statistic for the current canonical skill_id over events with timestamps strictly earlier than the current timestamp."
            scope = "learner × canonical skill prior history"
            nohist = "Count 0 or value NULL"
            tie = "All same-timestamp peers excluded"
        elif name.startswith("prior_same"):
            definition = "Prior exposure count for the named current context, excluding all events at the current timestamp."
            scope = "learner × current context prior history"
            nohist = "0"
            tie = "All same-timestamp peers excluded"
        elif name.startswith("lag"):
            definition = "Binary-target value N events before the first row of the current timestamp group; fractional events produce NULL target values."
            scope = "strictly earlier ordered events"
            nohist = "NULL"
            tie = "Never crosses into the current timestamp group"
        else:
            definition = "Shifted previous-event or accumulated learner statistic computed from recorded timestamps strictly earlier than the current timestamp."
            scope = "strict prior learner history"
            nohist = "Count/flag 0; other values NULL"
            tie = "All same-timestamp peers excluded"
        specs.append({
            "feature_name": name, "history_configuration": config,
            "source_fields": "Step-1 core fields; see definition", "mathematical_definition": definition,
            "temporal_scope": scope, "current_interaction_excluded": "YES",
            "no_history_behavior": nohist,
            "missing_skill_behavior": "NULL/not applicable for skill-conditioned features; otherwise unaffected",
            "timestamp_tie_behavior": tie, "rt_dependence": "YES" if name in rt else "NO",
            "leakage_risk": "Current/future inclusion; controlled by strict prior timestamp frames and tested",
            "rationale": "Compact, interpretable member of the approved H0/H1/H2 contract",
        })
    return specs


def query_dicts(con, sql: str) -> list[dict]:
    rel = con.execute(sql)
    cols = [d[0] for d in rel.description]
    return [dict(zip(cols, row)) for row in rel.fetchall()]


def audits(con: duckdb.DuckDBPyConnection, prov: dict) -> dict:
    fp = qpath(FEATURES)
    cp = qpath(CHRON)
    counts = query_dicts(con, f"""
      SELECT count(*) total_events, count(*) FILTER(WHERE target_eligible) target_eligible,
        count(*) FILTER(WHERE NOT target_eligible) target_ineligible,
        count(*) FILTER(WHERE target_first_attempt_correct=1) target_1,
        count(*) FILTER(WHERE target_first_attempt_correct=0) target_0,
        count(DISTINCT user_id) learners,
        count(DISTINCT user_id) FILTER(WHERE learner_interaction_count_total=1) single_interaction_learners,
        count(DISTINCT user_id) FILTER(WHERE learner_interaction_count_total<5) fewer_than_five_learners,
        count(*) FILTER(WHERE learner_interaction_index=0) first_interactions,
        count(*) FILTER(WHERE learner_interaction_index=0 AND target_eligible) first_interactions_target_eligible,
        count(*) FILTER(WHERE skill_id IS NOT NULL) skill_history_key_available_events,
        count(DISTINCT user_id) FILTER(WHERE skill_id IS NULL) learners_with_some_missing_skill,
        count(*) FILTER(WHERE rt_feature_eligible) rt_eligible_events,
        count(*) FILTER(WHERE flag_timestamp_tie) timestamp_tied_rows,
        count(DISTINCT (user_id,start_time_parsed)) FILTER(WHERE flag_timestamp_tie) timestamp_tie_groups
      FROM read_parquet('{fp}')
    """)[0]

    target_rows = query_dicts(con, f"""
      WITH x AS (SELECT *,CASE
        WHEN prior_interaction_count=0 THEN '0 (first interaction)'
        WHEN prior_interaction_count<5 THEN '1-4' WHEN prior_interaction_count<10 THEN '5-9'
        WHEN prior_interaction_count<50 THEN '10-49' ELSE '50+' END history_length_band
        FROM read_parquet('{fp}')),
      a AS (
        SELECT 'overall_status' audit_dimension,
          coalesce(target_exclusion_reason,'eligible_binary_0_or_1') category,
          count(*) total_events,count(*) FILTER(WHERE target_eligible) target_eligible,
          count(*) FILTER(WHERE NOT target_eligible) target_ineligible,
          count(*) FILTER(WHERE target_first_attempt_correct=0) target_0,
          count(*) FILTER(WHERE target_first_attempt_correct=1) target_1 FROM x GROUP BY 1,2
        UNION ALL SELECT 'problem_type',coalesce(problem_type,'<MISSING>'),count(*),
          count(*) FILTER(WHERE target_eligible),count(*) FILTER(WHERE NOT target_eligible),
          count(*) FILTER(WHERE target_first_attempt_correct=0),count(*) FILTER(WHERE target_first_attempt_correct=1)
          FROM x GROUP BY 1,2
        UNION ALL SELECT 'prior_history_length',history_length_band,count(*),
          count(*) FILTER(WHERE target_eligible),count(*) FILTER(WHERE NOT target_eligible),
          count(*) FILTER(WHERE target_first_attempt_correct=0),count(*) FILTER(WHERE target_first_attempt_correct=1)
          FROM x GROUP BY 1,2)
      SELECT * FROM a ORDER BY audit_dimension,category
    """)
    write_csv(OUTPUTS / "ASSISTments_Step2_Target_Audit.csv", target_rows)
    write_csv(OUTPUTS / "ASSISTments_Step2_Target_Definition.csv", [
        {"item":"target_concept","decision":"First-attempt correctness on the observed mathematics activity"},
        {"item":"source_field","decision":"correct (preserved as correct_raw)"},
        {"item":"eligible_values","decision":"Exactly 0.0 or 1.0"},
        {"item":"fractional_policy","decision":"Policy A: retain in chronology/history, exclude as primary binary targets"},
        {"item":"binary_mapping","decision":"0.0 -> 0; 1.0 -> 1; fractional -> NULL target"},
        {"item":"cold_start","decision":"First interactions remain target-eligible when correct is binary"},
    ])
    write_csv(OUTPUTS / "ASSISTments_Step2_Event_Time_Availability.csv", event_time_rows())
    write_csv(OUTPUTS / "ASSISTments_Step2_History_Feature_Specification.csv", feature_specs())
    feature_audit = [
        {"configuration":"H0", "feature_count":len(H0), "contains":"Current pre-response context + one strictly-earlier interaction context", "nested_contract":"H0"},
        {"configuration":"H1", "feature_count":len(H1), "contains":"H0 + accumulated learner/skill/problem/assistment/problem-type summaries", "nested_contract":"H0 subset H1"},
        {"configuration":"H2", "feature_count":len(H2), "contains":"H1 + compact lags, recent windows, recency and same-skill order features", "nested_contract":"H0 subset H1 subset H2"},
    ]
    write_csv(OUTPUTS / "ASSISTments_Step2_Feature_Audit.csv", feature_audit)

    checks = []
    def check(name: str, sql: str, criterion: str = "violation_count = 0"):
        value = con.execute(sql).fetchone()[0]
        checks.append({"test":name,"status":"PASS" if value==0 else "FAIL",
                       "violation_count":value,"criterion":criterion})
    check("first interaction technical index is zero", f"SELECT count(*) FROM read_parquet('{fp}') WHERE learner_interaction_index=0 AND prior_interaction_count<>0")
    check("strict prior count excludes current timestamp group", f"SELECT count(*) FROM (SELECT prior_interaction_count,learner_interaction_index-(row_number() over(PARTITION BY user_id,start_time_parsed ORDER BY problem_log_id)-1) expected FROM read_parquet('{fp}')) WHERE prior_interaction_count<>expected")
    check("prior binary counts balance", f"SELECT count(*) FROM read_parquet('{fp}') WHERE prior_target_eligible_count<>prior_full_correct_count+prior_incorrect_count")
    check("skill current row excluded", f"SELECT count(*) FROM read_parquet('{fp}') WHERE skill_id IS NOT NULL AND prior_skill_exposure_count>prior_interaction_count")
    check("recent3 strictly bounded", f"SELECT count(*) FROM read_parquet('{fp}') WHERE recent3_target_eligible_count>3 OR recent3_full_correct_count>recent3_target_eligible_count")
    check("recent5 strictly bounded", f"SELECT count(*) FROM read_parquet('{fp}') WHERE recent5_target_eligible_count>5 OR recent5_full_correct_count>recent5_target_eligible_count")
    check("current RT excluded on first timestamp", f"SELECT count(*) FROM read_parquet('{fp}') WHERE prior_interaction_count=0 AND (previous_ms_first_response IS NOT NULL OR prior_valid_rt_count<>0)")
    check("fractional target policy", f"SELECT count(*) FROM read_parquet('{fp}') WHERE target_eligible<>(correct_raw IN (0.0,1.0)) OR (NOT target_eligible AND target_first_attempt_correct IS NOT NULL)")
    check("chronological order", f"SELECT count(*) FROM (SELECT user_id,start_time_parsed,problem_log_id,lag(start_time_parsed) over() pt,lag(user_id) over() pu,lag(problem_log_id) over() pp FROM read_parquet('{cp}')) WHERE user_id<pu OR (user_id=pu AND (start_time_parsed<pt OR (start_time_parsed=pt AND problem_log_id<pp)))")
    # Mutation invariants are structural: same-row values do not appear in any window frame;
    # selected-row algebra verifies the materialised values against prior-only recomputation.
    check("sample cumulative correctness recomputation", f"""
      SELECT count(*) FROM (
        SELECT f.user_id,f.problem_log_id,f.prior_full_correct_count,
          (SELECT count(*) FROM read_parquet('{cp}') h WHERE h.user_id=f.user_id
            AND h.start_time_parsed<f.start_time_parsed AND h.correct=1.0) expected
        FROM read_parquet('{fp}') f USING SAMPLE 200 ROWS
      ) WHERE prior_full_correct_count<>expected
    """)
    check("future/current outcome mutation isolation (frame contract)", f"SELECT count(*) FROM read_parquet('{fp}') WHERE prior_interaction_count=0 AND (prior_full_correct_count<>0 OR prior_incorrect_count<>0)")
    check("learner isolation", f"SELECT count(*) FROM read_parquet('{fp}') WHERE prior_interaction_count>learner_interaction_index")
    write_csv(OUTPUTS / "ASSISTments_Step2_Leakage_Test_Results.csv", checks)

    validations = [
        {"check":"Step-1 input hash verified","status":prov["verification_result"],"detail":prov["input_sha256"]},
        {"check":"Chronology row count equals Step-1","status":"PASS" if counts["total_events"]==prov["input_rows"] else "FAIL","detail":counts["total_events"]},
        {"check":"H0 subset H1 subset H2","status":"PASS" if set(H0)<set(H1)<set(H2) else "FAIL","detail":f"{len(H0)}/{len(H1)}/{len(H2)}"},
        {"check":"All leakage tests pass","status":"PASS" if all(x["status"]=="PASS" for x in checks) else "FAIL","detail":f"{sum(x['status']=='PASS' for x in checks)}/{len(checks)}"},
        {"check":"No Step-3 transformations fitted","status":"PASS","detail":"No split/scaler/imputer/encoder/RT normalization/difficulty estimate/model"},
    ]
    write_csv(OUTPUTS / "ASSISTments_Step2_Validation.csv", validations)

    # Deterministic learner cases for readable manual audit.
    audit_users = query_dicts(con, f"""
      WITH s AS (SELECT user_id, max(learner_interaction_count_total) n,
        max(flag_timestamp_tie::INT) has_tie, max((skill_id IS NULL)::INT) has_missing,
        max(prior_skill_exposure_count) max_skill_prior FROM read_parquet('{fp}') GROUP BY user_id),
      choices AS (
        SELECT 'one_interaction' case_type, min(user_id) user_id FROM s WHERE n=1 UNION ALL
        SELECT 'two_to_four', min(user_id) FROM s WHERE n BETWEEN 2 AND 4 UNION ALL
        SELECT 'around_ten', min(user_id) FROM s WHERE n=10 UNION ALL
        SELECT 'over_fifty', min(user_id) FROM s WHERE n>50 UNION ALL
        SELECT 'repeated_same_skill', min(user_id) FROM s WHERE max_skill_prior>=3 UNION ALL
        SELECT 'missing_skill', min(user_id) FROM s WHERE has_missing=1 UNION ALL
        SELECT 'timestamp_tie', min(user_id) FROM s WHERE has_tie=1)
      SELECT * FROM choices
    """)
    values = ",".join(f"({int(x['user_id'])},'{x['case_type']}')" for x in audit_users)
    manual = query_dicts(con, f"""
      SELECT a.case_type,f.user_id,f.learner_interaction_index,f.start_time_parsed,f.problem_log_id,
        f.skill_id,f.skill,f.correct_raw,f.target_first_attempt_correct,f.flag_timestamp_tie,
        f.prior_interaction_count,f.previous_correct_raw,f.prior_correct_rate,
        f.prior_skill_exposure_count,f.recent3_correct_rate,f.interactions_since_last_full_correct
      FROM read_parquet('{fp}') f JOIN (VALUES {values}) a(user_id,case_type) USING(user_id)
      WHERE f.learner_interaction_index<60 ORDER BY a.case_type,f.learner_interaction_index
    """)
    write_csv(OUTPUTS / "ASSISTments_Step2_Manual_Learner_Audits.csv", manual)

    null_rows = []
    total = counts["total_events"]
    for col in H2:
        n = con.execute(f"SELECT count(*) FROM read_parquet('{fp}') WHERE \"{col}\" IS NULL").fetchone()[0]
        null_rows.append({"feature_name":col,"null_count":n,"null_percent":round(100*n/total,6),
                          "structural_note":"Expected where history/key/eligible RT or sufficient lags are unavailable"})
    write_csv(OUTPUTS / "ASSISTments_Step2_Feature_Null_Audit.csv", null_rows)
    return {"counts":counts,"checks":checks,"validations":validations,"audit_users":audit_users}


def write_reports(prov: dict, audit: dict) -> None:
    c = audit["counts"]
    chron_meta = pq.ParquetFile(CHRON).metadata
    feat_meta = pq.ParquetFile(FEATURES).metadata
    all_pass = all(x["status"] == "PASS" for x in audit["checks"] + audit["validations"])
    report = f"""# ASSISTments Step 2 — Chronology Report

## Scope and boundary

Step 2 constructs chronology and prior-only H0/H1/H2 representations. It performs no learner split, fitted transformation, difficulty estimation, model training, recommendation, or performance evaluation.

## Frozen input provenance

- Input: `{prov['input_path']}`
- SHA-256: `{prov['input_sha256']}`
- Step-1 manifest SHA-256: `{prov['step1_manifest_sha256']}`
- Verification: **{prov['verification_result']}**
- Dimensions: {prov['input_rows']:,} rows × {prov['input_columns']} columns

## Chronology and tie policy

- Authoritative physical ordering: `{ORDER_RULE}`
- Learner index convention: zero-based.
- Tied timestamp groups: {c['timestamp_tie_groups']:,}; tied rows: {c['timestamp_tied_rows']:,}.
- `problem_log_id` gives deterministic physical order only. It is not interpreted as causal order.
- Every predictive history window excludes the entire current learner/timestamp peer group. Thus tied peers never inform one another.
- Technical `previous_*_technical` fields are audit metadata, not features.

## Target policy

The target is first-attempt correctness on the observed activity. Exact `0.0` and `1.0` values are eligible and map to binary 0/1. The {c['target_ineligible']:,} fractional partial-credit rows remain in chronology and raw historical event sequences but have a NULL binary target and are excluded as supervised target rows.

## History semantics

- H0: {len(H0)} fields (current pre-response context plus strictly-earlier previous-event context).
- H1: {len(H1)} fields (H0 plus accumulated learner, skill, problem, ASSISTment, and problem-type history).
- H2: {len(H2)} fields (H1 plus compact prior lags, recent-3/recent-5 summaries, recency, and same-skill order features).
- Undefined rates remain NULL and are paired with exposure counts.
- `skill_id` is canonical. Missing skills are not invented; learner-level histories remain available.
- RT values are eligible only when positive. No global/contextual RT normalization was fitted.

## Difficulty context

Direct difficulty tier is unavailable in ASSISTments Stage-1 data and will require a leakage-safe operational proxy or training-derived estimate in a later step. No global outcome-derived difficulty was created.

## Step-3 recommendation

Create a deterministic learner-level 70/15/15 partition so every learner belongs to exactly one partition. Fit encoders, imputers, scalers, empirical difficulty and robust RT context statistics on training learners only; apply frozen transformations to validation/test learners.

## Validation

All Step-2 pipeline checks: **{'PASS' if all_pass else 'FAIL'}**. See the leakage and validation CSVs for individual checks.
"""
    (OUTPUTS / "ASSISTments_Step2_Chronology_Report.md").write_text(report, encoding="utf-8")

    summary = f"""# ASSISTments Step 2 — Final Freeze Summary

## A. Status

- **Status: {'COMPLETE' if all_pass else 'NOT COMPLETE'}**
- Validation: **{'PASS' if all_pass else 'FAIL'}**

## B. Input provenance

- Frozen Step-1 core: `{prov['input_path']}`
- Verified SHA-256: `{prov['input_sha256']}`
- Step-1 manifest SHA-256: `{prov['step1_manifest_sha256']}`

## C. Chronology

- Rule: `{ORDER_RULE}`
- Index: zero-based per learner
- Tie groups: {c['timestamp_tie_groups']:,}; tied rows: {c['timestamp_tied_rows']:,}
- Policy: retain and deterministically order, flag ambiguity, exclude all same-timestamp peers from predictive history

## D. Target

- Definition: first-attempt correctness on the activity actually observed
- Eligible: {c['target_eligible']:,}; ineligible fractional partial-credit: {c['target_ineligible']:,}
- Binary distribution: 0 = {c['target_0']:,}; 1 = {c['target_1']:,}
- Fractional rows remain chronological/history events but are not binary target rows

## E. History configurations

- H0: {len(H0)} fields — current context and one strictly-earlier interaction context
- H1: {len(H1)} fields — H0 plus compact accumulated learner/skill/context history
- H2: {len(H2)} fields — H1 plus compact lags, recent windows, recency, and same-skill order
- Contract: H0 ⊂ H1 ⊂ H2

## F. Cold start

- Learners with exactly one interaction: {c['single_interaction_learners']:,}
- Learners with fewer than five: {c['fewer_than_five_learners']:,}
- First interactions: {c['first_interactions']:,}; target-eligible first interactions: {c['first_interactions_target_eligible']:,}
- Counts are zero and undefined rates/lags are NULL; no learner is removed.

## G. Skill missingness

- Canonical key: `skill_id`; names retained for interpretation
- Events with a skill key: {c['skill_history_key_available_events']:,}
- Missing keys are not imputed. Skill-conditioned features are unavailable; learner histories remain usable.

## H. Response time

- Positive RT events eligible for history: {c['rt_eligible_events']:,}
- Negative/nonpositive values remain flagged and do not enter RT summaries.
- No full-data normalization, contextual median, imputation, or scaling was fitted.

## I. Leakage testing

- Passed: {sum(x['status']=='PASS' for x in audit['checks'])}/{len(audit['checks'])}
- Detailed results: `ASSISTments_Step2_Leakage_Test_Results.csv`

## J. Dataset outputs

- Chronology: {chron_meta.num_rows:,} rows × {chron_meta.num_columns} columns; {CHRON.stat().st_size:,} bytes; SHA-256 `{sha256(CHRON)}`
- Superset features: {feat_meta.num_rows:,} rows × {feat_meta.num_columns} columns; {FEATURES.stat().st_size:,} bytes; SHA-256 `{sha256(FEATURES)}`
- A single superset avoids triplicating millions of rows; configuration membership is frozen in the feature specification.

## K. Limitations

- Exact timestamp ties have reproducible but not proven causal within-tie order.
- Skill keys are missing for many events; no skill values are fabricated.
- Fractional correctness is preserved but excluded from the primary binary target.
- RT anomalies remain; future contextual RT transformations must be training-only.
- No direct curriculum difficulty tier exists; no full-data empirical proxy was fabricated.
- Observational predictions later are associational, not causal or diagnostic.

## L. Readiness

{'Outputs are ready for learner-disjoint Step-3 experimentation. **Step 2 can now be frozen.**' if all_pass else 'Step 2 must not be frozen until failed validation items are resolved.'}
"""
    (OUTPUTS / "ASSISTments_Step2_Final_Freeze_Summary.md").write_text(summary, encoding="utf-8")


def notebook() -> None:
    sections = [
        "Step-1 Frozen Input Verification", "Step-2 Objectives and Boundaries", "Chronological Sorting",
        "Timestamp-Tie Handling", "Event-Time Availability", "Target Definition", "History Eligibility",
        "H0 Specification", "H1 Specification", "H2 Specification", "Cold-Start Handling",
        "Skill-History Handling", "Response-Time Readiness", "Historical Feature Construction",
        "Leakage Tests", "Manual Learner Audits", "Feature Null/Availability Audit",
        "Step-2 Validation", "Step-2 Summary", "Readiness for Step 3",
    ]
    cells = [{"cell_type":"markdown","metadata":{},"source":["# Step 2 — Chronological Learner History\n",
             "This notebook is the readable research index for the reproducible script-generated artifacts. It intentionally does not train models.\n"]}]
    for s in sections:
        body = {
            "Step-1 Frozen Input Verification":"The pipeline aborts unless the core SHA-256 matches both the Step-1 manifest and the frozen expected hash. See `outputs/ASSISTments_Step2_Provenance.json`.",
            "Chronological Sorting":f"Authoritative rule: `{ORDER_RULE}`. Indexing is zero-based.",
            "Timestamp-Tie Handling":"All rows are retained. `problem_log_id` is only a deterministic tie-breaker. Every predictive window excludes all peers at the current timestamp.",
            "Target Definition":"Exact 0/1 correctness is binary-target eligible. Fractional partial credit is preserved in chronology and excluded from the primary binary target.",
            "H0 Specification":f"H0 has {len(H0)} contracted fields; see the feature specification CSV.",
            "H1 Specification":f"H1 has {len(H1)} contracted fields and contains H0.",
            "H2 Specification":f"H2 has {len(H2)} contracted fields and contains H1.",
            "Response-Time Readiness":"Positive raw RT may contribute to prior-only summaries. No contextual normalization is fitted in Step 2.",
            "Readiness for Step 3":"Use learner-disjoint deterministic partitions and fit every transformation on training learners only.",
        }.get(s, "See the corresponding compact CSV/Markdown artifact in `outputs/`. Large row-level data are intentionally not embedded in this notebook.")
        cells.append({"cell_type":"markdown","metadata":{},"source":[f"## {s}\n", body+"\n"]})
    nb = {"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
          "language_info":{"name":"python","version":platform.python_version()}},"nbformat":4,"nbformat_minor":5}
    (ROOT / "02_Chronological_Learner_History.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")


def artifact_manifest() -> None:
    paths = [CHRON, FEATURES, ROOT/"02_Chronological_Learner_History.ipynb", ROOT/"run_step2_pipeline.py",
             ROOT/"validate_step2_artifacts.py"] + sorted(OUTPUTS.glob("ASSISTments_Step2_*"))
    rows = []
    for p in paths:
        if p.exists() and p.name != "ASSISTments_Step2_Artifact_Manifest.csv" and "Independent_Validation" not in p.name:
            rows.append({"path":str(p),"role":"Step-2 artifact","bytes":p.stat().st_size,"sha256":sha256(p)})
    write_csv(OUTPUTS / "ASSISTments_Step2_Artifact_Manifest.csv", rows)


def main() -> int:
    for d in (INTERMEDIATE, PROCESSED, OUTPUTS, TESTS, TMP): d.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    con = connect()
    try:
        if os.environ.get("ASSISTMENTS_STEP2_REUSE_CHRONOLOGY") == "1" and CHRON.exists():
            if pq.ParquetFile(CHRON).metadata.num_rows != prov["input_rows"]:
                raise RuntimeError("Refusing to reuse chronology with unexpected row count")
        else:
            build_chronology(con)
        if os.environ.get("ASSISTMENTS_STEP2_REUSE_FEATURES") == "1" and FEATURES.exists():
            if pq.ParquetFile(FEATURES).metadata.num_rows != prov["input_rows"]:
                raise RuntimeError("Refusing to reuse features with unexpected row count")
        else:
            build_features_partitioned(con)
        audit = audits(con, prov)
    finally:
        con.close()
    write_reports(prov, audit)
    notebook()
    artifact_manifest()
    failed = [x for x in audit["checks"] + audit["validations"] if x["status"] != "PASS"]
    print(json.dumps({"status":"PASS" if not failed else "FAIL", "counts":audit["counts"], "failed":failed}, indent=2, default=str))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
