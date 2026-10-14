#!/usr/bin/env python3
"""Independent, read-only validator for frozen Step-2 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
STEP1 = ROOT.parent / "01_ASSISTments_Dataset_Preparation"
INPUT = STEP1 / "data" / "processed" / "ASSISTments_2012_2013_core.parquet"
CHRON = ROOT / "data" / "intermediate" / "ASSISTments_2012_2013_chronological.parquet"
FEATURES = ROOT / "data" / "processed" / "ASSISTments_2012_2013_history_features.parquet"
OUTPUTS = ROOT / "outputs"
MANIFEST = OUTPUTS / "ASSISTments_Step2_Artifact_Manifest.csv"
EXPECTED_INPUT = "32fcd2ee16ed635b26b48865f5e3130928c79f3a779af95d16ee5250b6d8d2f4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(8 * 1024 * 1024): h.update(b)
    return h.hexdigest()


def qp(path: Path) -> str:
    return str(path).replace("'", "''")


def main() -> int:
    required = [INPUT, CHRON, FEATURES, MANIFEST,
                OUTPUTS/"ASSISTments_Step2_History_Feature_Specification.csv",
                OUTPUTS/"ASSISTments_Step2_Leakage_Test_Results.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print(json.dumps({"status":"FAIL","missing":missing}, indent=2)); return 1

    results = []
    def add(name, passed, detail):
        results.append({"check":name,"status":"PASS" if passed else "FAIL","detail":str(detail)})

    actual_input = sha256(INPUT)
    add("Step-1 core SHA-256", actual_input == EXPECTED_INPUT, actual_input)
    manifest_bad = []
    with MANIFEST.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            p = Path(row["path"])
            if not p.exists() or int(row["bytes"]) != p.stat().st_size or row["sha256"] != sha256(p):
                manifest_bad.append(str(p))
    add("All manifest size/hash entries", not manifest_bad, manifest_bad or "all match")

    im, cm, fm = (pq.ParquetFile(x).metadata for x in (INPUT, CHRON, FEATURES))
    add("Row counts preserved", im.num_rows == cm.num_rows == fm.num_rows == 6_123_270,
        f"{im.num_rows}/{cm.num_rows}/{fm.num_rows}")
    feature_cols = set(pq.ParquetFile(FEATURES).schema_arrow.names)
    with (OUTPUTS/"ASSISTments_Step2_History_Feature_Specification.csv").open(newline="",encoding="utf-8-sig") as f:
        specs = list(csv.DictReader(f))
    contract = {r["history_configuration"]:set() for r in specs}
    for r in specs: contract[r["history_configuration"]].add(r["feature_name"])
    h0 = contract["H0"]
    h1 = h0 | contract["H1"]
    h2 = h1 | contract["H2"]
    add("Feature columns present", h2 <= feature_cols, sorted(h2-feature_cols))
    add("Nested H0/H1/H2 contract", h0 < h1 < h2, f"{len(h0)}/{len(h1)}/{len(h2)}")

    con = duckdb.connect()
    con.execute("SET threads=4")
    fp, cp = qp(FEATURES), qp(CHRON)
    def zero(name, sql):
        n = con.execute(sql).fetchone()[0]; add(name, n == 0, f"violations={n}")
    zero("Physical chronological ordering", f"""
      SELECT count(*) FROM (SELECT *,lag(user_id) over() pu,lag(start_time_parsed) over() pt,
        lag(problem_log_id) over() pp FROM read_parquet('{cp}'))
      WHERE user_id<pu OR (user_id=pu AND (start_time_parsed<pt OR
        (start_time_parsed=pt AND problem_log_id<pp)))
    """)
    zero("Zero-based learner index", f"""
      SELECT count(*) FROM (SELECT learner_interaction_index,
        row_number() over(PARTITION BY user_id ORDER BY start_time_parsed,problem_log_id)-1 expected
        FROM read_parquet('{fp}')) WHERE learner_interaction_index<>expected
    """)
    zero("Strict prior count independently recomputed", f"""
      SELECT count(*) FROM (SELECT prior_interaction_count, learner_interaction_index-
        (row_number() over(PARTITION BY user_id,start_time_parsed ORDER BY problem_log_id)-1) expected
        FROM read_parquet('{fp}')) WHERE prior_interaction_count<>expected
    """)
    zero("Binary target policy", f"""
      SELECT count(*) FROM read_parquet('{fp}') WHERE
        target_eligible<>(correct_raw IN (0.0,1.0)) OR
        (target_eligible AND target_first_attempt_correct<>CAST(correct_raw AS TINYINT)) OR
        (NOT target_eligible AND target_first_attempt_correct IS NOT NULL)
    """)
    zero("No same-timestamp contribution to learner count", f"""
      SELECT count(*) FROM (SELECT flag_timestamp_tie,prior_interaction_count,
        min(prior_interaction_count) over(PARTITION BY user_id,start_time_parsed) group_min
        FROM read_parquet('{fp}')) WHERE flag_timestamp_tie AND prior_interaction_count<>group_min
    """)
    zero("No current RT on first timestamp", f"""
      SELECT count(*) FROM read_parquet('{fp}') WHERE prior_interaction_count=0 AND
        (prior_valid_rt_count<>0 OR previous_ms_first_response IS NOT NULL)
    """)
    zero("Skill self-exclusion", f"""
      SELECT count(*) FROM read_parquet('{fp}') WHERE skill_id IS NOT NULL AND
        (prior_skill_exposure_count>prior_interaction_count OR prior_skill_target_eligible_count>prior_skill_exposure_count)
    """)
    zero("Recent window bounds", f"""
      SELECT count(*) FROM read_parquet('{fp}') WHERE recent3_target_eligible_count>3 OR
        recent5_target_eligible_count>5 OR recent3_full_correct_count>recent3_target_eligible_count OR
        recent5_full_correct_count>recent5_target_eligible_count
    """)
    zero("Independent sample prior-correct recomputation", f"""
      SELECT count(*) FROM (
        SELECT f.user_id,f.start_time_parsed,f.prior_full_correct_count,
          (SELECT count(*) FROM read_parquet('{cp}') h WHERE h.user_id=f.user_id
            AND h.start_time_parsed<f.start_time_parsed AND h.correct=1.0) expected
        FROM read_parquet('{fp}') f USING SAMPLE 250 ROWS
      ) WHERE prior_full_correct_count<>expected
    """)
    con.close()

    unit = subprocess.run([sys.executable, str(ROOT/"tests"/"test_leakage.py")], capture_output=True, text=True)
    add("Synthetic mutation tests", unit.returncode == 0, (unit.stdout+unit.stderr).strip())

    prior_csv = OUTPUTS/"ASSISTments_Step2_Leakage_Test_Results.csv"
    with prior_csv.open(newline="",encoding="utf-8-sig") as f:
        prior_results = list(csv.DictReader(f))
    add("Pipeline leakage results all PASS", all(r["status"]=="PASS" for r in prior_results),
        f"{sum(r['status']=='PASS' for r in prior_results)}/{len(prior_results)}")

    out_csv = OUTPUTS/"ASSISTments_Step2_Independent_Validation.csv"
    with out_csv.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["check","status","detail"]); w.writeheader(); w.writerows(results)
    payload={"all_checks_passed":all(r["status"]=="PASS" for r in results),"results":results}
    (OUTPUTS/"ASSISTments_Step2_Independent_Validation.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps(payload,indent=2))
    return 0 if payload["all_checks_passed"] else 1


if __name__ == "__main__": raise SystemExit(main())
