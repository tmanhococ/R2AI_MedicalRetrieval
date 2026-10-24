"""Validate a downloaded run: submission format plus non-degeneracy checks.

    .venv/Scripts/python.exe modal_pipeline/check_run.py artifacts/modal_runs/v1/p1_q30 --nq 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vb_common import quality_report, validate_results  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--queries", type=Path, default=Path("data/official/query.parquet"))
    ap.add_argument("--nq", type=int, default=0, help="number of queries in the run (0 = all)")
    args = ap.parse_args()

    ids = sorted(pq.read_table(args.queries, columns=["id"])["id"].to_pylist())
    ids = ids[: args.nq] if args.nq else ids
    with zipfile.ZipFile(args.run_dir / "submission.zip") as zf:
        names = zf.namelist()
        results = json.loads(zf.read("results.json"))
    errors = ([] if names == ["results.json"] else [f"zip entries {names}"]) + validate_results(results, ids)
    scored = pq.read_table(args.run_dir / "scored_pool.parquet", columns=["qid", "ce"]).to_pylist()
    report = {**quality_report(results, scored), "format_errors": errors[:5]}
    report["passed"] = report["passed"] and not errors
    print(json.dumps(report, indent=2))
    print("PASS" if report["passed"] else "FAIL")
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
