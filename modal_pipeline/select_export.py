"""Re-run output selection locally from scored_pool.parquet (no Modal credits needed).

    .venv/Scripts/python.exe modal_pipeline/select_export.py \
        --pool artifacts/modal_runs/v1/full/scored_pool.parquet \
        --queries data/official/query.parquet \
        --out artifacts/modal_runs/v1/full/try_docs60 --doc-max 60 --chunk-max 60
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vb_common import SelectionParams, build_results, results_to_zip_bytes, validate_results  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    for f in fields(SelectionParams):
        flag = "--" + f.name.replace("_", "-")
        if f.type in (bool, "bool"):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=f.default)
        else:
            ap.add_argument(flag, type=type(f.default), default=f.default)
    args = ap.parse_args()

    params = SelectionParams(**{f.name: getattr(args, f.name) for f in fields(SelectionParams)})
    rows = pq.read_table(args.pool, columns=["qid", "doc_id", "ce", "doc_rrf", "text"]).to_pylist()
    query_ids = pq.read_table(args.queries, columns=["id"])["id"].to_pylist()
    if len({r["qid"] for r in rows} - set(query_ids)):
        sys.exit("pool contains query ids not in the query file")
    if len(query_ids) != len({r["qid"] for r in rows}):
        print(f"note: {len(query_ids) - len({r['qid'] for r in rows})} queries have no candidates (empty output)")

    results = build_results(query_ids, rows, params)
    errors = validate_results(results, query_ids)
    if errors:
        sys.exit(f"validation failed: {errors[:5]}")
    js, zbytes = results_to_zip_bytes(results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_bytes(js)
    (args.out / "submission.zip").write_bytes(zbytes)
    (args.out / "selection.json").write_text(json.dumps(params.to_dict(), indent=2))
    n = len(results)
    print(f"{n} queries, avg {sum(len(r['relevant_docs']) for r in results) / n:.1f} docs, "
          f"avg {sum(len(r['relevant_chunks']) for r in results) / n:.1f} chunks -> {args.out / 'submission.zip'}")


if __name__ == "__main__":
    main()
