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
from vb_common import (  # noqa: E402
    SelectionParams,
    build_results,
    docs_as_chunks,
    results_to_zip_bytes,
    validate_results,
)


def _doc_chunks(bundle: Path, docs: set[int]) -> dict[int, list[tuple[int, str]]]:
    """doc_id -> [(row, text)] in row order for the given docs; row ids follow app.build_meta (sorted parts)."""
    import numpy as np

    out: dict[int, list[tuple[int, str]]] = {}
    wanted = np.array(sorted(docs), dtype=np.int64)
    offset = 0
    for part in sorted((bundle / "chunks").glob("part_*.parquet")):
        t = pq.read_table(part, columns=["doc_id", "text"])
        ids = t["doc_id"].to_numpy()
        for i in np.flatnonzero(np.isin(ids, wanted)).tolist():
            out.setdefault(int(ids[i]), []).append((offset + i, t["text"][i].as_py()))
        offset += t.num_rows
    return out


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
    ap.add_argument("--chunk-mode", choices=["pool", "doc"], default="pool",
                    help="pool: reranked chunks as is; doc: one prediction per selected doc (document window)")
    ap.add_argument("--doc-chars", type=int, default=0, help="doc mode: window size in chars (0 = whole document)")
    ap.add_argument("--bundle", type=Path, default=Path("artifacts/modal_upload_v2"),
                    help="doc mode: bundle whose chunks/ parts rebuild document texts")
    args = ap.parse_args()

    params = SelectionParams(**{f.name: getattr(args, f.name) for f in fields(SelectionParams)})
    rows = pq.read_table(args.pool, columns=["qid", "row", "doc_id", "ce", "doc_rrf", "text"]).to_pylist()
    query_ids = pq.read_table(args.queries, columns=["id"])["id"].to_pylist()
    if len({r["qid"] for r in rows} - set(query_ids)):
        sys.exit("pool contains query ids not in the query file")
    if len(query_ids) != len({r["qid"] for r in rows}):
        print(f"note: {len(query_ids) - len({r['qid'] for r in rows})} queries have no candidates (empty output)")

    results = build_results(query_ids, rows, params)
    if args.chunk_mode == "doc":
        best: dict[tuple[int, int], tuple[float, int]] = {}
        for r in rows:
            key = (int(r["qid"]), int(r["doc_id"]))
            if key not in best or r["ce"] > best[key][0]:
                best[key] = (r["ce"], int(r["row"]))
        docs = {d for r in results for d in r["relevant_docs"]}
        doc_chunks = _doc_chunks(args.bundle, docs)
        results = docs_as_chunks(results, {k: v[1] for k, v in best.items()}, doc_chunks, args.doc_chars)
    errors = validate_results(results, query_ids)
    if errors:
        sys.exit(f"validation failed: {errors[:5]}")
    js, zbytes = results_to_zip_bytes(results)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_bytes(js)
    (args.out / "submission.zip").write_bytes(zbytes)
    selection = {**params.to_dict(), "chunk_mode": args.chunk_mode, "doc_chars": args.doc_chars}
    (args.out / "selection.json").write_text(json.dumps(selection, indent=2))
    n = len(results)
    print(f"{n} queries, avg {sum(len(r['relevant_docs']) for r in results) / n:.1f} docs, "
          f"avg {sum(len(r['relevant_chunks']) for r in results) / n:.1f} chunks -> {args.out / 'submission.zip'}")


if __name__ == "__main__":
    main()
