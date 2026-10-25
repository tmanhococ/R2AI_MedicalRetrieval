"""Delete crawl data of documents outside the organizers' official corpus.

Prunes, under <root>/artifacts/vibiomir:
- raw/xx/*.gz payloads not referenced by an official document (a payload shared with an
  official document is kept),
- registry.sqlite rows (sources, then fetch_jobs no longer referenced), followed by VACUUM,
- failed_urls.jsonl lines of jobs that no official document uses,
- snapshots/*/source_gaps.parquet rows and the matching gaps.json counts.

Dry run by default (prints what would go); pass --apply to delete.

    .venv/Scripts/python.exe modal_pipeline/prune_to_official.py --official-corpus data/official/links_corpus_vi_en_zh.parquet
    .venv/Scripts/python.exe modal_pipeline/prune_to_official.py --official-corpus data/official/links_corpus_vi_en_zh.parquet --apply
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sqlite3
import time

import pyarrow as pa
import pyarrow.parquet as pq


def _basename(raw_path: str) -> str:
    return raw_path.replace("\\", "/").rsplit("/", 1)[-1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--official-corpus", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    art = args.root / "artifacts" / "vibiomir"
    official = set(pq.read_table(args.official_corpus, columns=["id"])["id"].to_pylist())
    mode = "APPLY" if args.apply else "DRY RUN"
    t0 = time.time()
    print(f"[{mode}] official corpus: {len(official):,} ids", flush=True)

    db = sqlite3.connect(art / "registry.sqlite")
    sources = db.execute("select doc_id, fetch_job_id from sources").fetchall()
    keep_jobs = {job for doc, job in sources if doc in official}
    keep_raw = {_basename(p) for job, p in db.execute("select job_id, raw_path from fetch_jobs where raw_path is not null")
                if job in keep_jobs}
    n_jobs = db.execute("select count(*) from fetch_jobs").fetchone()[0]
    print(f"registry: sources {len(sources):,} -> {sum(d in official for d, _ in sources):,}; "
          f"fetch_jobs {n_jobs:,} -> {len(keep_jobs):,}; raw payloads kept {len(keep_raw):,}", flush=True)

    drop_files = drop_bytes = kept_files = 0
    for sub in sorted((art / "raw").iterdir()):
        if not sub.is_dir():
            continue
        with os.scandir(sub) as it:
            for entry in it:
                if entry.name in keep_raw:
                    kept_files += 1
                    continue
                drop_files += 1
                drop_bytes += entry.stat().st_size
                if args.apply:
                    os.remove(entry.path)
    print(f"raw: keep {kept_files:,} files, {'deleted' if args.apply else 'would delete'} {drop_files:,} files "
          f"({drop_bytes / 1e9:.1f} GB) [{time.time() - t0:.0f}s]", flush=True)

    failed = art / "failed_urls.jsonl"
    if failed.exists():
        lines = failed.read_text(encoding="utf-8").splitlines()
        kept = [x for x in lines if x.strip() and json.loads(x).get("job_id") in keep_jobs]
        print(f"failed_urls.jsonl: {len(lines):,} -> {len(kept):,} lines", flush=True)
        if args.apply:
            tmp = failed.with_suffix(".tmp")
            tmp.write_text("".join(x + "\n" for x in kept), encoding="utf-8")
            os.replace(tmp, failed)

    for gaps_path in sorted((art / "snapshots").glob("*/source_gaps.parquet")):
        t = pq.read_table(gaps_path)
        mask = pa.array([d in official for d in t["doc_id"].to_pylist()])
        kept_t = t.filter(mask)
        print(f"{gaps_path.parent.name}/source_gaps.parquet: {t.num_rows:,} -> {kept_t.num_rows:,} rows", flush=True)
        if args.apply:
            pq.write_table(kept_t, gaps_path)
            gj = gaps_path.parent / "gaps.json"
            if gj.exists():
                g = json.loads(gj.read_text(encoding="utf-8"))
                reasons = Counter(kept_t["reason"].to_pylist()) if "reason" in kept_t.column_names else Counter()
                g.update({"total_gaps": kept_t.num_rows, "terminal_gap_count": kept_t.num_rows,
                          "gap_counts_by_reason": dict(reasons), "pruned_to_official_corpus": True})
                gj.write_text(json.dumps(g, indent=2), encoding="utf-8")

    if args.apply:
        db.execute("create temp table keep_docs (doc_id integer primary key)")
        db.executemany("insert into keep_docs values (?)", ((d,) for d in official))
        db.execute("delete from sources where doc_id not in (select doc_id from keep_docs)")
        db.execute("delete from fetch_jobs where job_id not in (select fetch_job_id from sources)")
        db.commit()
        db.execute("vacuum")
        print(f"registry pruned and vacuumed [{time.time() - t0:.0f}s]", flush=True)
    db.close()
    print(f"[{mode}] done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
