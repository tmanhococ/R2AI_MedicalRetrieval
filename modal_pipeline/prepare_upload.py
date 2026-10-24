"""Slim the local phase2_full snapshot into a compact upload bundle for Modal.

Pass 1 hashes every chunk text and counts exact duplicates across the corpus.
Pass 2 drops boilerplate (texts repeated >= --boiler-min-count times: menus,
"most read" sidebars, footers), keeps only the columns needed downstream,
re-shards so no document spans two parts, and writes zstd parquet parts.

The global row id of a chunk = its position across the sorted output parts;
every Modal stage (BM25, embeddings, rerank) keys chunks by that row id.

Usage (from mvp-rag-medical/):
    .venv/Scripts/python.exe modal_pipeline/prepare_upload.py \
        --snapshot artifacts/vibiomir/snapshots/phase2_full \
        --queries data/official/query.parquet \
        --out artifacts/modal_upload
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

CHUNK_COLS = ["doc_id", "chunk_index", "language", "section", "text"]


def _text_hashes(texts: list[str]) -> np.ndarray:
    return np.fromiter(
        (int.from_bytes(hashlib.blake2b(t.encode("utf-8"), digest_size=8).digest(), "little") for t in texts),
        dtype=np.uint64,
        count=len(texts),
    )


def _load_titles(snapshot: Path) -> dict[int, str]:
    titles: dict[int, str] = {}
    for f in sorted(snapshot.glob("documents_*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "title"])
        for d, ti in zip(t["doc_id"].to_pylist(), t["title"].to_pylist()):
            if ti:
                titles[d] = ti
    return titles


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows-per-part", type=int, default=80_000)
    ap.add_argument("--boiler-min-count", type=int, default=5,
                    help="Drop chunk texts that occur at least this many times in the corpus (0 = keep all)")
    ap.add_argument("--limit-shards", type=int, default=0, help="Debug: only read the first N chunk shards")
    ap.add_argument("--official-corpus", type=Path, default=None,
                    help="Organizers' links parquet (id, url); chunks of documents outside it are dropped")
    args = ap.parse_args()
    official = None
    if args.official_corpus:
        official = np.unique(pq.read_table(args.official_corpus, columns=["id"])["id"].to_numpy())

    shards = sorted(args.snapshot.glob("chunks_*.parquet"))
    if args.limit_shards:
        shards = shards[: args.limit_shards]
    if not shards:
        sys.exit(f"No chunks_*.parquet under {args.snapshot}")

    t0 = time.time()
    print(f"Pass 1/2: hashing chunk ids and texts in {len(shards)} shards ...", flush=True)
    shard_hashes: list[np.ndarray] = []
    id_hashes: list[np.ndarray] = []
    shard_official: list[np.ndarray] = []
    for i, shard in enumerate(shards, 1):
        t = pq.read_table(shard, columns=["chunk_id", "doc_id", "text"])
        id_hashes.append(_text_hashes(t["chunk_id"].to_pylist()))
        shard_hashes.append(_text_hashes(t["text"].to_pylist()))
        docs = t["doc_id"].to_numpy()
        shard_official.append(np.isin(docs, official) if official is not None else np.ones(len(docs), dtype=bool))
        if i % 20 == 0 or i == len(shards):
            print(f"  [{time.time() - t0:5.0f}s] hashed {i}/{len(shards)}", flush=True)
    # The snapshot re-wrote some chunks when it was resumed: keep each chunk_id's first copy.
    all_ids = np.concatenate(id_hashes)
    first_copy = np.zeros(len(all_ids), dtype=bool)
    first_copy[np.unique(all_ids, return_index=True)[1]] = True
    bounds = np.cumsum([0] + [len(h) for h in id_hashes])
    shard_first = [first_copy[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    # Boilerplate is judged inside the corpus that will be indexed (official docs only when filtering).
    in_scope = first_copy & np.concatenate(shard_official)
    uniq, counts = np.unique(np.concatenate(shard_hashes)[in_scope], return_counts=True)
    print(f"  {len(all_ids):,} rows, {int(first_copy.sum()):,} unique chunk ids, {len(uniq):,} distinct texts",
          flush=True)
    del all_ids, first_copy, id_hashes

    print("Loading document titles ...", flush=True)
    titles = _load_titles(args.snapshot)
    print(f"  {len(titles):,} titles", flush=True)

    out_chunks = args.out / "chunks"
    if out_chunks.exists():
        shutil.rmtree(out_chunks)
    out_chunks.mkdir(parents=True)

    print("Pass 2/2: filtering boilerplate and writing parts ...", flush=True)
    seen_docs: set[int] = set()
    last_doc: int | None = None
    pending: list[pa.Table] = []
    pending_rows = 0
    part_idx = 0
    stats = {"rows_in": 0, "rows_out": 0, "dropped_duplicate_id": 0, "dropped_off_corpus": 0,
             "dropped_boiler": 0, "dropped_empty": 0, "dropped_noncontiguous": 0}
    lang_counts: dict[str, int] = {}
    kept_docs: set[int] = set()

    def write_part(table: pa.Table) -> None:
        nonlocal part_idx
        title_col = pa.array([titles.get(d) for d in table["doc_id"].to_pylist()], type=pa.string())
        table = table.append_column("title", title_col)
        pq.write_table(
            table,
            out_chunks / f"part_{part_idx:05d}.parquet",
            compression="zstd",
            compression_level=6,
            row_group_size=20_000,
        )
        part_idx += 1

    for i, (shard, hashes, firsts, in_official) in enumerate(zip(shards, shard_hashes, shard_first, shard_official), 1):
        t = pq.read_table(shard, columns=CHUNK_COLS)
        stats["rows_in"] += t.num_rows

        keep = firsts.copy()
        stats["dropped_duplicate_id"] += int((~firsts).sum())
        stats["dropped_off_corpus"] += int((keep & ~in_official).sum())
        keep &= in_official
        if args.boiler_min_count:
            boiler = (counts[np.searchsorted(uniq, hashes)] >= args.boiler_min_count) & keep
            stats["dropped_boiler"] += int(boiler.sum())
            keep &= ~boiler
        empty = ~pc.greater(pc.utf8_length(pc.utf8_trim_whitespace(t["text"])), 0).to_numpy(zero_copy_only=False)
        stats["dropped_empty"] += int((empty & keep).sum())
        keep &= ~empty

        # Documents must be contiguous so that re-sharding never splits one. A doc
        # that reappears after other docs (re-written on resume) keeps its first run.
        docs = t["doc_id"].to_numpy()
        kept_idx = np.flatnonzero(keep)
        if len(kept_idx):
            kd = docs[kept_idx]
            starts = np.flatnonzero(np.r_[True, kd[1:] != kd[:-1]])
            ends = np.r_[starts[1:], len(kd)]
            for j, (s, e) in enumerate(zip(starts, ends)):
                d = int(kd[s])
                if j == 0 and d == last_doc:
                    continue
                if d in seen_docs:
                    keep[kept_idx[s:e]] = False
                    stats["dropped_noncontiguous"] += int(e - s)
                    continue
                seen_docs.add(d)
                last_doc = d
        t = t.filter(pa.array(keep))
        if t.num_rows == 0:
            continue
        kept_docs.update(np.unique(t["doc_id"].to_numpy()).tolist())
        for lang, n in zip(*np.unique(np.asarray(t["language"].to_pylist()), return_counts=True)):
            lang_counts[str(lang)] = lang_counts.get(str(lang), 0) + int(n)

        pending.append(t)
        pending_rows += t.num_rows
        while pending_rows >= args.rows_per_part:
            merged = pa.concat_tables(pending)
            mdocs = merged["doc_id"].to_numpy()
            cut = args.rows_per_part
            while cut < len(mdocs) and mdocs[cut] == mdocs[cut - 1]:
                cut += 1
            if cut >= len(mdocs):
                break  # doc boundary not in the buffer yet; wait for more rows
            write_part(merged.slice(0, cut))
            stats["rows_out"] += cut
            rest = merged.slice(cut)
            pending, pending_rows = [rest], rest.num_rows

        if i % 20 == 0 or i == len(shards):
            print(f"  [{time.time() - t0:5.0f}s] {i}/{len(shards)} shards, {part_idx} parts written", flush=True)

    if pending_rows:
        write_part(pa.concat_tables(pending))
        stats["rows_out"] += pending_rows

    shutil.copy2(args.queries, args.out / "query.parquet")
    manifest = {
        "source_snapshot": str(args.snapshot),
        "boiler_min_count": args.boiler_min_count,
        "official_corpus": str(args.official_corpus) if args.official_corpus else None,
        "official_total": int(len(official)) if official is not None else None,
        "parts": part_idx,
        "docs_out": len(kept_docs),
        "languages_out": lang_counts,
        "columns": CHUNK_COLS + ["title"],
        **stats,
    }
    (args.out / "upload_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(json.dumps(manifest, indent=2))
    print(f"Done in {time.time() - t0:.0f}s, bundle size {size / 1e9:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
