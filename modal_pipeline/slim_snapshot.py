"""Write a copy of a snapshot that keeps only documents in the official corpus.

Shard names, row order and columns are preserved, so prepare_upload.py produces the same
bundle from the slim copy as from the full snapshot. lookup.sqlite is not copied (the
repo's SnapshotSourceStore rebuilds it when missing).

    .venv/Scripts/python.exe modal_pipeline/slim_snapshot.py \
        --snapshot artifacts/vibiomir/snapshots/phase2_full \
        --official-corpus ../hf_check/links_corpus_vi_en_zh.parquet \
        --out artifacts/vibiomir/snapshots/phase2_official
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--official-corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    official = np.unique(pq.read_table(args.official_corpus, columns=["id"])["id"].to_numpy())
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    shards, docs, chunk_rows, langs = [], set(), 0, Counter()
    for kind in ("documents", "chunks"):
        files = sorted(args.snapshot.glob(f"{kind}_*.parquet"))
        for i, src in enumerate(files, 1):
            t = pq.read_table(src)
            t = t.filter(pa.array(np.isin(t["doc_id"].to_numpy(), official)))
            if t.num_rows == 0:
                continue
            dst = args.out / src.name
            pq.write_table(t, dst, compression="zstd", compression_level=6, row_group_size=5000)
            shards.append({"path": src.name, "sha256": _sha256(dst), "type": kind, "rows": t.num_rows,
                           "bytes": dst.stat().st_size})
            if kind == "documents":
                docs.update(t["doc_id"].to_pylist())
                if "language" in t.column_names:
                    langs.update(t["language"].to_pylist())
            else:
                chunk_rows += t.num_rows
            if i % 20 == 0 or i == len(files):
                print(f"  [{time.time() - t0:5.0f}s] {kind}: {i}/{len(files)}", flush=True)

    for extra in ("gaps.json", "source_gaps.parquet"):
        if (args.snapshot / extra).exists():
            shutil.copy2(args.snapshot / extra, args.out / extra)
    src_manifest = args.snapshot / "manifest.json"
    manifest = json.loads(src_manifest.read_text(encoding="utf-8")) if src_manifest.exists() else {}
    manifest.update({
        "derived_from": manifest.get("snapshot_id"),
        "snapshot_id": f"{manifest.get('snapshot_id', 'snapshot')}_official",
        "official_corpus": str(args.official_corpus),
        "official_total": int(len(official)),
        "ready_document_count": len(docs),
        "chunk_count": chunk_rows,
        "language_counts": dict(langs),
        "shard_paths_and_hashes": shards,
    })
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    size = sum(p.stat().st_size for p in args.out.iterdir() if p.is_file())
    print(f"done in {time.time() - t0:.0f}s: {len(docs):,} docs, {chunk_rows:,} chunks, {size / 1e9:.2f} GB -> {args.out}")


if __name__ == "__main__":
    main()
