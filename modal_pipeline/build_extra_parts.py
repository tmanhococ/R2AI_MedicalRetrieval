"""Append PubMed records (from fetch_pubmed.py) to an upload bundle as extra parts.

Parts are named part_90000.parquet, ... so they sort after the main parts and the
row ids of the main parts do not move. Schema matches prepare_upload.py output.

    .venv/Scripts/python.exe modal_pipeline/build_extra_parts.py \
        --pubmed artifacts/pubmed --bundle artifacts/modal_upload_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("chunk_index", pa.int64()), ("language", pa.string()),
    ("section", pa.string()), ("text", pa.string()), ("title", pa.string()),
])


def pubmed_chunks(rec: dict, max_words: int = 300) -> list[dict]:
    """Abstract sections grouped greedily into chunks of at most max_words words.

    A section longer than max_words is split by words. Records without an abstract
    become one title chunk so the document stays retrievable; missing records are skipped.
    """
    title = rec.get("title") or ""
    if rec.get("missing") or not (title or rec.get("sections")):
        return []
    paras: list[tuple[str | None, list[str]]] = []
    for label, text in rec.get("sections") or []:
        words = text.split()
        for i in range(0, len(words), max_words):
            paras.append((label, words[i : i + max_words]))
    if not paras:
        paras = [(None, title.split())]

    groups: list[list[tuple[str | None, list[str]]]] = []
    size = 0
    for p in paras:
        if groups and size + len(p[1]) <= max_words:
            groups[-1].append(p)
            size += len(p[1])
        else:
            groups.append([p])
            size = len(p[1])
    return [
        {"doc_id": int(rec["doc_id"]), "chunk_index": i, "language": "en", "section": g[0][0],
         "text": "\n\n".join(" ".join(words) for _, words in g), "title": title or None}
        for i, g in enumerate(groups)
    ]


def write_pubmed_parts(src: Path, out_chunks: Path, first_index: int = 90000, rows_per_part: int = 80_000,
                       max_words: int = 300) -> dict:
    recs = []
    for path in sorted(src.glob("batch_*.jsonl")):
        recs += [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    recs.sort(key=lambda r: int(r["doc_id"]))
    out_chunks.mkdir(parents=True, exist_ok=True)
    for old in out_chunks.glob("part_9*.parquet"):
        old.unlink()

    stats = {"records": len(recs), "docs": 0, "chunks": 0, "skipped_missing": 0}
    buf: list[dict] = []
    part = first_index

    def flush() -> None:
        nonlocal buf, part
        pq.write_table(pa.Table.from_pylist(buf, schema=SCHEMA), out_chunks / f"part_{part:05d}.parquet",
                       compression="zstd", compression_level=6, row_group_size=20_000)
        part += 1
        buf = []

    for rec in recs:
        rows = pubmed_chunks(rec, max_words)
        if not rows:
            stats["skipped_missing"] += 1
            continue
        buf += rows
        stats["docs"] += 1
        stats["chunks"] += len(rows)
        if len(buf) >= rows_per_part:  # cut only between documents
            flush()
    if buf:
        flush()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pubmed", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--max-words", type=int, default=300)
    args = ap.parse_args()
    stats = write_pubmed_parts(args.pubmed, args.bundle / "chunks", max_words=args.max_words)
    manifest_path = args.bundle / "upload_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest["pubmed"] = stats
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
