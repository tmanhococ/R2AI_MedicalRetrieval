from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from typing import Any

import pyarrow.parquet as pq

from src.data.phase2_contracts import IndexManifest
from src.indexing.lexical_tokens import tokenize_and_encode
from src.indexing.lucene import LexicalRecord, LuceneSparseBackend
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def _load_snapshot_documents(snapshot_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Loads documents from snapshot Parquet shards or fallback JSONL, grouped by shard."""
    shards: dict[str, list[dict[str, Any]]] = defaultdict(list)

    parquet_files = sorted(snapshot_path.glob("documents_*.parquet"))
    if parquet_files:
        for pf in parquet_files:
            shard_name = pf.stem
            table = pq.read_table(pf)
            dicts = table.to_pylist()
            shards[shard_name].extend(dicts)
        return shards

    jsonl_file = snapshot_path / "documents.jsonl"
    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    shards["shard_0"].append(item)
        return shards

    return shards


def _load_snapshot_chunks(snapshot_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Loads chunks from snapshot Parquet shards or fallback JSONL, grouped by shard."""
    shards: dict[str, list[dict[str, Any]]] = defaultdict(list)

    parquet_files = sorted(snapshot_path.glob("chunks_*.parquet"))
    if parquet_files:
        for pf in parquet_files:
            shard_name = pf.stem
            table = pq.read_table(pf)
            dicts = table.to_pylist()
            shards[shard_name].extend(dicts)
        return shards

    jsonl_file = snapshot_path / "chunks.jsonl"
    if jsonl_file.exists():
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    shards["shard_0"].append(item)
        return shards

    return shards


def run_index_sparse(
    snapshot_dir: Path | str,
    output_dir: Path | str,
    config: dict[str, Any] | None = None,
    resume: bool = False,
) -> dict[str, IndexManifest]:
    """Builds disk-backed partitioned Lucene sparse indexes and assembles logical indexes.

    Args:
        snapshot_dir: Path to frozen snapshot directory.
        output_dir: Target output directory for partitions and logical indexes.
        config: Optional configuration dictionary.
        resume: If True, resumes existing completed partition and assembled index checkpoints.

    Returns:
        Dictionary mapping partition name to IndexManifest.
    """
    snap_p = Path(snapshot_dir)
    out_p = Path(output_dir)
    parts_dir = out_p / "partitions"
    parts_dir.mkdir(parents=True, exist_ok=True)

    if not snap_p.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snap_p}")

    # Read snapshot manifest if present
    snapshot_id = "phase2_full"
    snap_manifest_file = snap_p / "manifest.json"
    if snap_manifest_file.exists():
        try:
            with open(snap_manifest_file, "r", encoding="utf-8") as f:
                s_meta = json.load(f)
                snapshot_id = s_meta.get("snapshot_id", snapshot_id)
        except Exception:
            pass

    retrieval_cfg = config.get("retrieval", {}) if config else {}
    sparse_cfg = retrieval_cfg.get("sparse", {})
    k1 = float(sparse_cfg.get("k1", 1.2))
    b = float(sparse_cfg.get("b", 0.75))

    accepted_langs = ["vi", "en", "zh", "und"]
    if config:
        ds_langs = config.get("dataset", {}).get("accepted_languages")
        if ds_langs and isinstance(ds_langs, (list, tuple)):
            accepted_langs = list(ds_langs)
            if "und" not in accepted_langs:
                accepted_langs.append("und")

    # 1. Build document partitions
    doc_shards = _load_snapshot_documents(snap_p)
    logger.info("Found %d document shards to process", len(doc_shards))

    for shard_name, docs in doc_shards.items():
        by_lang: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for d in docs:
            raw_lang = str(d.get("language") or "und").strip().lower()
            norm_lang = raw_lang if raw_lang in accepted_langs else "und"
            by_lang[norm_lang].append(d)

        for lang, lang_docs in by_lang.items():
            part_out = parts_dir / f"doc_{shard_name}_{lang}"
            stage_file = part_out / "stage_manifest.json"
            if resume and stage_file.exists() and (part_out / "partition.db").exists():
                logger.info("Resuming partition %s (already complete)", part_out.name)
                continue

            rows = [
                LexicalRecord(
                    external_id=str(d["doc_id"]),
                    official_doc_id=int(d["doc_id"]),
                    language=lang,
                    item_type="document",
                    encoded_contents=tokenize_and_encode(
                        f"{d.get('title') or ''} {d.get('source_text') or d.get('text') or ''}".strip(),
                        language=lang,
                    ),
                )
                for d in lang_docs
            ]
            LuceneSparseBackend.build_partition(rows, part_out, snapshot_id=snapshot_id)

    # 2. Build chunk partitions
    chunk_shards = _load_snapshot_chunks(snap_p)
    logger.info("Found %d chunk shards to process", len(chunk_shards))

    for shard_name, chunks in chunk_shards.items():
        by_lang = defaultdict(list)
        for c in chunks:
            raw_lang = str(c.get("language") or "und").strip().lower()
            norm_lang = raw_lang if raw_lang in accepted_langs else "und"
            by_lang[norm_lang].append(c)

        for lang, lang_chunks in by_lang.items():
            part_out = parts_dir / f"chunk_{shard_name}_{lang}"
            stage_file = part_out / "stage_manifest.json"
            if resume and stage_file.exists() and (part_out / "partition.db").exists():
                logger.info("Resuming partition %s (already complete)", part_out.name)
                continue

            rows = [
                LexicalRecord(
                    external_id=str(c["chunk_id"]),
                    official_doc_id=int(c["doc_id"]),
                    language=lang,
                    item_type="chunk",
                    encoded_contents=tokenize_and_encode(
                        f"{c.get('section') or ''} {c.get('text') or ''}".strip(),
                        language=lang,
                    ),
                )
                for c in lang_chunks
            ]
            LuceneSparseBackend.build_partition(rows, part_out, snapshot_id=snapshot_id)

    # 3. Assemble logical indexes per language and item_type
    assembled_manifests: dict[str, IndexManifest] = {}
    top_manifest: dict[str, Any] = {
        "backend": "lucene",
        "snapshot_id": snapshot_id,
        "parameters": {
            "k1": k1,
            "b": b,
            "analyzer": "whitespace",
            "token_encoding": "sha256_hex_v1",
        },
        "partitions": {},
    }

    all_part_dirs = [p for p in parts_dir.iterdir() if p.is_dir()]

    for lang in accepted_langs:
        # Assemble documents
        doc_part_paths = [
            p for p in all_part_dirs
            if p.name.startswith("doc_") and p.name.endswith(f"_{lang}")
        ]
        doc_out = out_p / f"{lang}_documents"
        doc_manifest_file = doc_out / "manifest.json"

        if resume and doc_manifest_file.exists() and (doc_out / "index.db").exists():
            with open(doc_manifest_file, "r", encoding="utf-8") as f:
                d_meta = json.load(f)
                d_manifest = IndexManifest(**d_meta)
        else:
            d_manifest = LuceneSparseBackend.assemble(
                partitions=doc_part_paths,
                output=doc_out,
                snapshot_id=snapshot_id,
                language=lang,
                item_type="document",
                k1=k1,
                b=b,
            )

        assembled_manifests[f"{lang}_documents"] = d_manifest
        top_manifest["partitions"][f"{lang}_documents"] = {
            "count": d_manifest.member_count,
            "item_type": "document",
            "language": lang,
            "membership_sha256": d_manifest.membership_sha256,
        }

        # Assemble chunks
        chunk_part_paths = [
            p for p in all_part_dirs
            if p.name.startswith("chunk_") and p.name.endswith(f"_{lang}")
        ]
        chunk_out = out_p / f"{lang}_chunks"
        chunk_manifest_file = chunk_out / "manifest.json"

        if resume and chunk_manifest_file.exists() and (chunk_out / "index.db").exists():
            with open(chunk_manifest_file, "r", encoding="utf-8") as f:
                c_meta = json.load(f)
                c_manifest = IndexManifest(**c_meta)
        else:
            c_manifest = LuceneSparseBackend.assemble(
                partitions=chunk_part_paths,
                output=chunk_out,
                snapshot_id=snapshot_id,
                language=lang,
                item_type="chunk",
                k1=k1,
                b=b,
            )

        assembled_manifests[f"{lang}_chunks"] = c_manifest
        top_manifest["partitions"][f"{lang}_chunks"] = {
            "count": c_manifest.member_count,
            "item_type": "chunk",
            "language": lang,
            "membership_sha256": c_manifest.membership_sha256,
        }

    # Write root manifest.json
    root_manifest_path = out_p / "manifest.json"
    with open(root_manifest_path, "w", encoding="utf-8") as f:
        json.dump(top_manifest, f, indent=2, ensure_ascii=False)

    logger.info("Successfully assembled %d logical indexes in %s", len(assembled_manifests), out_p)
    return assembled_manifests


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build disk-backed partitioned Lucene sparse indexes for ViBioMIR Phase 2"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to frozen snapshot directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write partitions and assembled logical indexes",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume completed partition and assembled index checkpoints",
    )

    parsed = parser.parse_args(args)

    cfg: dict[str, Any] = {}
    config_file = parsed.config
    if config_file is None:
        if Path("configs/vibiomir_phase2_B1.yaml").exists():
            config_file = Path("configs/vibiomir_phase2_B1.yaml")
        elif Path("configs/vibiomir_phase2.yaml").exists():
            config_file = Path("configs/vibiomir_phase2.yaml")

    if config_file is not None and Path(config_file).exists():
        cfg = load_config(config_file)

    out_dir = parsed.output_dir
    if out_dir is None:
        out_base = cfg.get("run", {}).get("output_dir", "artifacts/indexes")
        out_dir = Path(out_base) / "sparse"

    print("=== ViBioMIR Full-Corpus Lucene Sparse Indexing ===")
    print(f"Snapshot:   {parsed.snapshot}")
    print(f"Output dir: {out_dir}")
    print(f"Resume:     {parsed.resume}")

    manifests = run_index_sparse(
        snapshot_dir=parsed.snapshot,
        output_dir=out_dir,
        config=cfg,
        resume=parsed.resume,
    )

    print("\n=== Assembled Index Summary ===")
    for name, m in manifests.items():
        print(f"  {name:20s}: {m.member_count:,} members (sha256: {m.membership_sha256[:10]}...)")
    print("Done!")


if __name__ == "__main__":
    main()
