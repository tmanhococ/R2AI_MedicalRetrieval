from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

from src.data.vibiomir_contracts import (
    ChunkConfig,
    ExtractionConfig,
    SnapshotConfig,
    SnapshotManifest,
)
from src.data.vibiomir_registry import SourceRegistry
from src.ingestion.snapshot import build_snapshot
from src.utils.config import load_config


def run_build_snapshot(
    registry_path: Path | str,
    snapshot_root: Path | str,
    name: str,
    raw_root: Path | str | None = None,
    queries_path: Path | str | None = None,
    extraction_config: ExtractionConfig | None = None,
    chunk_config: ChunkConfig | None = None,
    parquet_row_group_size: int = 5000,
    target_shard_bytes: int = 67108864,
    source_limit: int | None = None,
    resume: bool = False,
) -> SnapshotManifest:
    """Freezes official sources and paragraph/token chunk spans into an immutable snapshot."""
    reg_path = Path(registry_path)
    snap_root = Path(snapshot_root)

    if not reg_path.exists():
        raise FileNotFoundError(f"Registry DB not found: {reg_path}")

    chk_cfg = chunk_config or ChunkConfig()
    snap_cfg = SnapshotConfig(
        snapshot_root=snap_root,
        name=name,
        parquet_row_group_size=parquet_row_group_size,
        target_shard_bytes=target_shard_bytes,
        chunk_config=chk_cfg,
    )

    with SourceRegistry(reg_path) as registry:
        manifest = build_snapshot(
            registry=registry,
            config=snap_cfg,
            raw_root=Path(raw_root) if raw_root else None,
            extraction_config=extraction_config,
            chunk_config=chk_cfg,
            queries_path=Path(queries_path) if queries_path else None,
            source_limit=source_limit,
            resume=resume,
        )

    return manifest


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze crawled sources into an immutable snapshot"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Unique snapshot name (e.g. phase1_A)",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Path to SQLite registry database",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=None,
        help="Root directory where frozen snapshots are stored",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Path to raw payload storage directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of sources to process",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume building from existing finalized shards and lookup db checkpoint",
    )

    parsed = parser.parse_args(args)

    cfg: dict[str, Any] = {}
    config_file = parsed.config
    if config_file is None and Path("configs/vibiomir_phase1.yaml").exists():
        config_file = Path("configs/vibiomir_phase1.yaml")

    if config_file is not None and Path(config_file).exists():
        cfg = load_config(config_file)

    ds_cfg = cfg.get("dataset", {})
    idx_cfg = cfg.get("indexing", {})
    cr_cfg = cfg.get("crawl", {})
    ex_cfg = cfg.get("extraction", {})

    reg_p = parsed.registry_path or Path(ds_cfg.get("registry_path", "artifacts/vibiomir/registry.sqlite"))
    snap_r = parsed.snapshot_root or Path(idx_cfg.get("snapshot_root", "artifacts/vibiomir/snapshots"))
    raw_p = parsed.raw_root or Path(cr_cfg.get("raw_root", "artifacts/vibiomir/raw"))
    queries_p = Path(ds_cfg.get("query_path", "data/official/query.parquet"))

    auto_strat = idx_cfg.get("chunking", {}).get("auto_strategy", {})
    chunk_cfg = ChunkConfig(
        mode=idx_cfg.get("chunking", {}).get("mode", "auto"),
        strategy_type=auto_strat.get("type", "paragraph_token_spans"),
        body_max_tokens=auto_strat.get("body_max_tokens", 320),
        long_block_overlap_tokens=auto_strat.get("long_block_overlap_tokens", 64),
        contextual_header_max_tokens=auto_strat.get("contextual_header_max_tokens", 32),
        drop_short_blocks=auto_strat.get("drop_short_blocks", False),
    )

    extraction_config = ExtractionConfig(
        html_backend=ex_cfg.get("html_backend", "trafilatura"),
        pdf_backend=ex_cfg.get("pdf_backend", "pymupdf"),
        language_model_path=Path(ex_cfg.get("language_model_path", "artifacts/models/lid.176.ftz")),
        language_confidence_min=ex_cfg.get("language_confidence_min", 0.70),
        accepted_languages=tuple(ex_cfg.get("accepted_languages", ["vi", "en", "zh"])),
        quarantine_unknown_language=ex_cfg.get("quarantine_unknown_language", True),
    )

    print("=== ViBioMIR Build Snapshot ===")
    print(f"Snapshot Name:  {parsed.name}")
    print(f"Registry DB:    {reg_p}")
    print(f"Snapshot Root:  {snap_r}")

    t0 = time.time()
    manifest = run_build_snapshot(
        registry_path=reg_p,
        snapshot_root=snap_r,
        name=parsed.name,
        raw_root=raw_p,
        queries_path=queries_p,
        extraction_config=extraction_config,
        chunk_config=chunk_cfg,
        parquet_row_group_size=idx_cfg.get("parquet_row_group_size", 5000),
        target_shard_bytes=idx_cfg.get("target_shard_bytes", 67108864),
        source_limit=parsed.limit,
        resume=parsed.resume,
    )
    elapsed = time.time() - t0

    print("\n=== Snapshot Built Successfully ===")
    print(f"Snapshot ID:     {manifest.snapshot_id}")
    print(f"Ready Documents: {manifest.ready_document_count:,}")
    print(f"Total Chunks:    {manifest.chunk_count:,}")
    print(f"Total Shards:    {len(manifest.shard_paths_and_hashes)}")
    print(f"Membership SHA:  {manifest.source_membership_sha256}")
    print(f"Elapsed Time:    {elapsed:.2f}s")


if __name__ == "__main__":
    main()
