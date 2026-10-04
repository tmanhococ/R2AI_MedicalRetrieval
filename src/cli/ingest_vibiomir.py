from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

from src.data.vibiomir_adapter import iter_official_sources, load_official_queries
from src.data.vibiomir_contracts import IngestReport
from src.data.vibiomir_registry import SourceRegistry, ingest_sources
from src.utils.config import load_config


def run_ingest(
    corpus_path: Path | str,
    query_path: Path | str,
    registry_path: Path | str,
    batch_size: int = 20000,
    expected_query_count: int | None = 1200,
) -> IngestReport:
    """Ingest ViBioMIR official corpus and queries into SQLite registry."""
    corpus_file = Path(corpus_path)
    query_file = Path(query_path)
    registry_file = Path(registry_path)

    if not corpus_file.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    if not query_file.exists():
        raise FileNotFoundError(f"Query file not found: {query_file}")

    # 1. Validate and load queries
    queries = load_official_queries(query_file, expected_count=expected_query_count)

    # 2. Ingest corpus into registry
    with SourceRegistry(registry_file) as registry:
        source_iterator = iter_official_sources(corpus_file, batch_size=batch_size)
        report = ingest_sources(
            source_iterator,
            registry,
            input_sha256=source_iterator.input_sha256,
            batch_size=batch_size,
        )

    return report


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ViBioMIR official corpus and queries into SQLite registry"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=None,
        help="Path to links_corpus.parquet",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        default=None,
        help="Path to query.parquet",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Path to target SQLite registry database",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20000,
        help="Batch size for transactional commits (default: 20000)",
    )
    parser.add_argument(
        "--expected-query-count",
        type=int,
        default=None,
        help="Expected query count (default: 1200)",
    )

    parsed = parser.parse_args(args)

    cfg: dict[str, Any] = {}
    config_file = parsed.config
    if config_file is None and Path("configs/vibiomir_phase1.yaml").exists():
        config_file = Path("configs/vibiomir_phase1.yaml")

    if config_file is not None and Path(config_file).exists():
        cfg = load_config(config_file)

    ds_cfg = cfg.get("dataset", {})
    corpus_p = parsed.corpus_path or Path(ds_cfg.get("source_path", "data/official/links_corpus.parquet"))
    query_p = parsed.query_path or Path(ds_cfg.get("query_path", "data/official/query.parquet"))
    registry_p = parsed.registry_path or Path(ds_cfg.get("registry_path", "artifacts/vibiomir/registry.sqlite"))
    expected_count = parsed.expected_query_count if parsed.expected_query_count is not None else ds_cfg.get("expected_query_count", 1200)

    print(f"=== ViBioMIR Ingestion Pipeline ===")
    print(f"Corpus file:   {corpus_p}")
    print(f"Query file:    {query_p}")
    print(f"Registry DB:   {registry_p}")
    print(f"Commit batch:  {parsed.batch_size}")
    print(f"Expected qcnt: {expected_count}")

    t0 = time.time()
    report = run_ingest(
        corpus_path=corpus_p,
        query_path=query_p,
        registry_path=registry_p,
        batch_size=parsed.batch_size,
        expected_query_count=expected_count,
    )
    total_time = time.time() - t0

    print("\n=== Ingestion Completed Successfully ===")
    print(f"Total Rows:       {report.input_rows:,}")
    print(f"Inserted Sources: {report.inserted_sources:,}")
    print(f"Unique URLs:      {report.unique_urls:,}")
    print(f"Input SHA-256:    {report.input_sha256}")
    print(f"Total Ingest Time:{total_time:.2f}s")


if __name__ == "__main__":
    main()
