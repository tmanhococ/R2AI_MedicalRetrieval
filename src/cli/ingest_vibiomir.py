import argparse
import sys
import time
from pathlib import Path

from src.data.vibiomir_adapter import iter_official_sources, load_official_queries
from src.data.vibiomir_registry import SourceRegistry, ingest_sources


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ViBioMIR official corpus and queries into SQLite registry"
    )
    parser.add_argument(
        "--corpus-path",
        type=Path,
        default=Path("data/official/links_corpus.parquet"),
        help="Path to links_corpus.parquet",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        default=Path("data/official/query.parquet"),
        help="Path to query.parquet",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("artifacts/vibiomir/registry.sqlite"),
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
        default=1200,
        help="Expected query count (default: 1200)",
    )

    args = parser.parse_args()

    print(f"=== ViBioMIR Ingestion Pipeline ===")
    print(f"Corpus file:   {args.corpus_path}")
    print(f"Query file:    {args.query_path}")
    print(f"Registry DB:   {args.registry_path}")
    print(f"Commit batch:  {args.batch_size}")

    if not args.corpus_path.exists():
        print(f"Error: Corpus file not found: {args.corpus_path}", file=sys.stderr)
        sys.exit(1)

    if not args.query_path.exists():
        print(f"Error: Query file not found: {args.query_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Validate and load queries
    print("\n[1/2] Validating official queries...")
    t0 = time.time()
    queries = load_official_queries(
        args.query_path, expected_count=args.expected_query_count
    )
    print(
        f"  Successfully validated {len(queries)} official queries in {time.time() - t0:.2f}s"
    )

    # 2. Ingest corpus into registry
    print("\n[2/2] Ingesting official corpus sources into SQLite registry...")
    t0 = time.time()
    with SourceRegistry(args.registry_path) as registry:
        # Wrapper around iter_official_sources to log progress periodically
        source_iterator = iter_official_sources(args.corpus_path, batch_size=args.batch_size)
        
        def progress_stream():
            count = 0
            last_log = time.time()
            for item in source_iterator:
                count += 1
                if count % 200000 == 0:
                    now = time.time()
                    elapsed = now - t0
                    rate = count / elapsed if elapsed > 0 else 0
                    print(
                        f"  Processed {count:,} rows ({rate:.0f} rows/s, elapsed: {elapsed:.1f}s)..."
                    )
                yield item

        report = ingest_sources(
            progress_stream(),
            registry,
            input_sha256=source_iterator.input_sha256,
            batch_size=args.batch_size,
        )

    total_time = time.time() - t0
    print("\n=== Ingestion Completed Successfully ===")
    print(f"Total Rows:       {report.input_rows:,}")
    print(f"Inserted Sources: {report.inserted_sources:,}")
    print(f"Unique URLs:      {report.unique_urls:,}")
    print(f"Input SHA-256:    {report.input_sha256}")
    print(f"Total Ingest Time:{total_time:.2f}s ({report.input_rows / total_time:.0f} rows/s)")


if __name__ == "__main__":
    main()
