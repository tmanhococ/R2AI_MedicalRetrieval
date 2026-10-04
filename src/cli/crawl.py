import argparse
import asyncio
from pathlib import Path
import sys
import time

from src.data.vibiomir_contracts import CrawlConfig
from src.data.vibiomir_registry import SourceRegistry
from src.ingestion.crawl_runner import run_crawl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ViBioMIR resumable web/PDF crawler"
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("artifacts/vibiomir/registry.sqlite"),
        help="Path to SQLite registry database",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("artifacts/vibiomir/raw"),
        help="Directory to store compressed raw payloads",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of URLs to crawl in this run (default: 100 for pilot, set -1 for all)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=16,
        help="Global concurrent workers (default: 16)",
    )
    parser.add_argument(
        "--per-domain-concurrency",
        type=int,
        default=2,
        help="Max concurrency per domain (default: 2)",
    )
    parser.add_argument(
        "--min-domain-interval",
        type=float,
        default=1.0,
        help="Minimum delay in seconds between requests to same domain (default: 1.0)",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Ignore robots.txt (not recommended)",
    )

    args = parser.parse_args()

    limit = None if args.limit <= 0 else args.limit

    print("=== ViBioMIR Crawler Runner ===")
    print(f"Registry DB:     {args.registry_path}")
    print(f"Raw storage:     {args.raw_root}")
    print(f"Job limit:       {limit if limit is not None else 'ALL (unlimited)'}")
    print(f"Concurrency:     {args.concurrency} (per-domain: {args.per_domain_concurrency})")
    print(f"Min interval:    {args.min_domain_interval}s")
    print(f"Respect robots:  {not args.ignore_robots}")

    if not args.registry_path.exists():
        print(f"Error: Registry DB not found: {args.registry_path}", file=sys.stderr)
        print("Please run 'python -m src.cli.ingest_vibiomir' first.", file=sys.stderr)
        sys.exit(1)

    config = CrawlConfig(
        concurrency=args.concurrency,
        per_domain_concurrency=args.per_domain_concurrency,
        min_domain_interval_seconds=args.min_domain_interval,
        respect_robots=not args.ignore_robots,
        raw_root=args.raw_root,
    )

    t0 = time.time()
    with SourceRegistry(args.registry_path) as registry:
        if not registry.is_complete():
            print(
                "Warning: Registry ingest is not marked complete. Running on available sources...",
                file=sys.stderr,
            )

        print("\nStarting crawl loop...")
        report = asyncio.run(run_crawl(registry, config, limit=limit))

    elapsed = time.time() - t0
    print("\n=== Crawl Execution Summary ===")
    print(f"Leased Jobs:      {report.leased}")
    print(f"Successfully Fetched: {report.fetched}")
    print(f"Retry Scheduled:  {report.retry_wait}")
    print(f"Failed:           {report.failed}")
    print(f"Blocked (Robots): {report.blocked}")
    print(f"Unsupported:      {report.unsupported}")
    print(f"Bytes Written:    {report.bytes_written:,} bytes ({report.bytes_written / 1024 / 1024:.2f} MiB)")
    print(f"Elapsed Time:     {elapsed:.1f}s")


if __name__ == "__main__":
    main()
