from __future__ import annotations

import argparse
import asyncio
import collections
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any
from urllib.parse import urlparse

from src.data.vibiomir_contracts import CrawlConfig, CrawlReport
from src.data.vibiomir_registry import SourceRegistry
from src.ingestion.crawl_runner import run_crawl
from src.utils.config import load_config


def run_crawl_cli(
    registry_path: Path | str,
    raw_root: Path | str,
    limit: int | None = 100,
    concurrency: int = 16,
    per_domain_concurrency: int = 2,
    min_domain_interval_seconds: float = 1.0,
    respect_robots: bool = True,
    max_redirects: int = 5,
    max_html_bytes: int = 10485760,
    max_pdf_bytes: int = 31457280,
    output_coverage_file: Path | str | None = None,
    pilot_size: int | None = None,
    seed: int | None = None,
    pilot_sample_path: Path | str | None = None,
) -> CrawlReport:
    """Run ViBioMIR resumable crawler and write coverage report."""
    reg_path = Path(registry_path)
    raw_path = Path(raw_root)

    if not reg_path.exists():
        raise FileNotFoundError(f"Registry DB not found: {reg_path}")

    config = CrawlConfig(
        concurrency=concurrency,
        per_domain_concurrency=per_domain_concurrency,
        min_domain_interval_seconds=min_domain_interval_seconds,
        respect_robots=respect_robots,
        max_redirects=max_redirects,
        max_html_bytes=max_html_bytes,
        max_pdf_bytes=max_pdf_bytes,
        raw_root=raw_path,
    )

    with SourceRegistry(reg_path) as registry:
        target_job_ids: list[str] | None = None
        if pilot_size is not None and pilot_size > 0:
            effective_seed = seed if seed is not None else 42
            sample_file = Path(pilot_sample_path) if pilot_sample_path else raw_path / "pilot_sample_ids.json"
            if sample_file.exists():
                try:
                    with open(sample_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    if meta.get("seed") == effective_seed and meta.get("actual_sample_size") == pilot_size:
                        target_job_ids = meta.get("sampled_job_ids")
                except Exception:
                    target_job_ids = None

            if target_job_ids is None:
                target_job_ids = registry.sample_stratified_jobs(
                    sample_size=pilot_size,
                    seed=effective_seed,
                    persist_path=sample_file,
                )

        report = asyncio.run(
            run_crawl(
                registry,
                config,
                limit=limit if limit is not None else pilot_size,
                job_ids=target_job_ids,
            )
        )

        # Write coverage.json with host and language breakdowns
        state_counts: dict[str, int] = {}
        for row in registry.conn.execute("SELECT state, count(*) FROM fetch_jobs GROUP BY state"):
            state_counts[row[0]] = row[1]

        host_counts: dict[str, dict[str, int]] = {}
        for u, st in registry.conn.execute("SELECT url, state FROM fetch_jobs"):
            h = urlparse(u).netloc.lower().split(":")[0]
            if h not in host_counts:
                host_counts[h] = collections.defaultdict(int)
            host_counts[h][st] += 1

        snapshot_manifest_path = Path("artifacts/vibiomir/snapshots/phase1_A/manifest.json")
        lang_counts: dict[str, int] = {}
        quarantine_counts: dict[str, int] = {}
        ready_docs_count = 0
        if snapshot_manifest_path.exists():
            try:
                with open(snapshot_manifest_path, "r", encoding="utf-8") as f:
                    sm = json.load(f)
                lang_counts = sm.get("language_counts", {})
                quarantine_counts = sm.get("quarantine_counts", {})
                ready_docs_count = sm.get("ready_document_count", 0)
            except Exception:
                pass

        coverage_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "registry_sources": registry.count_sources(),
            "unique_urls": registry.count_unique_urls(),
            "state_counts": state_counts,
            "ready_document_count": ready_docs_count,
            "host_breakdown": {h: dict(stats) for h, stats in sorted(host_counts.items())},
            "language_counts": lang_counts,
            "quarantine_counts": quarantine_counts,
            "crawl_run_report": asdict(report),
        }

        # Write to raw_path / coverage.json
        cov_file = Path(output_coverage_file) if output_coverage_file else raw_path / "coverage.json"
        cov_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cov_file, "w", encoding="utf-8") as f:
            json.dump(coverage_data, f, indent=2, ensure_ascii=False)

        # Also write to artifacts/experiments/vibiomir_phase1_A/coverage.json if exp dir exists
        exp_cov_file = Path("artifacts/experiments/vibiomir_phase1_A/coverage.json")
        if exp_cov_file.parent.exists():
            with open(exp_cov_file, "w", encoding="utf-8") as f:
                json.dump(coverage_data, f, indent=2, ensure_ascii=False)

    return report


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run ViBioMIR resumable web/PDF crawler"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Path to SQLite registry database",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Directory to store compressed raw payloads",
    )
    parser.add_argument(
        "--pilot-size",
        type=int,
        default=None,
        help="Pilot sample size (overrides limit)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of URLs to crawl in this run",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume crawler from current registry state",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Global concurrent workers",
    )
    parser.add_argument(
        "--per-domain-concurrency",
        type=int,
        default=None,
        help="Max concurrency per domain",
    )
    parser.add_argument(
        "--min-domain-interval",
        type=float,
        default=None,
        help="Minimum delay in seconds between requests to same domain",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for pilot sampling",
    )

    parsed = parser.parse_args(args)

    cfg: dict[str, Any] = {}
    config_file = parsed.config
    if config_file is None and Path("configs/vibiomir_phase1.yaml").exists():
        config_file = Path("configs/vibiomir_phase1.yaml")

    if config_file is not None and Path(config_file).exists():
        cfg = load_config(config_file)

    cr_cfg = cfg.get("crawl", {})
    ds_cfg = cfg.get("dataset", {})

    reg_p = parsed.registry_path or Path(ds_cfg.get("registry_path", "artifacts/vibiomir/registry.sqlite"))
    raw_p = parsed.raw_root or Path(cr_cfg.get("raw_root", "artifacts/vibiomir/raw"))

    effective_limit = parsed.pilot_size if parsed.pilot_size is not None else parsed.limit
    if effective_limit is None:
        effective_limit = cr_cfg.get("limit", 100)
    if effective_limit <= 0:
        effective_limit = None

    concurrency = parsed.concurrency or cr_cfg.get("concurrency", 16)
    per_domain = parsed.per_domain_concurrency or cr_cfg.get("per_domain_concurrency", 2)
    interval = parsed.min_domain_interval or cr_cfg.get("min_domain_interval_seconds", 1.0)
    respect_robots = cr_cfg.get("respect_robots", True)

    seed = parsed.seed if parsed.seed is not None else cfg.get("run", {}).get("seed", 42)
    pilot_size = parsed.pilot_size

    print("=== ViBioMIR Crawler Runner ===")
    print(f"Registry DB:     {reg_p}")
    print(f"Raw storage:     {raw_p}")
    print(f"Job limit:       {effective_limit if effective_limit is not None else 'ALL (unlimited)'}")
    if pilot_size is not None:
        print(f"Pilot sample:    {pilot_size} URLs (stratified, seed {seed})")
    print(f"Concurrency:     {concurrency} (per-domain: {per_domain})")
    print(f"Min interval:    {interval}s")
    print(f"Respect robots:  {respect_robots}")

    t0 = time.time()
    report = run_crawl_cli(
        registry_path=reg_p,
        raw_root=raw_p,
        limit=effective_limit,
        concurrency=concurrency,
        per_domain_concurrency=per_domain,
        min_domain_interval_seconds=interval,
        respect_robots=respect_robots,
        pilot_size=pilot_size,
        seed=seed,
    )
    elapsed = time.time() - t0

    print("\n=== Crawl Execution Summary ===")
    print(f"Leased Jobs:      {report.leased}")
    print(f"Successfully Fetched: {report.fetched}")
    print(f"Retry Scheduled:  {report.retry_wait}")
    print(f"Failed:           {report.failed}")
    print(f"Blocked (Robots): {report.blocked}")
    print(f"Bytes Written:    {report.bytes_written:,} bytes ({report.bytes_written / 1024 / 1024:.2f} MiB)")
    print(f"Elapsed Time:     {elapsed:.1f}s")


if __name__ == "__main__":
    main()
