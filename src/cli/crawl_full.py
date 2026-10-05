from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import signal
import sys
import time
from typing import Any

import httpx

from src.data.phase2_contracts import (
    FullCampaignConfig,
    SCOPE_FULL_SOURCE_PASS_WITH_GAPS,
    SCOPE_PARTIAL_BODY_INDEX,
)
from src.data.vibiomir_contracts import CrawlConfig, CrawlReport
from src.data.vibiomir_registry import SourceRegistry
from src.ingestion.full_campaign import (
    import_content_cache,
    plan_full_campaign,
    run_full_campaign_crawl,
)
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def run_full_crawl_cli(
    registry_path: Path | str,
    raw_root: Path | str,
    config: CrawlConfig | None = None,
    full_campaign_config: FullCampaignConfig | None = None,
    scope: str = "all",
    resume: bool = False,
    limit: int | None = None,
    cache_dir: Path | str | None = None,
    output_coverage_file: Path | str | None = None,
    test_client: httpx.AsyncClient | None = None,
) -> CrawlReport:
    """Full-corpus acquisition campaign runner with resume and content cache import."""
    reg_path = Path(registry_path)
    raw_path = Path(raw_root)

    if not reg_path.exists():
        raise FileNotFoundError(f"Registry DB not found: {reg_path}")

    raw_path.mkdir(parents=True, exist_ok=True)

    crawl_cfg = config or CrawlConfig(raw_root=raw_path)

    with SourceRegistry(reg_path) as registry:
        # 1. Content cache import if cache_dir is provided
        if cache_dir is not None:
            c_path = Path(cache_dir)
            if c_path.exists():
                imported = import_content_cache(registry=registry, cache_dir=c_path, raw_dir=raw_path)
                logger.info(f"Imported {imported} items from content cache at {c_path}")

        # 2. Plan or resume full acquisition campaign
        campaign_id: str | None = None
        if resume:
            row = registry.conn.execute(
                """
                SELECT campaign_id, scope, config_json
                FROM campaigns
                ORDER BY created_at DESC LIMIT 1
                """
            ).fetchone()
            if row:
                campaign_id = row[0]
                logger.info(f"Resuming existing campaign: {campaign_id}")

        camp_cfg = full_campaign_config or FullCampaignConfig(
            scope=scope,
            source_limit=limit,
            reuse_existing_content=True,
            priority_changes_order_only=True,
        )

        campaign_id = plan_full_campaign(
            registry=registry,
            config=camp_cfg,
            campaign_id=campaign_id,
            recover_all_inflight=True,
        )
        logger.info(f"Active acquisition campaign: {campaign_id} (scope={camp_cfg.scope})")

        # 3. Run crawl loop until no pending / retry_wait remain (or limit reached)
        report: CrawlReport | None = None
        interrupted = False

        try:
            report = asyncio.run(
                run_full_campaign_crawl(
                    registry=registry,
                    campaign_id=campaign_id,
                    config=crawl_cfg,
                    limit=limit,
                    client=test_client,
                    wait_for_retry=True,
                )
            )
        except KeyboardInterrupt:
            interrupted = True
            logger.warning(
                f"\nCrawling interrupted by user. Campaign {campaign_id} state preserved. "
                f"Resume with: python -m src.cli.crawl_full --resume"
            )
            # Query actual progress from registry
            counts = {
                r[0]: r[1]
                for r in registry.conn.execute(
                    "SELECT state, count(*) FROM fetch_jobs GROUP BY state"
                ).fetchall()
            }
            report = CrawlReport(
                leased=counts.get("fetching", 0) + counts.get("leased", 0),
                fetched=counts.get("fetched", 0),
                retry_wait=counts.get("retry_wait", 0),
                failed=counts.get("fetch_failed", 0),
                blocked=counts.get("blocked", 0),
                unsupported=counts.get("unsupported", 0),
                bytes_written=0,
            )

        # 4. Record status in campaigns table
        with registry.conn:
            remaining_pending = registry.conn.execute(
                "SELECT count(*) FROM fetch_jobs WHERE state IN ('pending', 'retry_wait', 'fetching', 'leased')"
            ).fetchone()[0]
            status_str = "interrupted" if interrupted else ("completed" if remaining_pending == 0 else "partial")
            registry.conn.execute(
                "UPDATE campaigns SET status = ? WHERE campaign_id = ?",
                (status_str, campaign_id),
            )

        # 5. Output coverage summary report if path provided
        if output_coverage_file is not None:
            cov_path = Path(output_coverage_file)
            cov_path.parent.mkdir(parents=True, exist_ok=True)
            stats: dict[str, int] = {}
            for row in registry.conn.execute("SELECT state, count(*) FROM fetch_jobs GROUP BY state"):
                stats[row[0]] = row[1]
            cov_data = {
                "campaign_id": campaign_id,
                "scope": SCOPE_PARTIAL_BODY_INDEX if limit is not None else SCOPE_FULL_SOURCE_PASS_WITH_GAPS,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "state_counts": stats,
                "report": asdict(report) if report else {},
            }
            with open(cov_path, "w", encoding="utf-8") as f:
                json.dump(cov_data, f, indent=2)

        return report or CrawlReport(0, 0, 0, 0, 0, 0, 0)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="ViBioMIR full acquisition campaign runner with resume and content cache import"
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
        help="Path to raw payload storage directory",
    )
    parser.add_argument(
        "--scope",
        type=str,
        default="all",
        help="Acquisition scope (all | partial)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume campaign without resetting progress",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional fetch limit (marks campaign scope as PARTIAL)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional path to directory containing pre-existing content cache to import",
    )
    parser.add_argument(
        "--output-coverage",
        type=Path,
        default=None,
        help="Path to write acquisition coverage summary JSON",
    )

    parsed = parser.parse_args(args)

    cfg: dict[str, Any] = {}
    config_file = parsed.config
    if config_file is None and Path("configs/vibiomir_phase1.yaml").exists():
        config_file = Path("configs/vibiomir_phase1.yaml")

    if config_file and config_file.exists():
        cfg = load_config(config_file)

    reg_path = (
        parsed.registry_path
        or (Path(cfg.get("dataset", {}).get("registry_path")) if "dataset" in cfg else None)
        or Path("artifacts/vibiomir/registry.sqlite")
    )
    raw_root = (
        parsed.raw_root
        or (Path(cfg.get("crawl", {}).get("raw_root")) if "crawl" in cfg else None)
        or Path("artifacts/vibiomir/raw")
    )

    crawl_cfg_data = cfg.get("crawl", {})
    crawl_cfg = CrawlConfig(
        concurrency=int(crawl_cfg_data.get("concurrency", 16)),
        per_domain_concurrency=int(crawl_cfg_data.get("per_domain_concurrency", 2)),
        min_domain_interval_seconds=float(crawl_cfg_data.get("min_domain_interval_seconds", 1.0)),
        connect_timeout_seconds=float(crawl_cfg_data.get("connect_timeout_seconds", 10.0)),
        read_timeout_seconds=float(crawl_cfg_data.get("read_timeout_seconds", 30.0)),
        total_deadline_seconds=float(crawl_cfg_data.get("total_deadline_seconds", 60.0)),
        max_attempts=int(crawl_cfg_data.get("max_attempts", 3)),
        base_backoff_seconds=float(crawl_cfg_data.get("base_backoff_seconds", 2.0)),
        lease_seconds=float(crawl_cfg_data.get("lease_seconds", 120.0)),
        max_redirects=int(crawl_cfg_data.get("max_redirects", 5)),
        max_html_bytes=int(crawl_cfg_data.get("max_html_bytes", 10485760)),
        max_pdf_bytes=int(crawl_cfg_data.get("max_pdf_bytes", 31457280)),
        respect_robots=bool(crawl_cfg_data.get("respect_robots", True)),
        raw_root=Path(raw_root),
    )

    report = run_full_crawl_cli(
        registry_path=reg_path,
        raw_root=raw_root,
        config=crawl_cfg,
        scope=parsed.scope,
        resume=parsed.resume,
        limit=parsed.limit,
        cache_dir=parsed.cache_dir,
        output_coverage_file=parsed.output_coverage,
    )

    print(
        f"Acquisition finished: leased={report.leased}, fetched={report.fetched}, "
        f"failed={report.failed}, blocked={report.blocked}, bytes={report.bytes_written}"
    )


if __name__ == "__main__":
    main()
