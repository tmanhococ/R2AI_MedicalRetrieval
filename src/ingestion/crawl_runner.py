import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse
import urllib.robotparser

import httpx

from src.data.vibiomir_contracts import CrawlConfig, CrawlReport, FetchJob, FetchResult
from src.data.vibiomir_registry import SourceRegistry
from src.ingestion.fetcher import fetch_one


class RobotsManager:
    """Caches and evaluates robots.txt rules per domain."""

    def __init__(self, user_agent: str = "ViBioMIR-Crawler/1.0"):
        self.user_agent = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._lock = asyncio.Lock()

    async def is_allowed(self, url: str, client: httpx.AsyncClient) -> bool:
        parsed = urlparse(url)
        origin = f"{parsed.scheme}://{parsed.netloc}"

        async with self._lock:
            if origin in self._parsers:
                rp = self._parsers[origin]
                if rp is None:
                    return True
                return rp.can_fetch(self.user_agent, url)

        # Fetch robots.txt
        robots_url = f"{origin}/robots.txt"
        rp: urllib.robotparser.RobotFileParser | None = None
        try:
            resp = await client.get(
                robots_url,
                headers={"User-Agent": self.user_agent},
                follow_redirects=True,
            )
            if resp.status_code == 200:
                rp = urllib.robotparser.RobotFileParser()
                rp.parse(resp.text.splitlines())
        except Exception:
            rp = None

        async with self._lock:
            self._parsers[origin] = rp

        if rp is None:
            return True
        return rp.can_fetch(self.user_agent, url)


class DomainThrottler:
    """Enforces per-domain concurrency and minimum request intervals."""

    def __init__(self, per_domain_concurrency: int, min_interval_seconds: float):
        self.per_domain_concurrency = per_domain_concurrency
        self.min_interval_seconds = min_interval_seconds
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._last_access: dict[str, float] = {}
        self._lock = asyncio.Lock()

    async def get_semaphore(self, domain: str) -> asyncio.Semaphore:
        async with self._lock:
            if domain not in self._semaphores:
                self._semaphores[domain] = asyncio.Semaphore(self.per_domain_concurrency)
            return self._semaphores[domain]

    async def wait_interval(self, domain: str) -> None:
        if self.min_interval_seconds <= 0:
            return
        wait = 0.0
        async with self._lock:
            last = self._last_access.get(domain, 0.0)
            now = asyncio.get_event_loop().time()
            wait = (last + self.min_interval_seconds) - now
            if wait > 0:
                self._last_access[domain] = now + wait
            else:
                self._last_access[domain] = now
                wait = 0.0
        if wait > 0:
            await asyncio.sleep(wait)


async def _run_crawl_with_client(
    registry: SourceRegistry,
    config: CrawlConfig,
    client: httpx.AsyncClient,
    limit: int | None = None,
    job_ids: Sequence[str] | None = None,
    campaign_id: str | None = None,
    wait_for_retry: bool = False,
) -> CrawlReport:
    global_sem = asyncio.Semaphore(config.concurrency)
    throttler = DomainThrottler(config.per_domain_concurrency, config.min_domain_interval_seconds)
    robots = RobotsManager()

    total_leased = 0
    fetched_count = 0
    retry_wait_count = 0
    failed_count = 0
    blocked_count = 0
    unsupported_count = 0
    total_bytes_written = 0

    active_tasks: set[asyncio.Task[FetchResult]] = set()

    async def _execute_job(job: FetchJob) -> FetchResult:
        parsed = urlparse(job.url)
        domain = parsed.netloc.lower().split(":")[0]

        async with global_sem:
            domain_sem = await throttler.get_semaphore(domain)
            async with domain_sem:
                await throttler.wait_interval(domain)

                if config.respect_robots:
                    allowed = await robots.is_allowed(job.url, client)
                    if not allowed:
                        now_iso = datetime.now(timezone.utc).isoformat()
                        return FetchResult(
                            job_id=job.job_id,
                            state="blocked",
                            raw_path=None,
                            raw_sha256=None,
                            final_url=job.url,
                            mime_type=None,
                            http_status=None,
                            fetched_at=now_iso,
                            error_code="robots_disallowed",
                            retry_at=None,
                            bytes_written=0,
                        )

                return await fetch_one(job, client, config)

    while True:
        available_slots = config.concurrency - len(active_tasks)
        if limit is not None:
            available_slots = min(available_slots, limit - total_leased)

        if available_slots > 0:
            new_jobs = registry.lease_jobs(
                limit=available_slots,
                lease_seconds=int(config.lease_seconds),
                max_attempts=config.max_attempts,
                job_ids=job_ids,
                campaign_id=campaign_id,
            )
            total_leased += len(new_jobs)
            for j in new_jobs:
                t = asyncio.create_task(_execute_job(j))
                active_tasks.add(t)

        if not active_tasks:
            if wait_for_retry and (limit is None or total_leased < limit):
                row = registry.conn.execute(
                    "SELECT min(retry_at) FROM fetch_jobs WHERE state = 'retry_wait'"
                ).fetchone()
                has_retry_jobs = bool(
                    (row and row[0] is not None)
                    or registry.conn.execute(
                        "SELECT 1 FROM fetch_jobs WHERE state = 'retry_wait' LIMIT 1"
                    ).fetchone()
                )
                if has_retry_jobs:
                    try:
                        if row and row[0]:
                            earliest = datetime.fromisoformat(row[0])
                            now_dt = datetime.now(timezone.utc)
                            diff = (earliest - now_dt).total_seconds()
                            if diff > 0:
                                await asyncio.sleep(min(diff + 0.1, 5.0))
                        continue
                    except Exception:
                        pass
            break

        done, active_tasks = await asyncio.wait(
            active_tasks, return_when=asyncio.FIRST_COMPLETED
        )

        for completed_task in done:
            result = completed_task.result()
            registry.finish_fetch(result)

            if result.state == "fetched":
                fetched_count += 1
            elif result.state == "retry_wait":
                retry_wait_count += 1
            elif result.state == "fetch_failed":
                failed_count += 1
            elif result.state == "blocked":
                blocked_count += 1
            elif result.state == "unsupported":
                unsupported_count += 1

            total_bytes_written += result.bytes_written

    return CrawlReport(
        leased=total_leased,
        fetched=fetched_count,
        retry_wait=retry_wait_count,
        failed=failed_count,
        blocked=blocked_count,
        unsupported=unsupported_count,
        bytes_written=total_bytes_written,
    )


async def run_crawl(
    registry: SourceRegistry,
    config: CrawlConfig,
    limit: int | None = None,
    client: httpx.AsyncClient | None = None,
    job_ids: Sequence[str] | None = None,
    campaign_id: str | None = None,
    wait_for_retry: bool = False,
) -> CrawlReport:
    """Executes crawler with global/domain semaphores, domain interval throttling, and atomic checkpointing."""
    if client is not None:
        return await _run_crawl_with_client(
            registry,
            config,
            client,
            limit,
            job_ids=job_ids,
            campaign_id=campaign_id,
            wait_for_retry=wait_for_retry,
        )

    limits = httpx.Limits(
        max_connections=config.concurrency,
        max_keepalive_connections=config.concurrency,
    )
    timeouts = httpx.Timeout(
        connect=config.connect_timeout_seconds,
        read=config.read_timeout_seconds,
        write=10.0,
        pool=10.0,
    )
    async with httpx.AsyncClient(
        limits=limits,
        timeout=timeouts,
        max_redirects=config.max_redirects,
    ) as managed_client:
        return await _run_crawl_with_client(
            registry,
            config,
            managed_client,
            limit,
            job_ids=job_ids,
            campaign_id=campaign_id,
            wait_for_retry=wait_for_retry,
        )
