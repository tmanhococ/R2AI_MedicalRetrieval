"""High-speed bulk corpus crawler for ViBioMIR.

Goals:
  - Fetch 4.39M URLs in < 5 hours (~250 URLs/sec sustained)
  - Checkpoint every 5000 URLs (resume-safe)
  - Domain-fair concurrency with per-domain semaphores
  - Batched SQLite registry updates (no per-URL lease overhead)
  - Progress reporting every 30 seconds

Architecture:
  1. Load all pending URLs from registry into memory (grouped by domain)
  2. Round-robin feed across domains into a global aiohttp pool
  3. Save raw content to disk (gzip + SHA256) reusing existing fetcher logic
  4. Batch-update registry every BATCH_COMMIT_SIZE items
  5. Write checkpoint file for resume
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import hashlib
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import uuid

import aiohttp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GLOBAL_CONCURRENCY = 600
PER_DOMAIN_CONCURRENCY_DEFAULT = 35
PER_DOMAIN_CONCURRENCY_LARGE = 75   # domains with > 100k URLs
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 10
TOTAL_TIMEOUT = 12
MAX_BODY_BYTES = 10 * 1024 * 1024   # 10 MiB per page
BATCH_COMMIT_SIZE = 5000
CHECKPOINT_INTERVAL = 5000
PROGRESS_INTERVAL = 15              # seconds
USER_AGENT = "Mozilla/5.0 (compatible; ViBioMIR-Crawler/2.0; +research)"

CAPTCHA_PATTERNS = [
    b"cf-browser-verification",
    b"cf-challenge",
    b"cf-chl-",
    b"<title>just a moment...</title>",
    b"attention required! | cloudflare",
    b"g-recaptcha",
    b"www.google.com/recaptcha",
    b"hcaptcha",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_captcha(body: bytes) -> bool:
    lower = body[:8192].lower()
    return any(p in lower for p in CAPTCHA_PATTERNS)


def _save_raw(raw_root: Path, body: bytes) -> tuple[str, str, int]:
    """Save body as content-addressed gzip. Returns (path, sha256, bytes_written)."""
    sha = hashlib.sha256(body).hexdigest()
    prefix_dir = raw_root / sha[:2]
    prefix_dir.mkdir(parents=True, exist_ok=True)
    target = prefix_dir / f"{sha}.gz"

    if target.exists():
        return str(target), sha, 0

    tmp = prefix_dir / f"{sha}.tmp_{uuid.uuid4().hex[:8]}.gz"
    try:
        with gzip.open(tmp, "wb") as f:
            f.write(body)
        bw = tmp.stat().st_size
        tmp.replace(target)
        return str(target), sha, bw
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Core crawler
# ---------------------------------------------------------------------------

class TurboCrawler:
    """High-throughput async crawler with domain-fair scheduling."""

    def __init__(
        self,
        urls: list[tuple[str, str]],  # (job_id, url)
        raw_root: Path,
        registry_path: Path,
        checkpoint_path: Path,
        global_concurrency: int = GLOBAL_CONCURRENCY,
        per_domain_default: int = PER_DOMAIN_CONCURRENCY_DEFAULT,
        per_domain_large: int = PER_DOMAIN_CONCURRENCY_LARGE,
        large_threshold: int = 100_000,
        failed_log_path: Path | None = None,
    ):
        self.raw_root = raw_root
        self.registry_path = registry_path
        self.checkpoint_path = checkpoint_path
        self.failed_log_path = failed_log_path
        self.global_concurrency = global_concurrency

        # Group URLs by domain for fair scheduling
        self.domain_queues: dict[str, collections.deque[tuple[str, str]]] = collections.defaultdict(collections.deque)
        self.domain_url_counts: dict[str, int] = collections.defaultdict(int)
        for job_id, url in urls:
            domain = urlparse(url).netloc.lower().split(":")[0]
            self.domain_queues[domain].append((job_id, url))
            self.domain_url_counts[domain] += 1

        # Per-domain semaphores
        self.domain_sems: dict[str, asyncio.Semaphore] = {}
        for domain, count in self.domain_url_counts.items():
            limit = per_domain_large if count > large_threshold else per_domain_default
            self.domain_sems[domain] = asyncio.Semaphore(limit)

        # Stats
        self.fetched = 0
        self.failed = 0
        self.blocked = 0
        self.bytes_written = 0
        self.total = len(urls)
        self.start_time = 0.0

        # Pending registry updates
        self._pending_updates: list[dict[str, Any]] = []
        self._update_lock = asyncio.Lock()

        # Shutdown flag
        self._shutdown = False

    def _progress_str(self) -> str:
        elapsed = time.time() - self.start_time
        done = self.fetched + self.failed + self.blocked
        rate = done / max(elapsed, 1)
        remaining = (self.total - done) / max(rate, 0.1)
        pct = done / max(self.total, 1) * 100
        return (
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"{done:,}/{self.total:,} ({pct:.1f}%) | "
            f"fetched={self.fetched:,} failed={self.failed:,} blocked={self.blocked:,} | "
            f"{rate:.1f} URLs/s | "
            f"data={self.bytes_written/1024/1024:.1f} MiB | "
            f"ETA={remaining/3600:.1f}h"
        )

    async def _fetch_one(
        self, session: aiohttp.ClientSession, job_id: str, url: str, domain: str
    ) -> None:
        """Fetch a single URL with domain semaphore."""
        now_iso = datetime.now(timezone.utc).isoformat()
        sem = self.domain_sems[domain]

        async with sem:
            if self._shutdown:
                return

            result: dict[str, Any] = {
                "job_id": job_id,
                "state": "fetch_failed",
                "raw_path": None,
                "raw_sha256": None,
                "final_url": url,
                "mime_type": None,
                "http_status": None,
                "fetched_at": now_iso,
                "error_code": None,
                "bytes_written": 0,
            }

            try:
                timeout = aiohttp.ClientTimeout(
                    total=TOTAL_TIMEOUT,
                    connect=CONNECT_TIMEOUT,
                    sock_read=READ_TIMEOUT,
                )
                async with session.get(
                    url,
                    timeout=timeout,
                    allow_redirects=True,
                    max_redirects=5,
                    headers={"User-Agent": USER_AGENT},
                    ssl=False,
                ) as resp:
                    status = resp.status
                    final_url = str(resp.url)
                    mime = resp.headers.get("Content-Type", "")
                    result["http_status"] = status
                    result["final_url"] = final_url
                    result["mime_type"] = mime

                    if status in (404, 410):
                        result["state"] = "fetch_failed"
                        result["error_code"] = f"http_{status}"
                    elif status == 403:
                        result["state"] = "blocked"
                        result["error_code"] = "http_403"
                        self.blocked += 1
                    elif status in (408, 429) or 500 <= status <= 599:
                        result["state"] = "fetch_failed"
                        result["error_code"] = f"http_{status}"
                        self.failed += 1
                    elif 400 <= status <= 499:
                        result["state"] = "fetch_failed"
                        result["error_code"] = f"http_{status}"
                        self.failed += 1
                    else:
                        # Read body
                        body = await resp.read()
                        if len(body) > MAX_BODY_BYTES:
                            result["state"] = "fetch_failed"
                            result["error_code"] = "body_too_large"
                            self.failed += 1
                        elif _is_captcha(body):
                            result["state"] = "blocked"
                            result["error_code"] = "captcha"
                            self.blocked += 1
                        elif len(body.strip()) == 0:
                            result["state"] = "fetch_failed"
                            result["error_code"] = "empty_body"
                            self.failed += 1
                        else:
                            raw_path, raw_sha, bw = _save_raw(self.raw_root, body)
                            result["state"] = "fetched"
                            result["raw_path"] = raw_path
                            result["raw_sha256"] = raw_sha
                            result["bytes_written"] = bw
                            self.fetched += 1
                            self.bytes_written += bw

                    if result["state"] == "fetch_failed" and result["error_code"] and result["error_code"].startswith("http_"):
                        self.failed += 1

            except asyncio.TimeoutError:
                result["error_code"] = "timeout"
                self.failed += 1
            except aiohttp.ClientError as exc:
                result["error_code"] = f"client_error:{type(exc).__name__}"
                self.failed += 1
            except Exception as exc:
                result["error_code"] = f"unexpected:{type(exc).__name__}"
                self.failed += 1

            # Queue registry update
            async with self._update_lock:
                self._pending_updates.append(result)

    async def _flush_to_registry(self) -> None:
        """Batch-commit pending results to SQLite."""
        async with self._update_lock:
            if not self._pending_updates:
                return
            batch = self._pending_updates[:]
            self._pending_updates.clear()

        import sqlite3
        conn = sqlite3.connect(str(self.registry_path), timeout=30)
        try:
            params = [
                (
                    r["state"],
                    r["raw_path"],
                    r["raw_sha256"],
                    r["final_url"],
                    r["mime_type"],
                    r["http_status"],
                    r["error_code"],
                    r["fetched_at"],
                    r["job_id"],
                )
                for r in batch
            ]
            with conn:
                conn.executemany(
                    """
                    UPDATE fetch_jobs
                    SET state = ?,
                        raw_path = ?,
                        raw_sha256 = ?,
                        final_url = ?,
                        mime_type = ?,
                        http_status = ?,
                        error_code = ?,
                        lease_until = NULL,
                        retry_at = NULL,
                        attempts = attempts + 1,
                        updated_at = ?
                    WHERE job_id = ?
                    """,
                    params,
                )
        finally:
            conn.close()

        # Append failures to failed_log_path
        if self.failed_log_path:
            failed_records = [
                r for r in batch if r["state"] in ("fetch_failed", "blocked")
            ]
            if failed_records:
                try:
                    self.failed_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.failed_log_path, "a", encoding="utf-8") as f:
                        for r in failed_records:
                            f.write(
                                json.dumps(
                                    {
                                        "job_id": r["job_id"],
                                        "url": r["final_url"],
                                        "state": r["state"],
                                        "error_code": r["error_code"],
                                        "http_status": r["http_status"],
                                        "timestamp": r["fetched_at"],
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                except Exception:
                    pass

    def _save_checkpoint(self) -> None:
        """Save checkpoint with list of completed job_ids."""
        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "fetched": self.fetched,
            "failed": self.failed,
            "blocked": self.blocked,
            "bytes_written": self.bytes_written,
            "total": self.total,
            "elapsed_seconds": time.time() - self.start_time,
        }
        tmp = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(self.checkpoint_path)

    async def _progress_reporter(self) -> None:
        """Print progress every PROGRESS_INTERVAL seconds."""
        while not self._shutdown:
            await asyncio.sleep(PROGRESS_INTERVAL)
            print(self._progress_str(), flush=True)
            # Flush pending updates periodically
            await self._flush_to_registry()
            self._save_checkpoint()

    async def run(self) -> None:
        """Main crawl loop with fixed worker pool and domain-fair queue feeding."""
        self.start_time = time.time()
        self.raw_root.mkdir(parents=True, exist_ok=True)

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_signal)
            except NotImplementedError:
                signal.signal(sig, lambda s, f: self._handle_signal())

        connector = aiohttp.TCPConnector(
            limit=self.global_concurrency * 2,
            limit_per_host=0,
            ttl_dns_cache=600,
            enable_cleanup_closed=True,
            force_close=False,
            keepalive_timeout=30,
        )

        work_queue: asyncio.Queue[tuple[str, str, str] | None] = asyncio.Queue(maxsize=self.global_concurrency * 2)

        async with aiohttp.ClientSession(connector=connector) as session:
            # 1. Start progress reporter
            reporter = asyncio.create_task(self._progress_reporter())

            # 2. Worker coroutine
            async def worker() -> None:
                while not self._shutdown:
                    item = await work_queue.get()
                    if item is None:
                        work_queue.task_done()
                        break
                    job_id, url, domain = item
                    try:
                        await self._fetch_one(session, job_id, url, domain)
                    except Exception:
                        pass
                    finally:
                        work_queue.task_done()

            # Spawn fixed worker pool
            workers = [asyncio.create_task(worker()) for _ in range(self.global_concurrency)]

            # 3. Feeder coroutine: round-robin over domain queues
            domains_with_work = list(self.domain_queues.keys())
            batch_counter = 0

            while domains_with_work and not self._shutdown:
                next_domains = []
                for domain in domains_with_work:
                    q = self.domain_queues[domain]
                    if q:
                        job_id, url = q.popleft()
                        await work_queue.put((job_id, url, domain))
                        batch_counter += 1

                        if q:
                            next_domains.append(domain)

                    if self._shutdown:
                        break

                domains_with_work = next_domains

                # Periodic flush to DB
                if batch_counter >= BATCH_COMMIT_SIZE:
                    await self._flush_to_registry()
                    batch_counter = 0

            # Feed termination sentinels to all workers
            for _ in range(self.global_concurrency):
                await work_queue.put(None)

            # Wait for all workers to finish
            await asyncio.gather(*workers, return_exceptions=True)

            # Final flush and checkpoint
            await self._flush_to_registry()
            self._save_checkpoint()

            # Stop reporter
            self._shutdown = True
            reporter.cancel()
            try:
                await reporter
            except asyncio.CancelledError:
                pass

        elapsed = time.time() - self.start_time
        print(f"\n{'='*60}", flush=True)
        print("CRAWL COMPLETE", flush=True)
        print(f"{'='*60}", flush=True)
        print(self._progress_str(), flush=True)
        print(f"Total time: {elapsed/3600:.2f} hours", flush=True)

    def _handle_signal(self) -> None:
        print("\n\n[WARNING] Shutdown signal received. Finishing in-flight requests...", flush=True)
        self._shutdown = True


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def load_pending_urls(registry_path: Path, limit: int | None = None) -> list[tuple[str, str]]:
    """Load all pending/fetching URLs from the registry."""
    import sqlite3
    conn = sqlite3.connect(str(registry_path), timeout=30)
    try:
        sql = "SELECT job_id, url FROM fetch_jobs WHERE state IN ('pending', 'fetching', 'retry_wait') ORDER BY rowid"
        if limit and limit > 0:
            sql += f" LIMIT {limit}"
        rows = conn.execute(sql).fetchall()
        return [(r[0], r[1]) for r in rows]
    finally:
        conn.close()


def reset_inflight(registry_path: Path) -> int:
    """Reset any stuck fetching/leased jobs back to pending."""
    import sqlite3
    conn = sqlite3.connect(str(registry_path), timeout=30)
    try:
        with conn:
            cur = conn.execute(
                "UPDATE fetch_jobs SET state = 'pending', lease_until = NULL WHERE state IN ('fetching', 'leased')"
            )
            return cur.rowcount
    finally:
        conn.close()


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Turbo bulk crawler for ViBioMIR corpus")
    parser.add_argument("--registry", type=Path, default=Path("artifacts/vibiomir/registry.sqlite"))
    parser.add_argument("--raw-root", type=Path, default=Path("artifacts/vibiomir/raw"))
    parser.add_argument("--checkpoint", type=Path, default=Path("artifacts/vibiomir/turbo_checkpoint.json"))
    parser.add_argument("--concurrency", type=int, default=GLOBAL_CONCURRENCY)
    parser.add_argument("--per-domain", type=int, default=PER_DOMAIN_CONCURRENCY_DEFAULT)
    parser.add_argument("--per-domain-large", type=int, default=PER_DOMAIN_CONCURRENCY_LARGE)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of URLs to crawl")
    parser.add_argument("--failed-log", type=Path, default=Path("artifacts/vibiomir/failed_urls.jsonl"), help="Path to write failed/blocked URLs log")
    parser.add_argument("--retry-failed", action="store_true", help="Reset all failed/blocked jobs in DB to pending before crawling")
    parsed = parser.parse_args(args)

    print("=" * 60, flush=True)
    print("ViBioMIR TURBO CRAWLER", flush=True)
    print("=" * 60, flush=True)

    # Optional retry of failed jobs
    if parsed.retry_failed:
        import sqlite3
        conn = sqlite3.connect(str(parsed.registry), timeout=30)
        try:
            with conn:
                cur = conn.execute(
                    "UPDATE fetch_jobs SET state = 'pending', lease_until = NULL, retry_at = NULL WHERE state IN ('fetch_failed', 'blocked')"
                )
                print(f"Reset {cur.rowcount:,} failed/blocked jobs to pending for retry", flush=True)
        finally:
            conn.close()

    # Reset stuck jobs
    recovered = reset_inflight(parsed.registry)
    if recovered:
        print(f"Recovered {recovered} stuck inflight jobs", flush=True)

    # Load all pending URLs
    print("Loading pending URLs from registry...", flush=True)
    urls = load_pending_urls(parsed.registry, limit=parsed.limit)
    print(f"Loaded {len(urls):,} pending URLs across domains", flush=True)
    print(f"Global concurrency: {parsed.concurrency}", flush=True)
    print(f"Per-domain concurrency: {parsed.per_domain} (large: {parsed.per_domain_large})", flush=True)
    print(f"Target rate: ~250 URLs/sec -> ETA ~{len(urls)/250/3600:.1f} hours", flush=True)
    print(f"Failed URLs log: {parsed.failed_log}", flush=True)
    print("=" * 60, flush=True)

    if not urls:
        print("No pending URLs. Done!", flush=True)
        return

    crawler = TurboCrawler(
        urls=urls,
        raw_root=parsed.raw_root,
        registry_path=parsed.registry,
        checkpoint_path=parsed.checkpoint,
        failed_log_path=parsed.failed_log,
        global_concurrency=parsed.concurrency,
        per_domain_default=parsed.per_domain,
        per_domain_large=parsed.per_domain_large,
    )

    asyncio.run(crawler.run())


if __name__ == "__main__":
    main()
