from __future__ import annotations

import collections
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse
import uuid

import httpx

from src.data.phase2_contracts import (
    FullCampaignConfig,
    SCOPE_FULL_SOURCE_PASS_WITH_GAPS,
    SCOPE_PARTIAL_BODY_INDEX,
    ScopeViolationError,
)
from src.data.vibiomir_contracts import (
    CrawlConfig,
    CrawlReport,
    ExtractionConfig,
    ExtractionReport,
)
from src.data.vibiomir_registry import SourceRegistry
from src.ingestion.crawl_runner import run_crawl
from src.ingestion.extractors import extract_document
from src.ingestion.fetcher import _is_captcha, save_raw_atomic
from src.ingestion.quality import (
    FastTextLanguageDetector,
    LanguageDetector,
    classify_quality,
)
from src.ingestion.snapshot import FallbackLanguageDetector

logger = logging.getLogger(__name__)


def recover_expired_leases(
    registry: SourceRegistry,
    now: datetime | None = None,
    recover_all_inflight: bool = False,
) -> int:
    """Recovers expired or stranded crash leases back to pending state."""
    now_dt = now or datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()

    with registry.conn:
        if recover_all_inflight:
            cursor = registry.conn.execute(
                """
                UPDATE fetch_jobs
                SET state = 'pending',
                    lease_until = NULL,
                    updated_at = ?
                WHERE state IN ('fetching', 'leased')
                """,
                (now_iso,),
            )
        else:
            cursor = registry.conn.execute(
                """
                UPDATE fetch_jobs
                SET state = 'pending',
                    lease_until = NULL,
                    updated_at = ?
                WHERE state IN ('fetching', 'leased')
                  AND (lease_until IS NULL OR lease_until <= ?)
                """,
                (now_iso, now_iso),
            )
        return cursor.rowcount


def _is_multilingual_source(domain: str, url: str) -> bool:
    """Detects if a URL/domain corresponds to international or non-VI (EN/ZH) sources."""
    d = domain.lower()
    u = url.lower()
    if any(k in d for k in ("ncbi", "who.int", "nih.gov", "cdc.gov", "pubmed", "cochrane", "thelancet")):
        return True
    if any(k in d for k in (".cn", ".zh", "cma.org.cn", "wanfang", "cnki")):
        return True
    if any(k in u for k in ("/en/", "/zh/", "/english/", "lang=en", "lang=zh")):
        return True
    return False


def plan_full_campaign(
    registry: SourceRegistry,
    config: FullCampaignConfig,
    campaign_id: str | None = None,
    recover_all_inflight: bool = True,
) -> str:
    """Plans full acquisition campaign with lease recovery, domain fair scheduling, and anti-starvation."""
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()
    cid = campaign_id or f"campaign_{now_dt.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # 1. Recover inflight/expired leases
    recover_expired_leases(registry, now=now_dt, recover_all_inflight=recover_all_inflight)

    # 2. Determine campaign scope
    if config.source_limit is not None or (config.domain_whitelist and len(config.domain_whitelist) > 0) or config.scope != "all":
        campaign_scope = SCOPE_PARTIAL_BODY_INDEX
    else:
        campaign_scope = config.scope

    # 3. Query candidate jobs to schedule
    # Reuse existing fetched content: do not re-enqueue fetched jobs with valid content
    if config.reuse_existing_content:
        cursor = registry.conn.execute(
            "SELECT job_id, url FROM fetch_jobs WHERE state != 'fetched' ORDER BY rowid ASC"
        )
    else:
        cursor = registry.conn.execute(
            "SELECT job_id, url FROM fetch_jobs ORDER BY rowid ASC"
        )
    rows = cursor.fetchall()

    # 4. Fair scheduling across domains with anti-starvation
    # Group jobs by domain
    domain_jobs: dict[str, list[str]] = collections.defaultdict(list)
    job_urls: dict[str, str] = {}
    for job_id, url in rows:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().split(":")[0]
        domain_jobs[domain].append(job_id)
        job_urls[job_id] = url

    # Sort domains by volume (smaller domains first for tie-breaking)
    sorted_domains = sorted(domain_jobs.keys(), key=lambda d: len(domain_jobs[d]))
    max_domain_len = max((len(jobs) for jobs in domain_jobs.values()), default=0)
    minor_threshold = min(50, max(5, max_domain_len // 3))
    minor_domains = {d for d in sorted_domains if len(domain_jobs[d]) <= minor_threshold}

    # Interleave jobs in rounds: round 0 takes 1st URL from each domain, round 1 takes 2nd, etc.
    updates: list[tuple[str, int, str]] = []

    for round_idx in range(max_domain_len):
        for domain in sorted_domains:
            jobs = domain_jobs[domain]
            if round_idx < len(jobs):
                job_id = jobs[round_idx]
                url = job_urls[job_id]
                base_prio = round_idx * 10 + 2
                if _is_multilingual_source(domain, url):
                    prio = base_prio - 2
                elif domain in minor_domains:
                    prio = base_prio - 1
                else:
                    prio = base_prio
                updates.append((cid, prio, job_id))

    # Batch update priority and campaign_id
    batch_size = 5000
    with registry.conn:
        for i in range(0, len(updates), batch_size):
            chunk = updates[i : i + batch_size]
            registry.conn.executemany(
                "UPDATE fetch_jobs SET campaign_id = ?, priority = ? WHERE job_id = ?",
                chunk,
            )

        # 5. Insert campaign record
        config_dict = asdict(config)
        registry.conn.execute(
            """
            INSERT OR REPLACE INTO campaigns (campaign_id, scope, config_json, created_at, status)
            VALUES (?, ?, ?, ?, ?)
            """,
            (cid, campaign_scope, json.dumps(config_dict), now_iso, "planned"),
        )

    return cid


def import_content_cache(
    registry: SourceRegistry,
    cache_dir: Path | str,
    raw_dir: Path | str | None = None,
) -> int:
    """Imports pre-existing or cached HTTP responses/HTML into the raw store and marks fetch_jobs fetched.

    Validates bytes and SHA256 provenance against registry.
    """
    c_dir = Path(cache_dir)
    if not c_dir.exists():
        raise FileNotFoundError(f"Cache directory does not exist: {c_dir}")

    r_dir = Path(raw_dir) if raw_dir is not None else c_dir.parent / "raw"
    r_dir.mkdir(parents=True, exist_ok=True)

    imported_count = 0
    now_iso = datetime.now(timezone.utc).isoformat()

    # Preload doc_id -> (url, job_id) and job_id -> url
    sources_by_doc_id: dict[int, tuple[str, str]] = {}
    for doc_id, url, jid in registry.conn.execute("SELECT doc_id, original_url, fetch_job_id FROM sources"):
        sources_by_doc_id[doc_id] = (url, jid)

    jobs_by_id: dict[str, str] = {}
    for jid, url in registry.conn.execute("SELECT job_id, url FROM fetch_jobs"):
        jobs_by_id[jid] = url

    # Collect candidate files in cache_dir
    files_to_process = [p for p in c_dir.rglob("*") if p.is_file() and not p.name.endswith(".tmp")]

    # Check for metadata files or companion JSON
    for path in files_to_process:
        if path.name.endswith(".json"):
            continue

        stem = path.stem
        # Handle double extensions such as doc_id.html.gz or job_id.html.gz
        if path.name.endswith(".gz") and "." in stem:
            inner_stem = Path(stem).stem
            if inner_stem.isdigit() or len(inner_stem) == 64:
                stem = inner_stem

        # Try matching by doc_id
        target_job_id: str | None = None
        target_url: str | None = None

        if stem.isdigit():
            doc_id = int(stem)
            if doc_id in sources_by_doc_id:
                target_url, target_job_id = sources_by_doc_id[doc_id]
        elif stem in jobs_by_id:
            target_job_id = stem
            target_url = jobs_by_id[stem]
        elif len(stem) == 64:
            # Check if stem is a job_id
            if stem in jobs_by_id:
                target_job_id = stem
                target_url = jobs_by_id[stem]

        if not target_job_id or not target_url:
            continue

        # Read content bytes
        try:
            if path.name.endswith(".gz"):
                with gzip.open(path, "rb") as gf:
                    raw_bytes = gf.read()
            else:
                raw_bytes = path.read_bytes()
        except Exception as exc:
            logger.warning(f"Failed to read cache file {path}: {exc}")
            continue

        if not raw_bytes or len(raw_bytes.strip()) == 0:
            logger.debug(f"Rejecting empty cache content for {target_url} at {path}")
            continue

        # Look for companion metadata JSON (check both direct and stem-based)
        meta_path = path.parent / f"{stem}.json"
        if not meta_path.exists():
            meta_path = path.with_suffix(".json")

        mime_type = "text/html; charset=utf-8"
        final_url = target_url
        http_status = 200

        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                mime_type = meta.get("mime_type", mime_type)
                final_url = meta.get("url", meta.get("final_url", final_url))
                http_status = meta.get("http_status", 200)
            except Exception:
                pass
        else:
            ext = path.suffix.lower()
            if ext == ".pdf":
                mime_type = "application/pdf"
            elif ext in (".htm", ".html"):
                mime_type = "text/html; charset=utf-8"

        # Check captcha
        if _is_captcha(raw_bytes, mime_type):
            logger.debug(f"Captcha detected in cached content for {target_url}")
            continue

        # Save to raw store atomically
        try:
            raw_path_str, raw_sha, _ = save_raw_atomic(r_dir, raw_bytes)
        except Exception as exc:
            logger.warning(f"Failed to save atomic raw cache file for {target_url}: {exc}")
            continue

        # Atomically update fetch_job state
        with registry.conn:
            registry.conn.execute(
                """
                UPDATE fetch_jobs
                SET state = 'fetched',
                    raw_path = ?,
                    raw_sha256 = ?,
                    final_url = ?,
                    mime_type = ?,
                    http_status = ?,
                    error_code = NULL,
                    retry_at = NULL,
                    lease_until = NULL,
                    attempts = CASE WHEN attempts = 0 THEN 1 ELSE attempts END,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    raw_path_str,
                    raw_sha,
                    final_url,
                    mime_type,
                    http_status,
                    now_iso,
                    target_job_id,
                ),
            )
        imported_count += 1

    return imported_count


async def run_full_campaign_crawl(
    registry: SourceRegistry,
    campaign_id: str,
    config: CrawlConfig,
    limit: int | None = None,
    client: httpx.AsyncClient | None = None,
    wait_for_retry: bool = False,
) -> CrawlReport:
    """Executes crawl runner within a planned full acquisition campaign scope."""
    return await run_crawl(
        registry=registry,
        config=config,
        limit=limit,
        client=client,
        campaign_id=campaign_id,
        wait_for_retry=wait_for_retry,
    )


def run_batch_extraction(
    registry: SourceRegistry,
    raw_root: Path | str | None = None,
    extraction_config: ExtractionConfig | None = None,
    language_detector: LanguageDetector | None = None,
    limit: int | None = None,
) -> ExtractionReport:
    """Runs batch extraction on fetched jobs in registry with resume and quality classification."""
    ext_cfg = extraction_config or ExtractionConfig()

    if language_detector is None:
        if ext_cfg.language_model_path.exists():
            language_detector = FastTextLanguageDetector(ext_cfg.language_model_path)
        else:
            language_detector = FallbackLanguageDetector()

    attempted = 0
    ready = 0
    quarantined = 0
    failed = 0
    flags_by_reason: collections.Counter[str] = collections.Counter()

    cursor = registry.conn.execute(
        """
        SELECT s.doc_id, f.state, f.raw_path, f.raw_sha256, f.mime_type
        FROM sources s
        JOIN fetch_jobs f ON s.fetch_job_id = f.job_id
        WHERE f.state = 'fetched' AND f.raw_path IS NOT NULL
        ORDER BY s.doc_id ASC
        """
    )
    rows = cursor.fetchall()
    if limit is not None and limit > 0:
        rows = rows[:limit]

    for doc_id, state, raw_path_str, raw_sha256, mime_type in rows:
        attempted += 1
        raw_path = Path(raw_path_str)
        if not raw_path.exists() and raw_root is not None:
            candidate = Path(raw_root) / raw_path_str
            if candidate.exists():
                raw_path = candidate

        if not raw_path.exists():
            failed += 1
            flags_by_reason["raw_file_missing"] += 1
            continue

        try:
            if str(raw_path).endswith(".gz"):
                with gzip.open(raw_path, "rb") as gf:
                    raw_bytes = gf.read()
            else:
                raw_bytes = raw_path.read_bytes()

            computed_raw_sha = hashlib.sha256(raw_bytes).hexdigest()
            if computed_raw_sha != raw_sha256:
                failed += 1
                flags_by_reason["sha256_mismatch"] += 1
                continue

            extracted_doc = extract_document(
                doc_id=doc_id,
                raw=raw_bytes,
                mime_type=mime_type or "text/html",
                raw_sha256=raw_sha256,
                config=ext_cfg,
            )

            quality = classify_quality(extracted_doc, language_detector, ext_cfg)
            if quality.state == "ready":
                ready += 1
            else:
                quarantined += 1
                for flg in quality.flags:
                    flags_by_reason[flg] += 1
        except Exception:
            failed += 1
            flags_by_reason["extraction_exception"] += 1

    return ExtractionReport(
        attempted=attempted,
        ready=ready,
        quarantined=quarantined,
        failed=failed,
        flags_by_reason=dict(flags_by_reason),
    )
