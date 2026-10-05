from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock
import httpx
import pytest

from src.data.phase2_contracts import (
    FullCampaignConfig,
    SCOPE_PARTIAL_BODY_INDEX,
    SCOPE_FULL_SOURCE_PASS_WITH_GAPS,
    ScopeViolationError,
)
from src.data.vibiomir_contracts import CrawlConfig, OfficialSource
from src.data.vibiomir_registry import SourceRegistry, ingest_sources
from src.ingestion.full_campaign import (
    import_content_cache,
    plan_full_campaign,
    recover_expired_leases,
    run_batch_extraction,
    run_full_campaign_crawl,
)
from src.cli.crawl_full import run_full_crawl_cli


def _make_sample_sources() -> list[OfficialSource]:
    sources = []
    # Major domain A: 30 URLs
    for i in range(1, 31):
        sources.append(OfficialSource(doc_id=i, url=f"https://large-hospital.vn/article/{i}"))
    # Minor domain B: 2 URLs
    sources.append(OfficialSource(doc_id=31, url="https://small-clinic.org/page/1"))
    sources.append(OfficialSource(doc_id=32, url="https://small-clinic.org/page/2"))
    # Multilingual domain C (EN/ZH): 3 URLs
    sources.append(OfficialSource(doc_id=33, url="https://ncbi.nlm.nih.gov/pmc/articles/PMC123456/"))
    sources.append(OfficialSource(doc_id=34, url="https://who.int/en/news-room/fact-sheets/detail/disease-1"))
    sources.append(OfficialSource(doc_id=35, url="https://cma.org.cn/zh/article/999"))
    return sources


def test_plan_full_campaign_initialization_and_id(tmp_path: Path):
    """plan_full_campaign generates a unique campaign_id and initializes queue metadata."""
    reg_path = tmp_path / "registry.sqlite"
    registry = SourceRegistry(reg_path)
    sources = _make_sample_sources()
    ingest_sources(sources, registry)

    config = FullCampaignConfig(scope="all")
    campaign_id = plan_full_campaign(registry, config)

    assert campaign_id is not None
    assert isinstance(campaign_id, str)
    assert campaign_id.startswith("campaign_")

    # Verify campaign record stored in database
    row = registry.conn.execute(
        "SELECT campaign_id, scope, status FROM campaigns WHERE campaign_id = ?",
        (campaign_id,),
    ).fetchone()
    assert row is not None
    assert row[0] == campaign_id
    assert row[1] == "all"

    # Verify fetch_jobs have campaign_id and priority assigned
    jobs = registry.conn.execute("SELECT job_id, campaign_id, priority FROM fetch_jobs").fetchall()
    assert len(jobs) == len(sources)
    for jid, cid, prio in jobs:
        assert cid == campaign_id
        assert prio is not None
        assert prio >= 0

    registry.close()


def test_reuses_existing_pilot_content(tmp_path: Path):
    """Existing fetched pilot content is reused without refetching or resetting state."""
    reg_path = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    registry = SourceRegistry(reg_path)
    sources = [
        OfficialSource(doc_id=1, url="https://example.com/pilot_article_1"),
        OfficialSource(doc_id=2, url="https://example.com/new_article_2"),
    ]
    ingest_sources(sources, registry)

    # Simulate pilot article 1 already fetched
    pilot_job_id = hashlib.sha256("https://example.com/pilot_article_1".encode("utf-8")).hexdigest()
    raw_content = b"<html><body>Pilot content article 1</body></html>"
    raw_sha = hashlib.sha256(raw_content).hexdigest()
    target_raw = raw_root / raw_sha[:2] / f"{raw_sha}.gz"
    target_raw.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(target_raw, "wb") as f:
        f.write(raw_content)

    with registry.conn:
        registry.conn.execute(
            """
            UPDATE fetch_jobs
            SET state = 'fetched', raw_path = ?, raw_sha256 = ?, http_status = 200, mime_type = 'text/html'
            WHERE job_id = ?
            """,
            (str(target_raw), raw_sha, pilot_job_id),
        )

    config = FullCampaignConfig(scope="all", reuse_existing_content=True)
    campaign_id = plan_full_campaign(registry, config)

    # Pilot article 1 should remain 'fetched' and keep its raw_path and raw_sha256
    pilot_job = registry.get_fetch_job(pilot_job_id)
    assert pilot_job is not None
    assert pilot_job["state"] == "fetched"
    assert pilot_job["raw_sha256"] == raw_sha

    # Only new_article_2 should be leased when leasing jobs
    leased = registry.lease_jobs(limit=10)
    assert len(leased) == 1
    assert leased[0].url == "https://example.com/new_article_2"

    registry.close()


def test_fair_domain_scheduling_covers_minor_domains_and_multilingual(tmp_path: Path):
    """Fair domain scheduling guarantees minor domains and EN/ZH are leased in early rounds."""
    reg_path = tmp_path / "registry.sqlite"
    registry = SourceRegistry(reg_path)
    sources = _make_sample_sources()  # 30 for hospital, 2 for clinic, 3 for multilingual
    ingest_sources(sources, registry)

    config = FullCampaignConfig(scope="all", priority_changes_order_only=True)
    plan_full_campaign(registry, config)

    # First lease batch of size 6 (representing slots across domains)
    leased = registry.lease_jobs(limit=6)
    leased_urls = [job.url for job in leased]

    # Must include small clinic, NCBI / WHO, alongside large hospital URLs
    domains_served = {job.url.split("/")[2] for job in leased}
    assert "small-clinic.org" in domains_served
    assert ("ncbi.nlm.nih.gov" in domains_served) or ("who.int" in domains_served)
    assert "large-hospital.vn" in domains_served

    registry.close()


@pytest.mark.asyncio
async def test_same_url_aliases_preserved_across_multiple_doc_ids(tmp_path: Path):
    """Same URL associated with multiple doc_ids is fetched once and satisfies all doc_ids."""
    reg_path = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    registry = SourceRegistry(reg_path)

    shared_url = "https://who.int/medical-standard-guideline"
    sources = [
        OfficialSource(doc_id=1001, url=shared_url),
        OfficialSource(doc_id=1002, url=shared_url),
        OfficialSource(doc_id=1003, url=shared_url),
    ]
    ingest_sources(sources, registry)

    config = FullCampaignConfig(scope="all")
    campaign_id = plan_full_campaign(registry, config)

    crawl_config = CrawlConfig(raw_root=raw_root)
    fetch_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal fetch_calls
        if "/robots.txt" in str(request.url):
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        fetch_calls += 1
        return httpx.Response(
            200,
            content=b"<html><body><h1>Official Guideline</h1></body></html>",
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_full_campaign_crawl(
            registry=registry,
            campaign_id=campaign_id,
            config=crawl_config,
            client=client,
        )

    assert report.fetched == 1
    assert fetch_calls == 1

    # Check all 3 official doc_ids resolve to fetched content
    job_id = hashlib.sha256(shared_url.encode("utf-8")).hexdigest()
    job_record = registry.get_fetch_job(job_id)
    assert job_record["state"] == "fetched"

    for doc_id in (1001, 1002, 1003):
        src = registry.get_source(doc_id)
        assert src is not None
        assert src.url == shared_url

    registry.close()


def test_crash_leases_recover_on_resume(tmp_path: Path):
    """Crash leases (stranded in 'leased' or 'fetching') are recovered back to 'pending' on resume."""
    reg_path = tmp_path / "registry.sqlite"
    registry = SourceRegistry(reg_path)
    sources = [
        OfficialSource(doc_id=1, url="https://example.com/crashed_1"),
        OfficialSource(doc_id=2, url="https://example.com/crashed_2"),
    ]
    ingest_sources(sources, registry)

    # Simulate crashed jobs left in 'fetching' or 'leased'
    with registry.conn:
        registry.conn.execute(
            """
            UPDATE fetch_jobs
            SET state = 'fetching', lease_until = '2026-10-01T00:00:00Z', attempts = 1
            WHERE job_id = (SELECT fetch_job_id FROM sources WHERE doc_id = 1)
            """
        )
        registry.conn.execute(
            """
            UPDATE fetch_jobs
            SET state = 'leased', lease_until = '2026-10-01T00:00:00Z', attempts = 1
            WHERE job_id = (SELECT fetch_job_id FROM sources WHERE doc_id = 2)
            """
        )

    recovered = recover_expired_leases(registry)
    assert recovered == 2

    # Verify both jobs are back to 'pending'
    rows = registry.conn.execute("SELECT state, lease_until FROM fetch_jobs").fetchall()
    assert all(st == "pending" and lu is None for st, lu in rows)

    registry.close()


def test_limit_or_domain_filter_marks_scope_partial(tmp_path: Path):
    """Setting --limit or domain_whitelist marks the campaign scope as PARTIAL_BODY_INDEX."""
    reg_path = tmp_path / "registry.sqlite"
    registry = SourceRegistry(reg_path)
    sources = _make_sample_sources()
    ingest_sources(sources, registry)

    # 1. With limit
    cfg_limit = FullCampaignConfig(scope="all", source_limit=10)
    cid_limit = plan_full_campaign(registry, cfg_limit)
    row = registry.conn.execute("SELECT scope FROM campaigns WHERE campaign_id = ?", (cid_limit,)).fetchone()
    assert row[0] == SCOPE_PARTIAL_BODY_INDEX

    # 2. With domain whitelist
    cfg_domain = FullCampaignConfig(scope="all", domain_whitelist=["small-clinic.org"])
    cid_domain = plan_full_campaign(registry, cfg_domain)
    row = registry.conn.execute("SELECT scope FROM campaigns WHERE campaign_id = ?", (cid_domain,)).fetchone()
    assert row[0] == SCOPE_PARTIAL_BODY_INDEX

    registry.close()


def test_no_domain_blacklist_masquerading_as_completion(tmp_path: Path):
    """Filtering domains does not allow claiming full completion; unresolved sources remain tracked."""
    reg_path = tmp_path / "registry.sqlite"
    registry = SourceRegistry(reg_path)
    sources = _make_sample_sources()
    ingest_sources(sources, registry)

    cfg = FullCampaignConfig(scope="all", domain_whitelist=["small-clinic.org"])
    cid = plan_full_campaign(registry, cfg)

    # The excluded domains cannot be silently marked terminal/complete
    pending_count = registry.conn.execute("SELECT count(*) FROM fetch_jobs WHERE state = 'pending'").fetchone()[0]
    assert pending_count == len(sources)

    registry.close()


def test_content_cache_import_success(tmp_path: Path):
    """Importing pre-cached HTML/bytes validates sha256, populates raw store, and marks jobs fetched."""
    reg_path = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    registry = SourceRegistry(reg_path)
    sources = [
        OfficialSource(doc_id=501, url="https://example.com/cached_doc_501"),
        OfficialSource(doc_id=502, url="https://example.com/cached_doc_502"),
    ]
    ingest_sources(sources, registry)

    # Prepare cache files:
    # 1. Cache file by doc_id
    doc501_content = b"<html><body>Content for Doc 501</body></html>"
    (cache_dir / "501.html").write_bytes(doc501_content)

    # 2. Cache file by job_id with metadata JSON
    job502_id = hashlib.sha256("https://example.com/cached_doc_502".encode("utf-8")).hexdigest()
    doc502_content = b"<html><body>Content for Doc 502 with metadata</body></html>"
    (cache_dir / f"{job502_id}.html").write_bytes(doc502_content)
    (cache_dir / f"{job502_id}.json").write_text(
        json.dumps({
            "url": "https://example.com/cached_doc_502",
            "mime_type": "text/html; charset=utf-8",
            "http_status": 200,
        })
    )

    imported_count = import_content_cache(registry=registry, cache_dir=cache_dir, raw_dir=raw_root)
    assert imported_count == 2

    # Check that both jobs are marked 'fetched' in registry
    j1 = registry.get_fetch_job(hashlib.sha256("https://example.com/cached_doc_501".encode("utf-8")).hexdigest())
    j2 = registry.get_fetch_job(job502_id)
    assert j1["state"] == "fetched"
    assert j2["state"] == "fetched"
    assert Path(j1["raw_path"]).exists()
    assert Path(j2["raw_path"]).exists()

    with gzip.open(j1["raw_path"], "rb") as gf:
        assert gf.read() == doc501_content

    registry.close()


def test_content_cache_import_corrupted_rejected(tmp_path: Path):
    """Empty or invalid content cache files are rejected and not marked fetched."""
    reg_path = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    registry = SourceRegistry(reg_path)
    sources = [
        OfficialSource(doc_id=701, url="https://example.com/corrupt_doc_701"),
    ]
    ingest_sources(sources, registry)

    # Empty file
    (cache_dir / "701.html").write_bytes(b"")

    imported = import_content_cache(registry=registry, cache_dir=cache_dir, raw_dir=raw_root)
    assert imported == 0

    job_id = hashlib.sha256("https://example.com/corrupt_doc_701".encode("utf-8")).hexdigest()
    j = registry.get_fetch_job(job_id)
    assert j["state"] == "pending"

    registry.close()


def test_crawl_full_cli_smoke(tmp_path: Path):
    """crawl_full CLI runner runs with --scope all and cleanly stops when queue completes."""
    reg_path = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    registry = SourceRegistry(reg_path)
    sources = [
        OfficialSource(doc_id=1, url="https://example.com/item1"),
    ]
    ingest_sources(sources, registry)
    registry.close()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html><body>Test</body></html>", headers={"Content-Type": "text/html"})

    report = run_full_crawl_cli(
        registry_path=reg_path,
        raw_root=raw_root,
        scope="all",
        test_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    assert report.fetched == 1
    assert report.failed == 0


@pytest.mark.asyncio
async def test_5k_scale_smoke_and_throughput(tmp_path: Path):
    """5k scale smoke test verifying planning, fair scheduling, host throughput, and disk estimates."""
    import collections

    reg_path = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    registry = SourceRegistry(reg_path)

    sources = []
    # Major domain: 4000
    for i in range(1, 4001):
        sources.append(OfficialSource(doc_id=i, url=f"https://hospital-large.vn/news/{i}"))
    # Medium domain: 800
    for i in range(4001, 4801):
        sources.append(OfficialSource(doc_id=i, url=f"https://clinic-medium.vn/post/{i}"))
    # Minor domains: 100
    for i in range(4801, 4901):
        sources.append(OfficialSource(doc_id=i, url=f"https://minor-clinic-{i % 5}.vn/doc/{i}"))
    # Multilingual: 100 (EN/ZH)
    for i in range(4901, 4951):
        sources.append(OfficialSource(doc_id=i, url=f"https://ncbi.nlm.nih.gov/article/{i}"))
    for i in range(4951, 5001):
        sources.append(OfficialSource(doc_id=i, url=f"https://cma.org.cn/zh/paper/{i}"))

    ingest_sources(sources, registry, batch_size=1000)
    assert registry.count_sources() == 5000

    cfg = FullCampaignConfig(scope="all")
    t_start = datetime.now(timezone.utc)
    cid = plan_full_campaign(registry, cfg)
    plan_duration = (datetime.now(timezone.utc) - t_start).total_seconds()
    assert plan_duration < 5.0

    # Inspect first 100 leased jobs:
    leased_sample = registry.lease_jobs(limit=100)
    domains_in_sample = {job.url.split("/")[2] for job in leased_sample}
    assert "ncbi.nlm.nih.gov" in domains_in_sample
    assert "cma.org.cn" in domains_in_sample
    assert any("minor-clinic" in d for d in domains_in_sample)

    # Reset leases for crawl run
    with registry.conn:
        registry.conn.execute("UPDATE fetch_jobs SET state = 'pending', lease_until = NULL")

    host_request_counts: dict[str, int] = collections.defaultdict(int)
    total_bytes = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal total_bytes
        h = str(request.url.host)
        host_request_counts[h] += 1
        html = f"<html><body><h3>Article for {request.url}</h3><p>Medical body text data.</p></body></html>".encode("utf-8")
        total_bytes += len(html)
        return httpx.Response(200, content=html, headers={"Content-Type": "text/html; charset=utf-8"})

    crawl_cfg = CrawlConfig(raw_root=raw_root, concurrency=32, min_domain_interval_seconds=0.0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(mock_handler)) as client:
        report = await run_full_campaign_crawl(
            registry=registry,
            campaign_id=cid,
            config=crawl_cfg,
            limit=200,
            client=client,
        )

    assert report.fetched == 200
    assert report.failed == 0
    assert len(host_request_counts) >= 4

    avg_bytes_per_doc = report.bytes_written / max(report.fetched, 1)
    estimated_full_corpus_gb = (avg_bytes_per_doc * 4394718) / (1024**3)
    assert avg_bytes_per_doc > 0
    assert estimated_full_corpus_gb > 0

    registry.close()


def test_round_zero_priority_boost(tmp_path: Path):
    """Round 0 guarantees distinct scheduling priorities for multilingual (0), minor (1), and major (2)."""
    reg_path = tmp_path / "registry.sqlite"
    registry = SourceRegistry(reg_path)
    sources = _make_sample_sources()
    ingest_sources(sources, registry)

    cfg = FullCampaignConfig(scope="all")
    cid = plan_full_campaign(registry, cfg)

    # In round 0, multilingual URL should have priority 0
    ncbi_job_id = hashlib.sha256("https://ncbi.nlm.nih.gov/pmc/articles/PMC123456/".encode("utf-8")).hexdigest()
    clinic_job_id = hashlib.sha256("https://small-clinic.org/page/1".encode("utf-8")).hexdigest()
    hospital_job_id = hashlib.sha256("https://large-hospital.vn/article/1".encode("utf-8")).hexdigest()

    ncbi_job = registry.get_fetch_job(ncbi_job_id)
    clinic_job = registry.get_fetch_job(clinic_job_id)
    hospital_job = registry.get_fetch_job(hospital_job_id)

    assert ncbi_job["priority"] == 0
    assert clinic_job["priority"] == 1
    assert hospital_job["priority"] == 2

    registry.close()


def test_resume_cli_reuses_existing_campaign_and_recovers_inflight_leases(tmp_path: Path):
    """Running CLI with --resume reuses previous campaign_id, recovers in-flight leases, and completes."""
    from datetime import timedelta
    reg_path = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    registry = SourceRegistry(reg_path)

    sources = [
        OfficialSource(doc_id=1, url="https://example.com/item1"),
        OfficialSource(doc_id=2, url="https://example.com/item2"),
    ]
    ingest_sources(sources, registry)

    # Initial campaign planned
    cfg = FullCampaignConfig(scope="all")
    orig_cid = plan_full_campaign(registry, cfg)

    # Simulate in-flight lease with lease_until in the future (within 120s)
    future_iso = (datetime.now(timezone.utc) + timedelta(seconds=100)).isoformat()
    job1_id = hashlib.sha256("https://example.com/item1".encode("utf-8")).hexdigest()
    with registry.conn:
        registry.conn.execute(
            "UPDATE fetch_jobs SET state = 'fetching', lease_until = ? WHERE job_id = ?",
            (future_iso, job1_id),
        )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html><body>Test</body></html>", headers={"Content-Type": "text/html"})

    # Run CLI with resume=True
    report = run_full_crawl_cli(
        registry_path=reg_path,
        raw_root=raw_root,
        scope="all",
        resume=True,
        test_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    # Verified: reused orig_cid and fetched both jobs
    assert report.fetched == 2
    row = registry.conn.execute(
        "SELECT status FROM campaigns WHERE campaign_id = ?", (orig_cid,)
    ).fetchone()
    assert row is not None
    assert row[0] == "completed"

    registry.close()


@pytest.mark.asyncio
async def test_retry_wait_matured_continuation(tmp_path: Path):
    """When a retry_wait job backoff is already matured (diff <= 0), crawl loop does not prematurely terminate."""
    from datetime import timedelta
    reg_path = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    registry = SourceRegistry(reg_path)

    sources = [
        OfficialSource(doc_id=1, url="https://example.com/retry_ready"),
    ]
    ingest_sources(sources, registry)

    # Put job in retry_wait with past retry_at
    past_iso = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
    job_id = hashlib.sha256("https://example.com/retry_ready".encode("utf-8")).hexdigest()
    with registry.conn:
        registry.conn.execute(
            "UPDATE fetch_jobs SET state = 'retry_wait', retry_at = ? WHERE job_id = ?",
            (past_iso, job_id),
        )

    cfg = FullCampaignConfig(scope="all")
    cid = plan_full_campaign(registry, cfg)

    crawl_cfg = CrawlConfig(raw_root=raw_root, min_domain_interval_seconds=0.0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html><body>Retry success</body></html>", headers={"Content-Type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_full_campaign_crawl(
            registry=registry,
            campaign_id=cid,
            config=crawl_cfg,
            client=client,
            wait_for_retry=True,
        )

    assert report.fetched == 1
    assert report.failed == 0

    job_rec = registry.get_fetch_job(job_id)
    assert job_rec["state"] == "fetched"

    registry.close()


def test_run_batch_extraction_integration(tmp_path: Path):
    """run_batch_extraction extracts fetched documents and produces quality classified report."""
    reg_path = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry(reg_path)

    sources = [
        OfficialSource(doc_id=1, url="https://example.com/doc1"),
        OfficialSource(doc_id=2, url="https://example.com/doc2"),
    ]
    ingest_sources(sources, registry)

    # 1. Valid Vietnamese HTML
    doc1_content = (
        b"<!DOCTYPE html><html><head><title>Dieu tri viem hong</title></head>"
        b"<body><article><h1>Viem hong cap tinh</h1>"
        b"<p>Viem hong cap la benh pho bien o tre em va nguoi lon. "
        b"Trieu chung thuong gap la dau hong, sot, kho nuot va ho khan. "
        b"Can nghi ngoi day du va uong nhieu nuoc am de phuc hoi suc khoe nhanh chong.</p>"
        b"</article></body></html>"
    )
    sha1 = hashlib.sha256(doc1_content).hexdigest()
    file1 = raw_root / sha1[:2] / f"{sha1}.gz"
    file1.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(file1, "wb") as gf:
        gf.write(doc1_content)

    # 2. Corrupt/empty raw HTML
    doc2_content = b"<html><body></body></html>"
    sha2 = hashlib.sha256(doc2_content).hexdigest()
    file2 = raw_root / sha2[:2] / f"{sha2}.gz"
    file2.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(file2, "wb") as gf:
        gf.write(doc2_content)

    with registry.conn:
        registry.conn.execute(
            """
            UPDATE fetch_jobs
            SET state = 'fetched', raw_path = ?, raw_sha256 = ?, mime_type = 'text/html'
            WHERE job_id = ?
            """,
            (str(file1), sha1, hashlib.sha256("https://example.com/doc1".encode("utf-8")).hexdigest()),
        )
        registry.conn.execute(
            """
            UPDATE fetch_jobs
            SET state = 'fetched', raw_path = ?, raw_sha256 = ?, mime_type = 'text/html'
            WHERE job_id = ?
            """,
            (str(file2), sha2, hashlib.sha256("https://example.com/doc2".encode("utf-8")).hexdigest()),
        )

    report = run_batch_extraction(registry=registry, raw_root=raw_root)
    assert report.attempted == 2
    assert report.ready >= 1
    assert (report.quarantined + report.failed) >= 1

    registry.close()


def test_content_cache_import_double_extensions(tmp_path: Path):
    """import_content_cache resolves double extension files like doc_id.html.gz and companion json."""
    reg_path = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry(reg_path)

    sources = [
        OfficialSource(doc_id=888, url="https://example.com/double_ext_doc"),
    ]
    ingest_sources(sources, registry)

    raw_data = b"<html><body>Data from double extension cache</body></html>"
    gz_file = cache_dir / "888.html.gz"
    with gzip.open(gz_file, "wb") as gf:
        gf.write(raw_data)

    imported = import_content_cache(registry=registry, cache_dir=cache_dir, raw_dir=raw_root)
    assert imported == 1

    job_id = hashlib.sha256("https://example.com/double_ext_doc".encode("utf-8")).hexdigest()
    rec = registry.get_fetch_job(job_id)
    assert rec["state"] == "fetched"
    assert Path(rec["raw_path"]).exists()

    registry.close()
