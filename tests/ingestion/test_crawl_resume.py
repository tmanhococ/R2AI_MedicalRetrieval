import json
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
from pathlib import Path
import pytest
import httpx

from src.data.vibiomir_contracts import CrawlConfig, OfficialSource, CrawlReport
from src.data.vibiomir_registry import SourceRegistry, ingest_sources
from src.ingestion.crawl_runner import run_crawl


def test_expired_lease_resumes_on_next_lease_jobs(tmp_path: Path):
    """Expired lease returns to pending queue and is leased on the next lease_jobs call."""
    db_file = tmp_path / "registry.sqlite"
    registry = SourceRegistry(db_file)

    sources = [
        OfficialSource(doc_id=1, url="https://example.com/item1"),
        OfficialSource(doc_id=2, url="https://example.com/item2"),
    ]
    ingest_sources(sources, registry)

    t0 = datetime.now(timezone.utc)

    # 1. Lease both jobs with 60 second lease
    leased_1 = registry.lease_jobs(limit=10, lease_seconds=60, now=t0)
    assert len(leased_1) == 2
    assert leased_1[0].attempt == 1

    # 2. Immediately call lease_jobs at t0 + 10s: nothing should be available
    t_early = t0 + timedelta(seconds=10)
    leased_early = registry.lease_jobs(limit=10, lease_seconds=60, now=t_early)
    assert len(leased_early) == 0

    # 3. Simulate lease expiration at t0 + 65s without finish_fetch:
    t_expired = t0 + timedelta(seconds=65)
    leased_recovered = registry.lease_jobs(limit=10, lease_seconds=60, now=t_expired)
    assert len(leased_recovered) == 2
    # Attempt count is incremented on re-lease
    assert leased_recovered[0].attempt == 2
    assert leased_recovered[1].attempt == 2

    registry.close()


@pytest.mark.asyncio
async def test_one_url_two_ids_only_fetches_once(tmp_path: Path):
    """Two official source IDs referencing the exact same URL only trigger one HTTP fetch."""
    db_file = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    config = CrawlConfig(raw_root=raw_root)
    registry = SourceRegistry(db_file)

    shared_url = "https://example.com/medical/guideline"
    sources = [
        OfficialSource(doc_id=583, url=shared_url),
        OfficialSource(doc_id=4294967297, url=shared_url),
    ]
    ingest_sources(sources, registry)

    request_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        if "/robots.txt" in str(request.url):
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        request_count += 1
        return httpx.Response(
            200,
            content=b"<html><body><h1>Guideline Content</h1></body></html>",
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_crawl(registry, config, client=client)

    assert isinstance(report, CrawlReport)
    assert report.leased == 1
    assert report.fetched == 1
    assert report.failed == 0
    assert report.bytes_written > 0
    assert request_count == 1  # Crucial: only fetched once!

    # Check that both IDs in registry resolve to the same fetched job
    job_583 = registry.get_source(583)
    job_alias = registry.get_source(4294967297)
    assert job_583 is not None and job_alias is not None
    assert job_583.url == shared_url
    assert job_alias.url == shared_url

    # Job record itself is fetched
    job_id = hashlib.sha256(shared_url.encode("utf-8")).hexdigest()
    job_record = registry.get_fetch_job(job_id)
    assert job_record is not None
    assert job_record["state"] == "fetched"
    assert job_record["raw_path"] is not None

    registry.close()


@pytest.mark.asyncio
async def test_raw_saved_before_registry_commit_survives_crash(tmp_path: Path):
    """Pre-existing valid raw file on disk survives crash and is reused without corruption."""
    db_file = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    config = CrawlConfig(raw_root=raw_root)
    registry = SourceRegistry(db_file)

    url = "https://example.com/crashed_doc"
    sources = [OfficialSource(doc_id=10, url=url)]
    ingest_sources(sources, registry)

    content = b"<html><body>Data saved before crash</body></html>"
    sha = hashlib.sha256(content).hexdigest()

    # Pre-populate raw file on disk as if worker wrote it then crashed before finish_fetch
    target_dir = raw_root / sha[:2]
    target_dir.mkdir(parents=True, exist_ok=True)
    raw_file = target_dir / f"{sha}.gz"
    with gzip.open(raw_file, "wb") as f:
        f.write(content)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=content,
            headers={"Content-Type": "text/html"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_crawl(registry, config, client=client)

    assert report.fetched == 1

    # Verify registry now references the raw file and it is valid
    job_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
    job_record = registry.get_fetch_job(job_id)
    assert job_record is not None
    assert job_record["state"] == "fetched"
    assert job_record["raw_sha256"] == sha
    assert Path(job_record["raw_path"]).exists()

    with gzip.open(job_record["raw_path"], "rb") as f:
        assert f.read() == content

    registry.close()


@pytest.mark.asyncio
async def test_crawl_idempotency_no_refetch(tmp_path: Path):
    """Re-running crawl on already fetched registry makes zero requests."""
    db_file = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    config = CrawlConfig(raw_root=raw_root)
    registry = SourceRegistry(db_file)

    sources = [
        OfficialSource(doc_id=1, url="https://example.com/item1"),
        OfficialSource(doc_id=2, url="https://example.com/item2"),
    ]
    ingest_sources(sources, registry)

    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if "/robots.txt" in str(request.url):
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        call_count += 1
        return httpx.Response(200, content=b"content", headers={"Content-Type": "text/html"})

    # First run: fetches both
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report1 = await run_crawl(registry, config, client=client)

    assert report1.leased == 2
    assert report1.fetched == 2
    assert call_count == 2

    # Second run: does not re-fetch already fetched jobs
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report2 = await run_crawl(registry, config, client=client)

    assert report2.leased == 0
    assert report2.fetched == 0
    assert call_count == 2  # No new requests!

    registry.close()


@pytest.mark.asyncio
async def test_crawl_concurrency_and_domain_limits(tmp_path: Path):
    """Crawler respects concurrency and completes multiple domains."""
    db_file = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    config = CrawlConfig(
        raw_root=raw_root,
        concurrency=4,
        per_domain_concurrency=2,
        min_domain_interval_seconds=0.01,
    )
    registry = SourceRegistry(db_file)

    urls = [
        "https://domain1.com/a",
        "https://domain1.com/b",
        "https://domain2.com/c",
        "https://domain2.com/d",
    ]
    sources = [OfficialSource(doc_id=i + 1, url=u) for i, u in enumerate(urls)]
    ingest_sources(sources, registry)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"OK", headers={"Content-Type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_crawl(registry, config, client=client)

    assert report.leased == 4
    assert report.fetched == 4
    registry.close()


@pytest.mark.asyncio
async def test_crawl_respects_robots_txt(tmp_path: Path):
    """Crawler respects robots.txt disallow rules, setting state to blocked."""
    db_file = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    config = CrawlConfig(raw_root=raw_root, respect_robots=True)
    registry = SourceRegistry(db_file)

    sources = [
        OfficialSource(doc_id=1, url="https://example.com/blocked/page1"),
        OfficialSource(doc_id=2, url="https://example.com/allowed/page2"),
    ]
    ingest_sources(sources, registry)

    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if "/robots.txt" in url_str:
            return httpx.Response(
                200,
                content=b"User-agent: *\nDisallow: /blocked/\n",
                headers={"Content-Type": "text/plain"},
            )
        if "/blocked/" in url_str:
            return httpx.Response(200, content=b"Secret data")
        return httpx.Response(200, content=b"Public data", headers={"Content-Type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_crawl(registry, config, client=client)

    assert report.leased == 2
    assert report.fetched == 1
    assert report.blocked == 1

    job_blocked_id = hashlib.sha256("https://example.com/blocked/page1".encode("utf-8")).hexdigest()
    job_blocked = registry.get_fetch_job(job_blocked_id)
    assert job_blocked is not None
    assert job_blocked["state"] == "blocked"
    assert job_blocked["error_code"] == "robots_disallowed"

    registry.close()


@pytest.mark.asyncio
async def test_crawl_with_limit(tmp_path: Path):
    """Crawler respects the limit parameter, leasing and fetching only up to the limit."""
    db_file = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    config = CrawlConfig(raw_root=raw_root)
    registry = SourceRegistry(db_file)

    sources = [
        OfficialSource(doc_id=i, url=f"https://example.com/doc_{i}")
        for i in range(1, 6)
    ]
    ingest_sources(sources, registry)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"content", headers={"Content-Type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_crawl(registry, config, limit=2, client=client)

    assert report.leased == 2
    assert report.fetched == 2

    # Remaining 3 jobs should still be pending
    cursor = registry.conn.execute("SELECT count(*) FROM fetch_jobs WHERE state = 'pending'")
    assert cursor.fetchone()[0] == 3

    registry.close()


def test_sample_stratified_jobs_deterministic_and_balanced(tmp_path: Path):
    """sample_stratified_jobs selects balanced quotas across domain::type strata with seed 42."""
    db_file = tmp_path / "registry.sqlite"
    registry = SourceRegistry(db_file)

    # Create 3 domains with different URL structures:
    # d1: 10 html, 5 pdf
    # d2: 20 slug
    # d3: 5 html, 5 dynamic
    sources = []
    doc_id = 1
    for i in range(10):
        sources.append(OfficialSource(doc_id=doc_id, url=f"https://domain1.vn/article_{i}.html"))
        doc_id += 1
    for i in range(5):
        sources.append(OfficialSource(doc_id=doc_id, url=f"https://domain1.vn/doc_{i}.pdf"))
        doc_id += 1
    for i in range(20):
        sources.append(OfficialSource(doc_id=doc_id, url=f"https://domain2.vn/news/{i}/"))
        doc_id += 1
    for i in range(5):
        sources.append(OfficialSource(doc_id=doc_id, url=f"https://domain3.com/info_{i}.html"))
        doc_id += 1
    for i in range(5):
        sources.append(OfficialSource(doc_id=doc_id, url=f"https://domain3.com/search.php?id={i}"))
        doc_id += 1

    ingest_sources(sources, registry)

    persist_path = tmp_path / "pilot_sample_ids.json"
    sampled_1 = registry.sample_stratified_jobs(sample_size=10, seed=42, persist_path=persist_path)
    assert len(sampled_1) == 10
    assert persist_path.exists()

    with open(persist_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["seed"] == 42
    assert meta["target_sample_size"] == 10
    assert meta["actual_sample_size"] == 10
    assert len(meta["sampled_job_ids"]) == 10

    # Test seed determinism: repeating with seed 42 yields exact same list
    sampled_2 = registry.sample_stratified_jobs(sample_size=10, seed=42)
    assert sampled_1 == sampled_2

    # Verify that multiple strata are represented in the sample
    sample_urls = [registry.get_fetch_job(jid)["url"] for jid in sampled_1]
    domains_in_sample = {httpx.URL(u).host for u in sample_urls}
    assert len(domains_in_sample) >= 3  # All 3 domains represented!

    registry.close()


def test_lease_jobs_with_specific_job_ids(tmp_path: Path):
    """lease_jobs with job_ids parameter restricts leases strictly to the provided subset."""
    db_file = tmp_path / "registry.sqlite"
    registry = SourceRegistry(db_file)

    sources = [
        OfficialSource(doc_id=i, url=f"https://example.com/page_{i}")
        for i in range(1, 11)
    ]
    ingest_sources(sources, registry)

    all_jobs = [r[0] for r in registry.conn.execute("SELECT job_id FROM fetch_jobs ORDER BY rowid ASC").fetchall()]
    target_ids = [all_jobs[1], all_jobs[4], all_jobs[7]]

    leased = registry.lease_jobs(limit=10, job_ids=target_ids)
    assert len(leased) == 3
    leased_ids = {j.job_id for j in leased}
    assert leased_ids == set(target_ids)

    # Calling again yields empty since all target_ids are leased
    leased_again = registry.lease_jobs(limit=10, job_ids=target_ids)
    assert len(leased_again) == 0

    registry.close()


@pytest.mark.asyncio
async def test_crawl_with_pilot_job_ids(tmp_path: Path):
    """run_crawl accepts job_ids and crawls only that designated subset."""
    db_file = tmp_path / "registry.sqlite"
    raw_root = tmp_path / "raw"
    config = CrawlConfig(raw_root=raw_root)
    registry = SourceRegistry(db_file)

    sources = [
        OfficialSource(doc_id=i, url=f"https://example.com/item_{i}")
        for i in range(1, 6)
    ]
    ingest_sources(sources, registry)

    all_jobs = [r[0] for r in registry.conn.execute("SELECT job_id FROM fetch_jobs ORDER BY rowid ASC").fetchall()]
    chosen_ids = [all_jobs[0], all_jobs[2]]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"content", headers={"Content-Type": "text/html"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        report = await run_crawl(registry, config, job_ids=chosen_ids, client=client)

    assert report.leased == 2
    assert report.fetched == 2

    # Check states in registry
    assert registry.get_fetch_job(chosen_ids[0])["state"] == "fetched"
    assert registry.get_fetch_job(chosen_ids[1])["state"] == "fetched"
    assert registry.get_fetch_job(all_jobs[1])["state"] == "pending"

    registry.close()

