from datetime import datetime, timezone
import gzip
import hashlib
from pathlib import Path
import pytest
import httpx

from src.data.vibiomir_contracts import CrawlConfig, FetchJob, FetchResult
from src.ingestion.fetcher import fetch_one


@pytest.mark.asyncio
async def test_fetch_one_success_atomic_gzip_sha256(tmp_path: Path):
    """Successful fetch atomically stores gzipped body with verified SHA-256."""
    raw_root = tmp_path / "raw"
    config = CrawlConfig(raw_root=raw_root)
    body_content = b"<html><body><h1>Medical Article</h1><p>Clinical data</p></body></html>"
    expected_sha = hashlib.sha256(body_content).hexdigest()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body_content,
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    job = FetchJob(job_id="test_job_1", url="https://example.com/article1", attempt=1)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await fetch_one(job, client, config)

    assert isinstance(result, FetchResult)
    assert result.job_id == "test_job_1"
    assert result.state == "fetched"
    assert result.http_status == 200
    assert result.raw_sha256 == expected_sha
    assert result.raw_path is not None
    assert result.error_code is None
    assert result.mime_type is not None and "text/html" in result.mime_type

    # Verify physical file on disk
    raw_file = Path(result.raw_path)
    assert raw_file.exists()
    assert raw_file.suffix == ".gz"
    # Content must match exactly when decompressed
    with gzip.open(raw_file, "rb") as f:
        assert f.read() == body_content

    # Ensure no leftover temporary files in directory
    parent_dir = raw_file.parent
    tmp_files = list(parent_dir.glob("*.tmp*"))
    assert len(tmp_files) == 0


@pytest.mark.asyncio
async def test_fetch_one_404_410_terminates_failed_immediately(tmp_path: Path):
    """404 and 410 terminate immediately as fetch_failed without scheduling useless retries."""
    config = CrawlConfig(raw_root=tmp_path / "raw", max_attempts=3)

    def handler_404(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, content=b"Not Found")

    def handler_410(request: httpx.Request) -> httpx.Response:
        return httpx.Response(410, content=b"Gone")

    job1 = FetchJob(job_id="job_404", url="https://example.com/notfound", attempt=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_404)) as client:
        res1 = await fetch_one(job1, client, config)

    assert res1.state == "fetch_failed"
    assert res1.http_status == 404
    assert res1.error_code == "http_404"
    assert res1.raw_path is None
    assert getattr(res1, "retry_at", None) is None

    job2 = FetchJob(job_id="job_410", url="https://example.com/gone", attempt=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_410)) as client:
        res2 = await fetch_one(job2, client, config)

    assert res2.state == "fetch_failed"
    assert res2.http_status == 410
    assert res2.error_code == "http_410"
    assert res2.raw_path is None


@pytest.mark.asyncio
async def test_fetch_one_429_5xx_schedules_retry_with_backoff(tmp_path: Path):
    """429 and 5xx schedule retry with exponential backoff or Retry-After; terminates on max_attempts."""
    config = CrawlConfig(raw_root=tmp_path / "raw", max_attempts=3, base_backoff_seconds=2.0)

    # 1. 429 with explicit Retry-After: 7
    def handler_429(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "7"}, content=b"Rate limited")

    job_429 = FetchJob(job_id="job_429", url="https://example.com/throttled", attempt=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_429)) as client:
        res_429 = await fetch_one(job_429, client, config)

    assert res_429.state == "retry_wait"
    assert res_429.http_status == 429
    assert res_429.retry_at is not None

    retry_time = datetime.fromisoformat(res_429.retry_at)
    now_utc = datetime.now(timezone.utc)
    diff_sec = (retry_time - now_utc).total_seconds()
    assert 5 <= diff_sec <= 10  # roughly 7s ahead

    # 2. 503 without Retry-After (exponential backoff attempt 1 -> 2.0s, attempt 2 -> 4.0s)
    def handler_503(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"Service Unavailable")

    job_503_att1 = FetchJob(job_id="job_503", url="https://example.com/unavail", attempt=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_503)) as client:
        res_503_1 = await fetch_one(job_503_att1, client, config)

    assert res_503_1.state == "retry_wait"
    assert res_503_1.http_status == 503
    retry_1 = datetime.fromisoformat(res_503_1.retry_at)
    assert 1.0 <= (retry_1 - datetime.now(timezone.utc)).total_seconds() <= 3.0

    job_503_att2 = FetchJob(job_id="job_503", url="https://example.com/unavail", attempt=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_503)) as client:
        res_503_2 = await fetch_one(job_503_att2, client, config)

    assert res_503_2.state == "retry_wait"
    retry_2 = datetime.fromisoformat(res_503_2.retry_at)
    assert 3.0 <= (retry_2 - datetime.now(timezone.utc)).total_seconds() <= 5.5

    # 3. Attempt 3 with max_attempts=3 should exhaust retries -> fetch_failed
    job_503_att3 = FetchJob(job_id="job_503", url="https://example.com/unavail", attempt=3)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler_503)) as client:
        res_503_3 = await fetch_one(job_503_att3, client, config)

    assert res_503_3.state == "fetch_failed"
    assert res_503_3.http_status == 503


@pytest.mark.asyncio
async def test_fetch_one_total_deadline_enforced(tmp_path: Path):
    """Enforces total deadline around the fetch operation."""
    config = CrawlConfig(
        raw_root=tmp_path / "raw",
        total_deadline_seconds=0.05,
        max_attempts=3,
    )

    import asyncio

    async def slow_stream(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.2)
        return httpx.Response(200, content=b"Too slow")

    job = FetchJob(job_id="job_slow", url="https://example.com/slow", attempt=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(slow_stream)) as client:
        res = await fetch_one(job, client, config)

    assert res.state == "retry_wait"
    assert res.error_code == "timeout"
    assert res.raw_path is None


@pytest.mark.asyncio
async def test_fetch_one_decompressed_body_limit_enforced(tmp_path: Path):
    """Enforces decompressed body limit, aborting oversized downloads."""
    # HTML limit 100 bytes, response is 500 bytes
    config = CrawlConfig(raw_root=tmp_path / "raw", max_html_bytes=100)
    big_content = b"A" * 500

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=big_content,
            headers={"Content-Type": "text/html"},
        )

    job = FetchJob(job_id="job_big", url="https://example.com/big_html", attempt=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await fetch_one(job, client, config)

    assert res.state == "unsupported"
    assert res.error_code == "body_limit_exceeded"
    assert res.raw_path is None


@pytest.mark.asyncio
async def test_fetch_one_http_200_captcha_retains_raw_bytes(tmp_path: Path):
    """HTTP 200 with captcha challenge is classified as blocked, but raw bytes are retained."""
    config = CrawlConfig(raw_root=tmp_path / "raw")
    captcha_html = (
        b"<html><head><title>Just a moment...</title></head>"
        b"<body><div class=\"cf-browser-verification\">Checking your browser before accessing</div>"
        b"<div id=\"cf-challenge\">Please wait</div></body></html>"
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=captcha_html,
            headers={"Content-Type": "text/html; charset=utf-8"},
        )

    job = FetchJob(job_id="job_captcha", url="https://example.com/protected", attempt=1)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        res = await fetch_one(job, client, config)

    assert res.state == "blocked"
    assert res.error_code == "captcha_detected"
    assert res.raw_path is not None
    assert res.raw_sha256 == hashlib.sha256(captcha_html).hexdigest()

    # Raw bytes must be safely preserved on disk
    raw_file = Path(res.raw_path)
    assert raw_file.exists()
    with gzip.open(raw_file, "rb") as f:
        assert f.read() == captcha_html


@pytest.mark.asyncio
async def test_fetch_one_unsupported_scheme(tmp_path: Path):
    """Rejects non-http/https schemes as unsupported."""
    config = CrawlConfig(raw_root=tmp_path / "raw")
    job = FetchJob(job_id="job_ftp", url="ftp://ftp.example.com/file.txt", attempt=1)

    async with httpx.AsyncClient() as client:
        res = await fetch_one(job, client, config)

    assert res.state == "unsupported"
    assert res.error_code == "unsupported_scheme"
