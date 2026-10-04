import asyncio
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
from pathlib import Path
from urllib.parse import urlparse
import uuid

import httpx

from src.data.vibiomir_contracts import CrawlConfig, FetchJob, FetchResult

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


def _is_captcha(body_bytes: bytes, mime_type: str | None) -> bool:
    if mime_type and "pdf" in mime_type.lower():
        return False
    lower_bytes = body_bytes.lower()
    return any(pattern in lower_bytes for pattern in CAPTCHA_PATTERNS)


def save_raw_atomic(raw_root: Path, body_bytes: bytes) -> tuple[str, str, int]:
    """Saves raw body bytes as gzip with SHA-256 content addressing.

    Returns (raw_path, raw_sha256, bytes_written).
    If an intact file already exists on disk, reuses it with bytes_written=0.
    """
    raw_sha256 = hashlib.sha256(body_bytes).hexdigest()
    prefix = raw_sha256[:2]
    target_dir = raw_root / prefix
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{raw_sha256}.gz"

    # Reuse if already exists and matches hash
    if target_path.exists():
        try:
            with gzip.open(target_path, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() == raw_sha256:
                    return str(target_path), raw_sha256, 0
        except Exception:
            pass

    tmp_path = target_dir / f"{raw_sha256}.tmp_{uuid.uuid4().hex}.gz"
    try:
        with gzip.open(tmp_path, "wb") as f:
            f.write(body_bytes)
        with gzip.open(tmp_path, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() != raw_sha256:
                raise ValueError(f"Integrity check failed for {target_path}")
        tmp_path.replace(target_path)
        bytes_written = target_path.stat().st_size
        return str(target_path), raw_sha256, bytes_written
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _calculate_retry_at(attempt: int, config: CrawlConfig, retry_after_header: str | None = None) -> str:
    """Calculates ISO UTC timestamp for next retry attempt."""
    wait_seconds: float = 0.0
    if retry_after_header:
        try:
            wait_seconds = float(retry_after_header.strip())
        except ValueError:
            wait_seconds = config.base_backoff_seconds * (2 ** max(0, attempt - 1))
    else:
        wait_seconds = config.base_backoff_seconds * (2 ** max(0, attempt - 1))

    return (datetime.now(timezone.utc) + timedelta(seconds=max(0.0, wait_seconds))).isoformat()


async def fetch_one(job: FetchJob, client: httpx.AsyncClient, config: CrawlConfig) -> FetchResult:
    """Fetches a single FetchJob according to crawl specification.

    Enforces total deadline, body size limits, atomic gzip write, captcha detection,
    and retry state machine.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    parsed = urlparse(job.url)
    if parsed.scheme.lower() not in ("http", "https"):
        return FetchResult(
            job_id=job.job_id,
            state="unsupported",
            raw_path=None,
            raw_sha256=None,
            final_url=job.url,
            mime_type=None,
            http_status=None,
            fetched_at=now_iso,
            error_code="unsupported_scheme",
            retry_at=None,
            bytes_written=0,
        )

    try:
        async with asyncio.timeout(config.total_deadline_seconds):
            headers = {"User-Agent": "ViBioMIR-Crawler/1.0"}
            async with client.stream(
                "GET",
                job.url,
                follow_redirects=True,
                headers=headers,
            ) as response:
                status = response.status_code
                final_url = str(response.url)
                mime_type = response.headers.get("content-type")

                # Terminal non-retryable 404 / 410 errors
                if status in (404, 410):
                    return FetchResult(
                        job_id=job.job_id,
                        state="fetch_failed",
                        raw_path=None,
                        raw_sha256=None,
                        final_url=final_url,
                        mime_type=mime_type,
                        http_status=status,
                        fetched_at=now_iso,
                        error_code=f"http_{status}",
                        retry_at=None,
                        bytes_written=0,
                    )

                # Retryable HTTP status codes: 408, 429, 5xx
                if status in (408, 429) or (500 <= status <= 599):
                    if job.attempt < config.max_attempts:
                        retry_after = response.headers.get("retry-after")
                        retry_at = _calculate_retry_at(job.attempt, config, retry_after)
                        return FetchResult(
                            job_id=job.job_id,
                            state="retry_wait",
                            raw_path=None,
                            raw_sha256=None,
                            final_url=final_url,
                            mime_type=mime_type,
                            http_status=status,
                            fetched_at=now_iso,
                            error_code=f"http_{status}",
                            retry_at=retry_at,
                            bytes_written=0,
                        )
                    else:
                        return FetchResult(
                            job_id=job.job_id,
                            state="fetch_failed",
                            raw_path=None,
                            raw_sha256=None,
                            final_url=final_url,
                            mime_type=mime_type,
                            http_status=status,
                            fetched_at=now_iso,
                            error_code=f"http_{status}",
                            retry_at=None,
                            bytes_written=0,
                        )

                # Blocked: 403 Forbidden
                if status == 403:
                    return FetchResult(
                        job_id=job.job_id,
                        state="blocked",
                        raw_path=None,
                        raw_sha256=None,
                        final_url=final_url,
                        mime_type=mime_type,
                        http_status=status,
                        fetched_at=now_iso,
                        error_code="http_403",
                        retry_at=None,
                        bytes_written=0,
                    )

                # Other 4xx client errors
                if 400 <= status <= 499:
                    return FetchResult(
                        job_id=job.job_id,
                        state="fetch_failed",
                        raw_path=None,
                        raw_sha256=None,
                        final_url=final_url,
                        mime_type=mime_type,
                        http_status=status,
                        fetched_at=now_iso,
                        error_code=f"http_{status}",
                        retry_at=None,
                        bytes_written=0,
                    )

                # Stream response body and enforce decompressed body limit
                is_pdf = mime_type is not None and "application/pdf" in mime_type.lower()
                max_bytes = config.max_pdf_bytes if is_pdf else config.max_html_bytes

                chunks: list[bytes] = []
                total_bytes = 0
                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > max_bytes:
                        return FetchResult(
                            job_id=job.job_id,
                            state="unsupported",
                            raw_path=None,
                            raw_sha256=None,
                            final_url=final_url,
                            mime_type=mime_type,
                            http_status=status,
                            fetched_at=now_iso,
                            error_code="body_limit_exceeded",
                            retry_at=None,
                            bytes_written=0,
                        )
                    chunks.append(chunk)

                body_bytes = b"".join(chunks)

                # Check for captcha challenge
                is_captcha_detected = _is_captcha(body_bytes, mime_type)

                # Atomically store raw bytes (gzip + SHA-256)
                raw_path, raw_sha256, bytes_written = save_raw_atomic(config.raw_root, body_bytes)

                if is_captcha_detected:
                    return FetchResult(
                        job_id=job.job_id,
                        state="blocked",
                        raw_path=raw_path,
                        raw_sha256=raw_sha256,
                        final_url=final_url,
                        mime_type=mime_type,
                        http_status=status,
                        fetched_at=now_iso,
                        error_code="captcha_detected",
                        retry_at=None,
                        bytes_written=bytes_written,
                    )

                return FetchResult(
                    job_id=job.job_id,
                    state="fetched",
                    raw_path=raw_path,
                    raw_sha256=raw_sha256,
                    final_url=final_url,
                    mime_type=mime_type,
                    http_status=status,
                    fetched_at=now_iso,
                    error_code=None,
                    retry_at=None,
                    bytes_written=bytes_written,
                )

    except (TimeoutError, asyncio.TimeoutError, httpx.TimeoutException):
        if job.attempt < config.max_attempts:
            retry_at = _calculate_retry_at(job.attempt, config)
            return FetchResult(
                job_id=job.job_id,
                state="retry_wait",
                raw_path=None,
                raw_sha256=None,
                final_url=job.url,
                mime_type=None,
                http_status=None,
                fetched_at=now_iso,
                error_code="timeout",
                retry_at=retry_at,
                bytes_written=0,
            )
        else:
            return FetchResult(
                job_id=job.job_id,
                state="fetch_failed",
                raw_path=None,
                raw_sha256=None,
                final_url=job.url,
                mime_type=None,
                http_status=None,
                fetched_at=now_iso,
                error_code="timeout",
                retry_at=None,
                bytes_written=0,
            )

    except (httpx.NetworkError, httpx.TransportError) as exc:
        if job.attempt < config.max_attempts:
            retry_at = _calculate_retry_at(job.attempt, config)
            return FetchResult(
                job_id=job.job_id,
                state="retry_wait",
                raw_path=None,
                raw_sha256=None,
                final_url=job.url,
                mime_type=None,
                http_status=None,
                fetched_at=now_iso,
                error_code=f"network_error:{type(exc).__name__}",
                retry_at=retry_at,
                bytes_written=0,
            )
        else:
            return FetchResult(
                job_id=job.job_id,
                state="fetch_failed",
                raw_path=None,
                raw_sha256=None,
                final_url=job.url,
                mime_type=None,
                http_status=None,
                fetched_at=now_iso,
                error_code=f"network_error:{type(exc).__name__}",
                retry_at=None,
                bytes_written=0,
            )

    except Exception as exc:
        return FetchResult(
            job_id=job.job_id,
            state="fetch_failed",
            raw_path=None,
            raw_sha256=None,
            final_url=job.url,
            mime_type=None,
            http_status=None,
            fetched_at=now_iso,
            error_code=f"unexpected_error:{type(exc).__name__}",
            retry_at=None,
            bytes_written=0,
        )
