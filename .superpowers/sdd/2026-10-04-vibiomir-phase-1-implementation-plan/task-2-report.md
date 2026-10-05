# Task 2 Report: Fetch scheduler, raw store và resume

## Summary of Implementation
Implemented the resumable crawler subsystem for ViBioMIR Phase 1 according to Section 5 of the implementation plan and the Task 2 brief:
1. **SourceRegistry extensions** (`src/data/vibiomir_registry.py`):
   - `lease_jobs(limit, lease_seconds, max_attempts, now)`: Leases pending jobs in batches matching worker capacity, automatically reclaiming expired `fetching` leases and releasing eligible `retry_wait` jobs whose backoff timer has elapsed.
   - `finish_fetch(result, now)`: Atomically updates fetch job states, HTTP statuses, raw storage paths, checksums, and error codes.
   - `get_fetch_job(job_id)`: Inspection and assertion helper for fetch job records.
   - Updated table schema with `retry_at`, `mime_type`, and `final_url` columns.
2. **Contracts** (`src/data/vibiomir_contracts.py`):
   - Added `retry_at` and `bytes_written` to `FetchResult`.
   - Updated `CrawlReport` dataclass with fields `leased`, `fetched`, `retry_wait`, `failed`, `blocked`, `unsupported`, and `bytes_written`, keeping backward-compatibility properties.
3. **Atomic Raw Fetcher** (`src/ingestion/fetcher.py`):
   - `fetch_one(job, client, config)`: Enforces `total_deadline_seconds` via `asyncio.timeout`, streams response bodies while enforcing decompressed body limits (`max_html_bytes` vs `max_pdf_bytes`).
   - Content-addressed atomic storage (`save_raw_atomic`): Writes gzipped content to `.tmp` file, verifies decompressed SHA-256 integrity, and renames atomically to `<raw_root>/<sha256[:2]>/<sha256>.gz`. Existing intact files are reused without corruption.
   - Status & retry handling: Immediately terminates 404/410 as `fetch_failed` without retries; applies exponential backoff or respects `Retry-After` header for 429/5xx and network/timeout errors; classifies 403 as `blocked`; detects HTTP 200 captcha challenges and classifies as `blocked` while retaining raw bytes.
4. **Crawl Runner** (`src/ingestion/crawl_runner.py`):
   - `run_crawl(registry, config, limit, client)`: Coordinates worker tasks using global concurrency semaphore (`asyncio.Semaphore(concurrency)`), per-domain semaphore (`per_domain_concurrency`), domain interval throttler (`min_domain_interval_seconds`), and domain-cached robots.txt evaluation (`RobotsManager`).
   - Supports passing mock clients for tests and managed httpx client in production.

---

## TDD Evidence

### RED Test Phase
Command:
```bash
uv run pytest tests/ingestion/test_fetcher.py tests/ingestion/test_crawl_resume.py -q
```
Output:
```
=================================== ERRORS ====================================
______________ ERROR collecting tests/ingestion/test_fetcher.py _______________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\ingestion\test_fetcher.py'.
tests\ingestion\test_fetcher.py:9: in <module>
    from src.ingestion.fetcher import fetch_one
E   ModuleNotFoundError: No module named 'src.ingestion'
____________ ERROR collecting tests/ingestion/test_crawl_resume.py ____________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\ingestion\test_crawl_resume.py'.
tests\ingestion\test_crawl_resume.py:10: in <module>
    from src.ingestion.crawl_runner import run_crawl
E   ModuleNotFoundError: No module named 'src.ingestion'
=========================== short test summary info ===========================
ERROR tests/ingestion/test_fetcher.py
ERROR tests/ingestion/test_crawl_resume.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
2 errors in 0.21s
```

### GREEN Test Phase
Command:
```bash
uv run pytest tests/ingestion/test_fetcher.py tests/ingestion/test_crawl_resume.py -q
```
Output:
```
..............                                                           [100%]
14 passed in 3.67s
```

---

## Full Regression Test Suite
Command:
```bash
uv run pytest -q
```
Output:
```
........................................................................ [ 32%]
........................................................................ [ 65%]
........................................................................ [ 98%]
....                                                                     [100%]
============================== warnings summary ===============================
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\site-packages\faiss\loader.py:49
  C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\site-packages\faiss\loader.py:49: DeprecationWarning: numpy.core._multiarray_umath is deprecated and has been renamed to numpy._core._multiarray_umath. The numpy._core namespace contains private NumPy internals and its use is discouraged, as NumPy internals can change without warning in any release. In practice, most real-world usage of numpy.core is to access functionality in the public NumPy API. If that is the case, use the public NumPy API. If not, you are using NumPy internals. If you would still like to access an internal attribute, use numpy._core._multiarray_umath.__cpu_features__.
    from numpy.core._multiarray_umath import __cpu_features__

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
220 passed, 1 warning in 16.43s
```
Result: 220 passed (206 baseline + 14 new tests), 0 failures, 0 regressions.

---

## Files Created / Modified
- `src/data/vibiomir_contracts.py` (modified): Added `retry_at` and `bytes_written` to `FetchResult`; updated `CrawlReport` fields.
- `src/data/vibiomir_registry.py` (modified): Implemented `lease_jobs`, `finish_fetch`, `get_fetch_job`, and schema migration.
- `src/ingestion/__init__.py` (created): Package initialization.
- `src/ingestion/fetcher.py` (created): Atomic gzip raw store, deadline wrapper, streaming limits, captcha detection, and retry rules.
- `src/ingestion/crawl_runner.py` (created): Concurrency semaphores, domain interval throttler, robots.txt manager, and crawl orchestration.
- `tests/ingestion/test_fetcher.py` (created): Tests for unit fetcher rules (404/410, 429/5xx, timeouts, body limits, captcha raw retention).
- `tests/ingestion/test_crawl_resume.py` (created): Tests for multi-doc single-fetch, lease expiration recovery, crash survival, idempotency, robots, and limit.

---

## Git Commit
- **SHA**: `08c1fd5`
- **Subject**: `feat: add resumable source crawler (Task 2)`

---

## Self-Review Checklist
- [x] Strictly followed TDD: recorded RED before implementation, verified GREEN after.
- [x] Zero real network requests during testing (`httpx.MockTransport` used).
- [x] Atomic write with gzip and verified SHA-256 before registry commit.
- [x] Terminal states for 404/410 without useless retries.
- [x] Exponential backoff / Retry-After handling for 429/5xx.
- [x] Pre-existing raw files survive crash without corruption.
- [x] One URL shared by multiple IDs fetched only once.
- [x] Full regression test suite passing with 0 regressions.
