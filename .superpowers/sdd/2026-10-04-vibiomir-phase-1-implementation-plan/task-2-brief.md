### Task 2: Fetch scheduler, raw store và resume

**Files:** Create ingestion/fetcher.py, crawl_runner.py; extend registry; tests/ingestion/test_crawl_resume.py, test_fetcher.py.

**Interfaces:**

- SourceRegistry.lease_jobs(limit: int, lease_seconds: int = 120) -> list[FetchJob]
- SourceRegistry.finish_fetch(result: FetchResult) -> None
- async fetch_one(job: FetchJob, client: httpx.AsyncClient, config: CrawlConfig) -> FetchResult
- async run_crawl(registry: SourceRegistry, config: CrawlConfig, limit: int | None) -> CrawlReport
- CrawlReport: leased, fetched, retry_wait, failed, blocked, unsupported, bytes_written.

- [ ] Viết tests dùng fake HTTP transport + fake clock: one URL/two IDs fetch once; expired lease resumes; raw saved before registry commit survives crash; 429 schedules retry; 404 terminates; total deadline and decompressed body limits; HTTP 200 captcha raw retained.
- [ ] Run: pytest tests/ingestion/test_fetcher.py tests/ingestion/test_crawl_resume.py -q. Expected: FAIL thiếu scheduler/constraints.
- [ ] Implement semaphore global/domain, per-host timing, retry states và atomic content-addressed raw write. Respect robots; tests dùng stub để không gọi network thật.
- [ ] Kiểm tra idempotency bằng chạy cùng queue hai lần: fetched jobs không fetch lại nếu không có explicit refresh snapshot.
- [ ] Run tests. Expected: PASS, không trùng official rows/objects, retry counters đúng.
- [ ] Commit: feat: add resumable source crawler.

