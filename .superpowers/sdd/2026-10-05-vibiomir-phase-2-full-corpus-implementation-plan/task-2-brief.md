### Task 2: Full acquisition campaign và resume

**Files:** full_campaign.py, crawl runner/registry integration, cli/crawl_full.py; tests/ingestion/test_full_campaign.py.

**Interfaces:** plan_full_campaign → campaign_id; existing crawler APIs; content cache importer contract nhận explicit Path, official registry mapping và raw/source provenance.

- [ ] Tests reuse pilot content; all official IDs served including minor domains/EN/ZH; same URL aliases preserved; limit implies partial; crash leases recover; no domain blacklist masquerading as completion.
- [ ] Run pytest tests/ingestion/test_full_campaign.py -q; expected FAIL.
- [ ] Implement full queue, fair scheduling, content cache import when supplied, batch extraction/progress ledger. Preserve rate/robots; verify import bytes/source version before mark ready.
- [ ] Full-mode command runs to no pending/retry_wait, hoặc exits interrupted with resume state; blocked/failure statuses recorded, không count needs_review là completed.
- [ ] Run tests + crawl 5k scale smoke; inspect throughput by host and disk estimates. Commit: feat: run acquisition across the complete source registry.

