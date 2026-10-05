# Task 11 Report: Pilot, Freeze Baseline A và Bàn giao Submission

## Summary of Implementation (including Fix Round 1)

Completed Task 11: Freezing baseline snapshot, running full 1,200 query inference, generating and validating official submission archive, and writing runbook & baseline report:

1. **Stratified Pilot Sampling & Crawl (`seed: 42`, 500 URLs)**:
   - Implemented `SourceRegistry.sample_stratified_jobs(sample_size=500, seed=42, persist_path=...)` using Largest Remainder Method across 178 `(domain, url_type)` strata.
   - Saved sample manifest to `artifacts/vibiomir/pilot_sample_ids.json`.
   - Wired `--pilot-size` and `--seed` into `src/cli/crawl.py` and `src/ingestion/crawl_runner.py`.
   - Executed live pilot crawl on all 500 sampled URLs:
     * Leased: 500
     * Successfully fetched: 383
     * Blocked by robots.txt: 93 (`robots_disallowed`)
     * Retry scheduled: 24 (transient timeouts)
     * Failed: 0
     * Raw data: 6.30 MiB compressed content-addressed payloads.
   - Total fetched sources in registry: 468 URLs.

2. **Snapshot Freezing (`artifacts/vibiomir/snapshots/phase1_A`)**:
   - Extracted 468 raw documents: 465 ready, 3 quarantined (`empty_text`), 0 failed.
   - Built immutable snapshot `phase1_A`: 465 clean ready documents (197 vi, 268 zh), 1,238 chunks (`body_max_tokens: 320`, `long_block_overlap: 64`).
   - Exported Parquet shards (`documents_00000.parquet`, `chunks_00000.parquet`), B-tree `lookup.sqlite`, and `manifest.json`.
   - Manifest verified: `source_membership_sha256: 42077515dcf25c4543c8302d335a81365edc8ff295b9f047fc150e212bcbfd8c`.

3. **Index Building (`artifacts/vibiomir/indexes/phase1_A`)**:
   - Built BM25S index across all 465 documents and 1,238 chunks using Vietnamese syllable tokenizer.
   - Built manifest and partition mappings.

4. **Inference & Submission Package Generation (`artifacts/experiments/vibiomir_phase1_A`)**:
   - Ran `run_predict` on official `data/official/query.parquet` (all 1,200 queries).
   - Applied fixed top-K policy (`documents: 5`, `chunks: 10`).
   - Completed all 1,200 queries with 0 inference errors in 26.05 seconds on CPU.
   - Measured Peak RAM: **10.07 MB** (`tracemalloc`), GPU VRAM: **0.0 MB**.
   - Produced transactional checkpoint `predictions_checkpoint.jsonl`, `predictions.jsonl`, `retrieval_debug.jsonl`, and `run_manifest.json` (`run_fingerprint: eea4619928b681a0d1fb007b4d78367d4d1541a23be96bc92827034e7aaa9182`).
   - Generated `coverage.json` in both `artifacts/experiments/vibiomir_phase1_A/` and `artifacts/vibiomir/raw/`.
   - Exported single-root `results.json` and generated `submission_A.zip` (SHA-256: `8ee757873b18ddec7775a333907f83778110766163a3ceddc2980e1c1a03e6ee`, 5.44 MB).
   - Ran strict `validate_submission`:
     * Valid: True
     * Query count: 1,200 / 1,200
     * Document count: 6,000 (5 per query)
     * Chunk count: 12,000 (10 per query)
     * Exact source provenance: 100% of the 12,000 chunks verified as exact substrings of their referenced documents (`doc.source_text[start:end] == chunk.text`).
     * Errors: 0, Warnings: 0.

5. **Qualitative Audit (30 Sources & 20 Chunks)**:
   - 30 diverse sources audited: 100% exact character span matches (`doc.source_text[start:end] == chunk.text`), 100% compliant with 320 token budget (max tokens: 312), 3 soft-errors quarantined, language classification validated.
   - 20 exported chunks audited across queries 1..1200: 100% exact substrings in parent documents, `chunk_order` 0..9 sequential, zero internal metadata leakage.

6. **Documentation**:
   - `docs/vibiomir/phase1_runbook.md`: Operator guide from ingestion to validation.
   - `docs/vibiomir/baseline_A.md`: Full acceptance report against the 12 gates in Section 14 of the plan, including explicit disclosure of Scorer Parity (`false`), peak RAM/GPU, and qualitative audit findings.

---

## TDD and Verification Evidence
- `uv run pytest tests/ingestion/test_crawl_resume.py -q`:
  `10 passed in 4.41s` (including new tests for stratified sampling, leased job IDs, and pilot crawl).
- `uv run python -m src.cli.validate_submission --config configs/vibiomir_phase1.yaml --snapshot artifacts/vibiomir/snapshots/phase1_A --zip artifacts/experiments/vibiomir_phase1_A/submission_A.zip`:
  ```text
  === ViBioMIR Submission Validation ===
  Snapshot:  artifacts\vibiomir\snapshots\phase1_A
  ZIP file:  artifacts\experiments\vibiomir_phase1_A\submission_A.zip
  Queries:   data\official\query.parquet

  === Validation Report ===
  Valid:           True
  Query Count:     1200
  Document Count:  6000
  Chunk Count:     12000
  Errors:          0
  Warnings:        0

  All checks passed successfully!
  ```
- Full Regression Test Suite: `uv run pytest -q`:
  `354 passed in 15.30s` (0 failures, 0 regressions).
