# Task 1 Report: Official inputs, ID contracts và registry import

**Date:** 2026-10-04  
**Author:** Subagent (Task 1)  
**Branch:** `feature/vibiomir-phase-1`  
**Commit:** `61e67196751976df426368851395704b8ba7b42c` (`feat: add ViBioMIR official inputs and source registry (Task 1)`)

---

## 1. Summary of Implementation

Implemented the official data contracts, Parquet streaming adapters, and SQLite staging registry for ViBioMIR Phase 1 according to Section 4.1 & 4.2 of the implementation plan:

1. **`src/data/vibiomir_contracts.py`**:
   - Defined frozen dataclasses with validation and unsigned/signed int64 guards: `OfficialSource`, `OfficialQuery`, `FetchJob`, `FetchResult`, `ExtractedBlock`, `ExtractedDocument`, `SpanChunk`, `ReferenceChunk`, `DevQueryMetadata`, `EvaluationRecord`, `LevelMetrics`, `QueryMetrics`, `IngestReport`, `CrawlReport`, `ExtractionReport`, `SnapshotManifest`, `ValidationReport`, `EvaluationSummary`, `RetrievalTrace`.
   - Defined official submission schemas `SubmissionChunk` and `SubmissionRow` with `NotRequired[int]` for optional `chunk_order`.
   - Defined validated configuration dataclasses: `CrawlConfig`, `ExtractionConfig`, `ChunkConfig`, `PairPackingConfig`, `SnapshotConfig`, `EvaluationProfile`.
   - Defined runtime checkable protocol `SourceStore`.

2. **`src/data/vibiomir_registry.py`**:
   - Implemented `SourceRegistry` backed by SQLite with WAL mode (`PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL; PRAGMA foreign_keys = ON;`).
   - Managed tables: `sources` (INTEGER PRIMARY KEY `doc_id`, `original_url`, `fetch_job_id`), `fetch_jobs` (`job_id` PRIMARY KEY, SHA256 of trimmed URL), and `ingest_state` (`input_sha256` PRIMARY KEY, `last_committed_row`, `complete`, `updated_at`).
   - Implemented `ingest_sources(rows, registry, input_sha256, batch_size)` with atomic transactions, resume cursor (`last_committed_row`), duplicate ID rejection without upsert masking, and idempotent re-execution upon completion.

3. **`src/data/vibiomir_adapter.py`**:
   - Implemented `iter_official_sources(path, batch_size)` using `pyarrow.parquet.ParquetFile.iter_batches()`, strictly validating schema, column types (signed int64, string), rejecting boolean/float/null values, and exposing `.input_sha256`.
   - Implemented `load_official_queries(path, expected_count)` reading `query.parquet`, checking exact signed int64 IDs, non-empty queries, uniqueness, and validating against `expected_count`.

4. **`configs/vibiomir_phase1.yaml`**:
   - Created the configuration overlay for ViBioMIR Phase 1 baseline A with crawl, extraction, chunking, reranking window, evaluation profile, and submission settings.

---

## 2. TDD Evidence

### Step 2.1: RED Phase (Failing Test)
- **Command:** `uv run pytest tests/data/test_vibiomir_inputs.py -q`
- **Output:**
```
=================================== ERRORS ====================================
_____________ ERROR collecting tests/data/test_vibiomir_inputs.py _____________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\data\test_vibiomir_inputs.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\data\test_vibiomir_inputs.py:7: in <module>
    from src.data.vibiomir_contracts import (
E   ModuleNotFoundError: No module named 'src.data.vibiomir_contracts'
=========================== short test summary info ===========================
ERROR tests/data/test_vibiomir_inputs.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.36s
```

### Step 2.2: GREEN Phase (Passing Tests)
- **Command:** `uv run pytest tests/data/test_vibiomir_inputs.py -q`
- **Output:**
```
.....                                                                    [100%]
5 passed in 0.55s
```

### Test Coverage Highlights:
- `test_sparse_int64_ids_and_alias_urls`: Confirms doc_id 583 and doc_id 4294967297 (> 2^31) are preserved; two IDs sharing the same URL produce two distinct rows in `sources` and exactly 1 shared `fetch_job`.
- `test_reject_bool_float_null_and_duplicate_ids`: Confirms rejection of boolean, float, null IDs, null/whitespace URLs, and transactional failure/rollback on duplicate IDs across batch boundaries without upsert masking (`is_complete() == False`).
- `test_query_exact_set_and_count`: Validates signed int64 IDs, non-empty queries, exact count matching, and rejection of mismatched expected count or duplicate query IDs.
- `test_ingest_resume_idempotent`: Validates resume from intermediate cursor (`last_committed_row = 4`), completion of remaining rows, idempotent re-runs on completed input, and rejection of changed checksums on the same registry.
- `test_contracts_and_unit_validations`: Direct unit coverage of dataclass field validators, signed int64 bounds, and `SourceRegistry` helper methods.

---

## 3. Regression Suite Verification

- **Command:** `uv run pytest -q`
- **Exit Code:** `0`
- **Result:** `206 passed, 1 warning in 16.97s`
- **Regressions:** `0` regressions (201 baseline tests + 5 new tests = 206 passed).

---

## 4. Files Created / Modified

- `src/data/vibiomir_contracts.py` (Created, 319 lines)
- `src/data/vibiomir_registry.py` (Created, 295 lines)
- `src/data/vibiomir_adapter.py` (Created, 149 lines)
- `configs/vibiomir_phase1.yaml` (Created, 87 lines)
- `tests/data/test_vibiomir_inputs.py` (Created, 315 lines)

---

## 5. Commit Metadata

- **Commit SHA:** `61e67196751976df426368851395704b8ba7b42c`
- **Short SHA:** `61e6719`
- **Subject:** `feat: add ViBioMIR official inputs and source registry (Task 1)`

---

## 6. Self-Review Checklist

- [x] All contracts match Section 4.1 & 4.2 specs.
- [x] Signed int64 IDs preserved and bounds checked.
- [x] Aliased URLs sharing doc_ids produce multiple source records and a single deduplicated fetch job.
- [x] Duplicate doc_ids fail transactionally without being masked by upsert (`is_complete()` remains `False`).
- [x] Ingestion cursor supports checkpoint resume and idempotent re-execution.
- [x] Full regression test suite executed with 0 regressions.
- [x] Code is clean, documented, and committed to git branch `feature/vibiomir-phase-1`.
