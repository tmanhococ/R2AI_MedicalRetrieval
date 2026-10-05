# Task 3 Report: Full Snapshot và Source-Store Batch I/O

## 1. Summary of Implementation

Implemented Task 3 of ViBioMIR Phase 2, focusing on scalable batch text access, frozen snapshot provenance with separate gap tracking, and support for Phase 2 evidence span resolution:

1. **`src/data/source_store.py`**:
   - **`LRUCache[K, V]`**:
     - Configurable bounded cache enforcing both entry count (`max_entries`) and total memory in bytes (`max_bytes`, default 256 MiB).
     - Evicts least-recently-used items when either limit is exceeded.
     - Provided tailored byte estimators for `ExtractedDocument` and `SpanChunk`.
   - **`SnapshotSourceStore`**:
     - Implemented `get_documents(doc_ids: Sequence[int]) -> list[ExtractedDocument]`:
       - Checks `_doc_cache` first.
       - Missing doc_ids are queried from `lookup.sqlite` (`document_locations`) and grouped by parquet shard so each shard is read at most once per batch (`filters=[("doc_id", "in", missing_in_shard)]`).
       - Loads, parses, caches in `LRUCache`, and returns results strictly matching the requested input order.
     - Implemented `get_chunks(chunk_ids: Sequence[str]) -> list[SpanChunk]`:
       - Checks `_chunk_cache` first.
       - Missing chunk_ids are queried from `lookup.sqlite` (`chunk_locations`) and grouped by parquet shard, read via PyArrow predicate pushdown once per shard, cached in bounded LRU, and returned in input order.
     - Implemented `iter_chunks_for_doc(doc_id: int) -> Iterator[SpanChunk]`:
       - Queries chunk IDs for `doc_id` from `lookup.sqlite` and yields `SpanChunk` objects strictly in ascending `chunk_index` order.
     - Updated `_build_missing_lookup_index` to index `chunk_index` and add an index on `doc_id` for fast random doc chunk lookups.
   - **`EvidenceSourceStore`**:
     - Implemented wrapper adhering to `SourceStore` protocol (`get_document`, `get_chunk`, `contains_official_doc`).
     - Supports `get_documents`, `get_chunks`, `iter_chunks_for_doc`.
     - Supports registration of `EvidenceSpan` instances and resolving dynamic `span:{doc_id}:{start}:{end}` patterns by slicing `source_text[start:end]` from underlying documents.
     - Batched `get_chunks` delegates regular chunk IDs to underlying store while resolving evidence spans seamlessly.

2. **`src/ingestion/snapshot.py`**:
   - **Streaming manifest membership**:
     - Replaced in-memory accumulation of all doc IDs with streaming SHA-256 updates (`membership_hasher.update(f"{doc_id}\n".encode("utf-8"))`), ensuring bounded memory footprint across millions of documents.
   - **Preserve usable `und` language content**:
     - Content classified as `und` is retained as ready if it does not exhibit hard errors (`captcha`, `soft_404`, `empty_text`, `extraction_failed`, `needs_ocr`, `unsupported_format`, `unsupported_mime_type`), preventing unwarranted quarantine of valid medical texts.
   - **Alias `doc_id` deduplication and preservation**:
     - When multiple official `doc_id`s share the same URL and fetch job, each official ID receives its own extracted document row and corresponding chunks with exact ID prefixes (`f"{doc_id}:..."`).
     - Added in-memory extraction caching by `(raw_sha256, mime_type)` to avoid repeated HTML/PDF parsing for identical content.
   - **Separate terminal gap and scope recording**:
     - Non-ready sources are partitioned and exported to `source_gaps.parquet` and `gaps.json`.
     - Automatically labels snapshot scope as `SCOPE_FULL_BODY_INDEX` (no gaps), `SCOPE_FULL_SOURCE_PASS_WITH_GAPS` (terminal gaps present), or `SCOPE_PARTIAL_BODY_INDEX` (unresolved jobs present).

3. **`src/data/vibiomir_contracts.py` & `src/data/vibiomir_adapter.py`**:
   - Added optional `source_scope: str | None = None` and `gap_count: int = 0` to `SnapshotManifest`.
   - Updated `ViBioMIRAdapter` to read `source_scope` and `gap_count` when loading snapshot manifests.

---

## 2. TDD Evidence: RED and GREEN

### RED Phase
- Created test suite `tests/data/test_full_snapshot.py` with 8 targeted tests covering batch document/chunk lookups, ordered chunk iteration, `EvidenceSourceStore`, usable `und` language preservation, alias doc IDs, gap recording, and LRU eviction.
- Ran `pytest tests/data/test_full_snapshot.py -q`.
- **Result:** FAILED (ImportError: cannot import name 'EvidenceSourceStore' from 'src.data.source_store').

### GREEN Phase
- Implemented `EvidenceSourceStore`, `LRUCache`, batch methods in `SnapshotSourceStore`, and streaming/gap/und logic in `build_snapshot`.
- Ran `pytest tests/data/test_full_snapshot.py -v`.
- **Result:** 8 passed in 4.73s.

---

## 3. Regression Testing

- Ran `pytest tests/data/test_snapshot_adapter.py -v`: 11 passed in 3.98s.
- Ran `pytest tests/operations/ -v`: 24 passed in 26.70s.
- Ran full test suite `pytest -q`: **407 passed, 3 warnings in 67.81s** (0 failures, 100% pass rate).

---

## 4. Files Changed

- `src/data/source_store.py`: Added `LRUCache`, batched `get_documents`, `get_chunks`, `iter_chunks_for_doc`, and `EvidenceSourceStore`.
- `src/ingestion/snapshot.py`: Added streaming manifest hashing, usable `und` language preservation, extraction caching for alias IDs, `source_gaps.parquet`, and `gaps.json` output.
- `src/data/vibiomir_contracts.py`: Added `source_scope` and `gap_count` fields to `SnapshotManifest`.
- `src/data/vibiomir_adapter.py`: Read `source_scope` and `gap_count` in `ViBioMIRAdapter`.
- `tests/data/test_full_snapshot.py`: 8 comprehensive test cases for Task 3 functionality.

---

## 5. Self-Review Findings

- **Order preservation:** Verified that `get_documents` and `get_chunks` retain exact duplicate and arbitrary ordering requested by caller.
- **Shard grouping:** Confirmed that PyArrow reads each parquet shard at most once per batch lookup.
- **LRU bounds:** Verified both entry-count and byte-size eviction mechanics under memory pressure.
- **Backwards compatibility:** Verified existing Phase 1 tests (`test_snapshot_adapter.py`, `test_vibiomir_export.py`, `test_full_scope.py`) continue to pass without modification.

---

- **Initial Commit SHA:** `b4e1b03`
- **Initial Commit Message:** `feat: freeze complete source-pass snapshots with batched text access`

---

## 7. Review Fixes (Post-Review Iteration)

### Critical & Important Fixes Addressed:
1. **Token Budget Validation and Deterministic Re-chunking Prior to Freeze (`src/ingestion/snapshot.py`)**:
   - Added `_validate_and_rechunk_spans` and `_rechunk_oversized_span` in `build_snapshot`.
   - Prior to persisting chunks to Parquet, validates `dense_token_count <= chk_cfg.body_max_tokens` and `reranker_token_count <= chk_cfg.body_max_tokens`.
   - Subdivides any oversized span deterministically using tokenizer offsets while preserving strict character slice identity and reassigning sequential `chunk_index` and `chunk_id`.
2. **Thread-Safe `LRUCache` (`src/data/source_store.py`)**:
   - Introduced `threading.RLock` guarding mutating operations (`get`, `__getitem__`, `__setitem__`, `pop`, `_evict`, `clear`, `__len__`, `current_bytes`).
   - Validated thread safety under concurrent multi-threaded read/write/evict stress testing.
3. **Slice Bounds Validation in `EvidenceSourceStore` (`src/data/source_store.py`)**:
   - In `_resolve_evidence_span`, strictly enforced `0 <= start < end <= len(doc.source_text)` for both registered `EvidenceSpan` instances and dynamic `span:{doc_id}:{start}:{end}` patterns.
   - Raises informative `ValueError` on inverted, negative, or out-of-bounds coordinates.
4. **Optimized Token Estimates & Memory Overhead (`src/data/source_store.py`)**:
   - Replaced temporary `.encode("utf-8")` allocations in `_doc_size` and `_chunk_size` with fast O(1) string length scaling.
   - Replaced heap-allocating `len(text.split())` in span resolution with fast whitespace count.

### Additional Covering Tests:
- `test_snapshot_rechunks_oversized_spans_to_fit_budget`: Verifies documents with oversized spans are split to strictly obey token budgets.
- `test_lru_cache_thread_safety`: Multi-threaded stress test with 8 concurrent workers executing 800 cache mutations.
- `test_evidence_source_store_rejects_invalid_slice_bounds`: Verifies rejection of inverted, out-of-bounds, and negative coordinate spans.

### Final Test Suite Verification:
- `tests/data/test_full_snapshot.py`: **11 passed in 5.74s**.
- Full test suite `pytest -q`: **410 passed, 3 warnings in 69.91s** (0 failures, 100% pass rate).
