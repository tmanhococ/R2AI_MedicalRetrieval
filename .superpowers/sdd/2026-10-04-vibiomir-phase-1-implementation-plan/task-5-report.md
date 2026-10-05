# Task 5 Report: Immutable snapshot, source store và adapter/index integration

## Summary
Task 5 delivers the connection between frozen crawled source repositories and retrieval indexes through immutable versioned snapshots, high-performance sharded Parquet source storage, protocol-compliant dataset adapters, and provenance-tracked indexers.

## Commits Created
- `6bf23a4`: `feat: connect frozen source snapshots to retrieval indexes (Task 5)`

## Implementation Overview

### 1. Snapshot Publication & Sharding (`src/ingestion/snapshot.py`)
- Implemented `ShardedParquetWriter` with configurable row-group size (default 5,000) and byte threshold rotation (default 64MB) with atomic tmp-file rename upon shard completion.
- Implemented `build_snapshot()` to read fetched documents from `SourceRegistry`, run extractor and quality gates, write `documents_*.parquet` and `chunks_*.parquet` shards, populate an embedded SQLite lookup index (`lookup.sqlite`), and atomically publish `manifest.json`.
- Captured all provenance metadata in `SnapshotManifest`: `registry_sha256`, `queries_sha256`, `source_membership_sha256`, `extractor_fingerprint`, `chunker_fingerprint`, counts, and per-shard checksums.

### 2. High-Performance Source Store (`src/data/source_store.py`)
- Implemented `SnapshotSourceStore` conforming to the `SourceStore` protocol.
- Supported fast zero-copy lookups for `get_document(doc_id)` and `get_chunk(chunk_id)` backed by `lookup.sqlite` and PyArrow filter pushes on Parquet shards, with LRU-style object cache.
- Implemented `contains_official_doc(doc_id)` to verify official registry membership regardless of crawl status.

### 3. ViBioMIR Dataset Adapter (`src/data/vibiomir_adapter.py`)
- Implemented `ViBioMIRAdapter` conforming to `DatasetAdapter` protocol.
- Streamed `iter_documents()` and `iter_chunks()` over Parquet shards without materializing full dataset in RAM.
- Converted official int64 IDs (e.g. `583`) to core string IDs (`"583"`) while preserving official integer IDs in record metadata.
- Implemented `load_queries()` and `load_ground_truth()` parsing official query and qrels data.
- Implemented `validate_index_manifest()` verifying index snapshot alignment and chunker fingerprints against the snapshot manifest.

### 4. Memory-Bounded Dual Indexing (`src/indexing/bm25_index.py`, `src/indexing/dense_index.py`, `src/cli/index.py`)
- Updated `DenseIndexer.build_index()` to accept streaming iterables of chunks, encoding text in bounded batches (`batch_size`) per partition and tracking vector-to-chunk mapping hashes and resource estimates (RAM, bytes per vector).
- Updated `BM25Indexer.build_index()` with snapshot provenance fields (`snapshot_id`, `source_membership_sha256`, `chunker_fingerprint`) and resource estimation.
- Updated `src/cli/index.py` CLI to support `--snapshot` alongside `--dataset`, automatically detecting snapshot directories and validating index manifests.

### 5. CLI Runners (`src/cli/crawl.py`, `src/cli/ingest_vibiomir.py`)
- Added production entrypoints for crawling and corpus ingestion.

## Test Verification

1. **Snapshot & Adapter Unit & Integration Tests**:
   - Command: `uv run pytest tests/data/test_snapshot_adapter.py -q`
   - Result: `8 passed in 1.46s`
   - Verified tests:
     * `test_shard_rotation_and_atomic_publication`: Shard byte limits trigger rotation, tmp files cleaned up, manifest written last.
     * `test_atomic_publication_failure_does_not_publish_manifest`: Aborted runs do not leave corrupt manifests.
     * `test_snapshot_immutability_invariant`: Subsequent crawls and new snapshots leave earlier snapshot bits 100% untouched.
     * `test_same_url_keeps_two_distinct_official_doc_ids`: Duplicate URL mappings retain independent official IDs.
     * `test_snapshot_source_store_lookups`: Document and chunk lookups via SQLite lookup table.
     * `test_vibiomir_adapter_protocol_and_id_string_conversion`: Int-to-string ID normalization and `DatasetAdapter` compliance.
     * `test_one_pass_stream_consumption`: `OnePassIterator` verification proving bounded memory streaming.
     * `test_validate_index_manifest`: Strict checks for mismatched snapshot IDs or fingerprints.

2. **Full Test Suite Regression**:
   - Command: `uv run pytest -q`
   - Result: `263 passed in 8.82s`

## Invariants & Design Adherence
- **Immutability**: Snapshots are strictly append-only directories with content-addressed Parquet shards and atomic manifest publication.
- **Memory efficiency**: No full-corpus text list materialization in indexing streams.
- **Provenance guarantee**: Indices record the exact snapshot ID and source membership hash, catching out-of-sync indexes prior to search.

## Concerns / Notes
- None. All components adhere strictly to specifications and test coverage is comprehensive.

## Fix Round 1 Summary & Verification

### Fix Round 1 Commits
- `f816295`: `fix: enforce strict provenance hashes, streaming CLI indexers, and pre-search validation (Task 5 Round 1)`

### Addressed Findings:
1. **Strict Manifest Provenance & Mapping Verification**:
   - Updated `validate_index_manifest()` in `src/data/vibiomir_adapter.py` so that `source_membership_sha256` and `chunker_fingerprint` are strictly required fields (raising `ValueError` if missing or mismatched).
   - Added validation for `vector_chunk_mapping_hash` against expected mapping hashes.
   - Added `validate_search_compatibility()` and integrated pre-search checks into `DenseRetriever.__init__()` and `validate_index_manifest()`, failing prior to search if `model_name`, `instruction_prefix`, or `normalize_embeddings` mismatch.
   - Recorded `model_name` in `DenseIndexer.build_index()` manifest.

2. **Full Streaming in CLI (`src/cli/index.py`)**:
   - Refactored `run_index()` so that when indexing from a snapshot (`effective_snapshot is not None`), `adapter.load_documents()` and `adapter.load_chunks()` are never called.
   - Streaming generators `adapter.iter_documents()` and `adapter.iter_chunks()` are passed directly to `bm25_indexer.build_indexes()` and `dense_indexer.build_index()`, preserving memory-bounded single-pass streaming.
   - Verified that `vector_chunk_mapping_hash` is passed to `validate_index_manifest()` upon index creation.

3. **Stream SQL Cursor in `src/ingestion/snapshot.py`**:
   - Replaced `cursor.fetchall()` in `build_snapshot()` with `cursor.fetchmany(1000)` batch streaming, ensuring millions of rows are never buffered concurrently into a single Python list.

4. **Added Mandated Tests in `tests/data/test_snapshot_adapter.py`**:
   - `test_missing_foreign_key_detection`: Verifies `SnapshotSourceStore` raises `KeyError` when resolving orphan document/chunk IDs not present in snapshot, and tests that `validate_corpus(..., strict=True)` catches missing foreign keys.
   - `test_vector_chunk_mapping_hash_and_presearch_verification`: Verifies `validate_index_manifest()` catches missing or mismatched `vector_chunk_mapping_hash`, and tests that pre-search compatibility checks fail on model, prefix, or normalization mismatches.
   - `test_vibiomir_adapter_load_ground_truth`: Verifies `ViBioMIRAdapter.load_ground_truth()` parses Qrels files into typed `EvaluationRecord` objects with `ReferenceChunk` and `frozenset` document IDs.
   - Expanded `test_validate_index_manifest` to verify missing `source_membership_sha256` and missing `chunker_fingerprint`.

### Fix Round 1 Test Results:
- `uv run pytest tests/data/test_snapshot_adapter.py -v`: 11 passed in 3.65s
- `uv run pytest -q`: 266 passed in 10.21s
