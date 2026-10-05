# Task 4 Report: Disk-backed global sparse indexes

## Overview
Task 4 implements disk-backed global sparse indexes for ViBioMIR Phase 2 full-corpus retrieval using the Lucene/BM25 scoring protocol (`src/indexing/lucene.py`). It provides robust, partitioned disk builds, global multi-partition assembly with unified IDF and `avgdl` across segments, crash/replay safety, and sub-second BM25 retrieval without storing raw body text.

---

## Key Components Implemented
1. **`src/indexing/lucene.py`**:
   - `LuceneSparseBackend`: Disk-backed BM25 retriever using SQLite storage per partition and assembled global indices.
   - `build_partition`: Builds isolated disk indices with atomic commit and StageManifest tracking.
   - `assemble`: Combines multiple partitions into a unified global index with exact global DF, IDF, and document length statistics.
   - `search`: Executes sub-second BM25 retrieval with configurable $k_1 = 1.2$ and $b = 0.75$.
2. **`tests/indexing/test_lucene_sparse.py`**:
   - Comprehensive unit and integration tests covering clinical symbols, negation preservation, multi-lingual query-document alignment (VI, EN, ZH, und), official document IDs exceeding 32-bit integer limits (`> 2^31 - 1`), and multi-partition DF assembly.
3. **`tests/integration/test_phase2_sparse_checkpoint.py`**:
   - Integration test verifying the B1 sparse-only checkpoint flow:
     - Building Lucene sparse index with large/out-of-pilot IDs.
     - Running prediction via sparse route without loading E5 or BGE.
     - Exporting submission ZIP.
     - Strictly validating submission ZIP via `validate_zip` and `validate_predictions`.
     - Handling empty hits (min results = 0) and capped result counts correctly.

---

## Test Results
- **Lucene Sparse Tests**: `tests/indexing/test_lucene_sparse.py` (6/6 tests passing).
- **Integration Checkpoint Test**: `tests/integration/test_phase2_sparse_checkpoint.py` (Passing).
- **Full Test Suite**: Verified passing across core index, validation, and submission suites.

---

## Conclusion
Task 4 is fully completed with robust test coverage, strict validator compliance, and clean integration.
