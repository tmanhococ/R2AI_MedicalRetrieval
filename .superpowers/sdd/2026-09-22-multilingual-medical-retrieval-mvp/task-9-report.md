# Task 9 Report: Retrieval Orchestrator (6 Parallel Streams)

## 1. What was Implemented

In this task, we implemented the retrieval orchestrator module coordinating up to 6 multilingual parallel streams across BM25 and Dense vector search engines for documents and chunks:

1. **`StreamResults` Dataclass (`src/retrieval/orchestrator.py`)**:
   - Holds structured results partitioned into `doc_streams: dict[str, list[SearchHit]]` and `chunk_streams: dict[str, list[SearchHit]]`.
   - Default factory initializers allow instantiation with empty dicts or pre-populated stream hit lists.

2. **`RetrievalOrchestrator` (`src/retrieval/orchestrator.py`)**:
   - Coordinates the 6 candidate retrieval streams:
     1. `vi_bm25`: Query is `vi_condensed` (falling back to `vi` if condensed is None or whitespace), `language="vi"`, BM25 retriever.
     2. `vi_dense`: Query is `vi`, `language="vi"`, Dense retriever.
     3. `en_bm25`: Query is `en`, `language="en"`, BM25 retriever (omitted if `en` is None or empty).
     4. `en_dense`: Query is `en`, `language="en"`, Dense retriever (omitted if `en` is None or empty).
     5. `zh_bm25`: Query is `zh`, `language="zh"`, BM25 retriever (omitted if `zh` is None or empty).
     6. `zh_dense`: Query is `zh`, `language="zh"`, Dense retriever (omitted if `zh` is None or empty).
   - Core Methods:
     - `retrieve_all_streams(query_variants, top_k=None, enabled_streams=None) -> StreamResults`: Executes candidate retrieval across active streams for both documents and chunks.
     - `retrieve_documents(query_variants, top_k=None, enabled_streams=None) -> dict[str, list[SearchHit]]`: Retrieves candidate documents across active streams.
     - `retrieve_chunks(query_variants, top_k=None, enabled_streams=None) -> dict[str, list[SearchHit]]`: Retrieves candidate chunks across active streams.
   - Resilience & Fault Tolerance:
     - Missing Language: If a language is `None`, empty `""`, or whitespace, all corresponding `{lang}_bm25` and `{lang}_dense` streams are omitted from outputs.
     - Missing Retriever: If `bm25_retriever=None`, BM25 streams are omitted. If `dense_retriever=None`, Dense streams are omitted.
     - Stream Filtering: If `enabled_streams` is supplied, only streams present in that list are executed.
     - Exception Isolation: If a retriever raises an exception for any stream (e.g. partition missing, corrupt index), the exception is logged with full context, and the failed stream is safely omitted while all other streams proceed normally.
     - Method Fallback: If a dense retriever only implements `search_chunks` and `aggregate_documents`, `retrieve_documents` transparently falls back to chunk retrieval followed by max-score aggregation.
     - Signature Flexibility: Automatically supports both keyword-argument (`query=..., language=..., top_k=...`) and positional-argument retriever signatures.
   - Concurrency:
     - When `parallel=True`, stream retrievals are dispatched across a `ThreadPoolExecutor(max_workers=max_workers)` maintaining deterministic stream order in output dictionaries.
     - When `parallel=False`, stream retrievals execute sequentially.

3. **Package Exports (`src/retrieval/__init__.py`)**:
   - Exported `RetrievalOrchestrator` and `StreamResults` alongside existing `BM25Retriever` and `DenseRetriever`.

---

## 2. TDD Evidence

### Step 1 & 2: RED Phase (Initial Test Failure)
Unit test suite `tests/test_orchestrator.py` was created before implementation. Running `pytest tests/test_orchestrator.py -v` confirmed expected failure:
```
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\test_orchestrator.py'.
tests\test_orchestrator.py:7: in <module>
    from src.retrieval import RetrievalOrchestrator, StreamResults
E   ImportError: cannot import name 'RetrievalOrchestrator' from 'src.retrieval'
=========================== short test summary info ===========================
ERROR tests/test_orchestrator.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
========================= 1 warning, 1 error in 1.80s =========================
```

### Step 3 & 4: GREEN Phase (Implementation & Verification)
Following implementation of `src/retrieval/orchestrator.py` and updating `src/retrieval/__init__.py`, all 15 tests passed with 95% statement coverage:
```
tests/test_orchestrator.py::test_orchestrator_imports PASSED             [  6%]
tests/test_orchestrator.py::test_orchestrator_all_six_streams_success PASSED [ 13%]
tests/test_orchestrator.py::test_orchestrator_vi_condensed_fallback PASSED [ 20%]
tests/test_orchestrator.py::test_orchestrator_missing_language_handling PASSED [ 26%]
tests/test_orchestrator.py::test_orchestrator_missing_retriever_handling PASSED [ 33%]
tests/test_orchestrator.py::test_orchestrator_enabled_streams_filtering PASSED [ 40%]
tests/test_orchestrator.py::test_orchestrator_exception_resilience PASSED [ 46%]
tests/test_orchestrator.py::test_orchestrator_standalone_documents_and_chunks_methods PASSED [ 53%]
tests/test_orchestrator.py::test_orchestrator_concurrency_and_parallelism PASSED [ 60%]
tests/test_orchestrator.py::test_orchestrator_top_k_default_and_custom PASSED [ 66%]
tests/test_orchestrator.py::test_orchestrator_with_real_retrievers PASSED [ 73%]
tests/test_orchestrator.py::test_orchestrator_query_variants_none PASSED [ 80%]
tests/test_orchestrator.py::test_orchestrator_retriever_returns_none PASSED [ 86%]
tests/test_orchestrator.py::test_orchestrator_positional_args_and_method_missing PASSED [ 93%]
tests/test_orchestrator.py::test_orchestrator_aggregate_documents_fallback PASSED [100%]

=============================== tests coverage ================================
Name                            Stmts   Miss  Cover   Missing
-------------------------------------------------------------
src\retrieval\orchestrator.py     116      6    95%   63, 98, 208-209, 228-229
-------------------------------------------------------------
TOTAL                             116      6    95%
======================= 15 passed, 4 warnings in 5.41s ========================
```

Full repository test suite verification:
```
======================= 99 passed, 1 warning in 2.52s =========================
```

---

## 3. Files Changed

- `src/retrieval/orchestrator.py` (Created: `RetrievalOrchestrator` and `StreamResults`)
- `src/retrieval/__init__.py` (Modified: exported `RetrievalOrchestrator`, `StreamResults`)
- `tests/test_orchestrator.py` (Created: 15 comprehensive unit and integration tests)

---

## 4. Self-Review Findings

1. **Specification Alignment**:
   - All 6 stream names and query routing definitions match specifications precisely (`vi_bm25`, `vi_dense`, `en_bm25`, `en_dense`, `zh_bm25`, `zh_dense`).
   - `StreamResults` encapsulates `doc_streams` and `chunk_streams` cleanly for subsequent Weighted RRF fusion (Task 10).
2. **Robustness**:
   - Tested partial failures where individual streams raise exceptions; remaining streams return expected `SearchHit` lists without crashes.
   - Correct handling of `None`, empty string, and whitespace-only query variants.
3. **Concurrency**:
   - Multi-threading benchmark confirmed parallel execution achieves significant speedup over sequential execution.
   - Deterministic dictionary insertion ordering prevents non-deterministic ranking or iteration order downstream.
4. **Real Retriever Integration**:
   - Integration tested against actual `BM25Retriever` and `DenseRetriever` instances operating on synthetic corpus partitions.

---

## 5. Concerns / Notes

- **Dense Multi-threading**: When running heavy deep-learning encoders (e.g. PyTorch `SentenceTransformer`) under `parallel=True`, Python GIL and OpenMP threads may contend. For CPU inference with limited cores, `max_workers=2` or `parallel=False` can be configured easily if desired.
- **Benign FAISS Deprecation Warning**: A warning from `faiss/loader.py` regarding `numpy.core._multiarray_umath` appears during imports in Python 3.12 / NumPy 2.x; this is benign and does not affect retrieval correctness.
