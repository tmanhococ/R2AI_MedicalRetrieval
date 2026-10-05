### Task 9: Retrieval Orchestrator (6 Parallel Streams)

**Files:**
- Create: `src/retrieval/orchestrator.py`
- Modify: `src/retrieval/__init__.py`
- Test: `tests/test_orchestrator.py`

**Interfaces:**
- Consumes: `QueryVariants` (from `src.query.query_processor`), `SearchHit` (from `src.data.schemas`), `BM25Retriever` (from `src.retrieval.bm25_retriever`), `DenseRetriever` (from `src.retrieval.dense_retriever`)
- Produces: `RetrievalOrchestrator`, `StreamResults`

**Stream Definitions:**
The 6 candidate retrieval streams are:
1. `vi_bm25`: Query is `query_variants.vi_condensed or query_variants.vi`, language="vi", source BM25
2. `vi_dense`: Query is `query_variants.vi`, language="vi", source Dense
3. `en_bm25`: Query is `query_variants.en`, language="en", source BM25 (only if `query_variants.en` is non-empty)
4. `en_dense`: Query is `query_variants.en`, language="en", source Dense (only if `query_variants.en` is non-empty)
5. `zh_bm25`: Query is `query_variants.zh`, language="zh", source BM25 (only if `query_variants.zh` is non-empty)
6. `zh_dense`: Query is `query_variants.zh`, language="zh", source Dense (only if `query_variants.zh` is non-empty)

**Requirements & Specifications:**
1. **Data Contract:**
   ```python
   @dataclass
   class StreamResults:
       doc_streams: dict[str, list[SearchHit]]
       chunk_streams: dict[str, list[SearchHit]]
   ```
2. **Retrieval Orchestrator Class:**
   ```python
   class RetrievalOrchestrator:
       def __init__(
           self,
           bm25_retriever: Any = None,
           dense_retriever: Any = None,
           default_top_k: int = 200,
           parallel: bool = True,
           max_workers: int = 6,
       ): ...
   ```
3. **Methods:**
   - `retrieve_all_streams(self, query_variants: QueryVariants, top_k: int | None = None, enabled_streams: list[str] | None = None) -> StreamResults`
   - `retrieve_documents(self, query_variants: QueryVariants, top_k: int | None = None, enabled_streams: list[str] | None = None) -> dict[str, list[SearchHit]]`
   - `retrieve_chunks(self, query_variants: QueryVariants, top_k: int | None = None, enabled_streams: list[str] | None = None) -> dict[str, list[SearchHit]]`
4. **Resilience & Missing Language Handling:**
   - If a language in `query_variants` is `None` or empty string, all corresponding streams (`{lang}_bm25`, `{lang}_dense`) are omitted from `doc_streams` and `chunk_streams`.
   - If a retriever raises an exception for any stream (e.g. partition missing), the orchestrator logs or suppresses the failure and omits that stream, ensuring the overall pipeline never crashes.
   - If `enabled_streams` is supplied, only streams in that list are executed.
   - If `bm25_retriever` is `None`, BM25 streams are skipped. If `dense_retriever` is `None`, Dense streams are skipped.
5. **Concurrency:**
   - When `parallel=True`, dispatches stream retrievals concurrently via `ThreadPoolExecutor(max_workers=max_workers)`. When `parallel=False`, executes sequentially.
6. **Package Exports:**
   - Export `RetrievalOrchestrator` and `StreamResults` in `src/retrieval/__init__.py`.

**TDD Steps:**
- Step 1: Write comprehensive failing tests in `tests/test_orchestrator.py` testing mock retrievers, missing languages, enabled_streams filtering, concurrency, exception resilience, and edge cases.
- Step 2: Run `pytest tests/test_orchestrator.py -v` (Verify RED).
- Step 3: Implement `src/retrieval/orchestrator.py` and update `src/retrieval/__init__.py`.
- Step 4: Run `pytest tests/test_orchestrator.py -v` (Verify GREEN).
- Step 5: Commit changes.
