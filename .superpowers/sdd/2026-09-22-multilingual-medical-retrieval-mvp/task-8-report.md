# Task 8 Report: Dense Indexing & Vector Search (FAISS & NumPy Fallback)

## 1. What was Implemented

In this task, we implemented the dense indexing and vector retrieval module supporting both FAISS (`IndexFlatIP`) and NumPy dot-product backends with language partitioning, instruction prefixes, and document-level score aggregation:

1. **`DenseIndexer` in `src/indexing/dense_index.py`**:
   - Manages partitioned vector indexing by language (`vi`, `en`, `zh`, etc.).
   - Supports flexible encoder inputs: pre-instantiated model, callable function, or model name string (with lazy-loaded `SentenceTransformer`).
   - Supports instruction prefix injection (`instruction_prefix["passage"]`) for asymmetric retrieval models (e.g. `intfloat/multilingual-e5-large`).
   - Generates $L_2$-normalized float32 embedding matrices.
   - Dual backend support:
     - FAISS backend: builds and serializes `faiss.IndexFlatIP` (`faiss_{lang}.index`) alongside `.npy` vector files.
     - NumPy backend: serializes `chunk_embeddings_{lang}.npy`.
   - Generates JSONL metadata (`metadata_{lang}.jsonl`) tracking `chunk_id`, `doc_id`, `language`, and `chunk_index`.
   - Generates index manifest (`manifest.json`) recording backend, dimensionality, instruction prefixes, and partition counts.

2. **`DenseRetriever` in `src/retrieval/dense_retriever.py`**:
   - Auto-infers backend, embedding dimensionality, and instruction prefixes from `manifest.json` or initialization arguments.
   - Implements `encode_query(query: str) -> np.ndarray`: prepends `instruction_prefix["query"]` and normalizes to unit $L_2$ norm.
   - Implements `search_chunks(query: str, language: str, top_k: int = 10) -> list[SearchHit]`:
     - Queries FAISS index (`faiss_index.search`) or computes NumPy dot product (`embeddings @ query_vec`).
     - Returns ranked `SearchHit` items with `item_type="chunk"`, `source="dense"`, and 1-indexed descending rank.
     - Robust edge-case handling: returns `[]` on empty/whitespace queries, $top\_k \le 0$, or non-existent/empty language partitions.
     - Automatic cross-backend fallback: falls back to in-memory FAISS from `.npy` or reconstructs NumPy vectors from FAISS index if a file is missing.
   - Implements `aggregate_documents(chunk_hits: list[SearchHit]) -> list[SearchHit]`:
     - Groups chunk hits by `doc_id` and takes the maximum score (`max(raw_score)`).
     - Returns ranked `SearchHit` items with `item_type="document"`, `item_id=doc_id`, and 1-indexed descending rank.
   - Implements `search_documents(query: str, language: str, top_k: int = 10) -> list[SearchHit]` for end-to-end document search.

3. **Package Exports**:
   - Exported `DenseIndexer` in `src/indexing/__init__.py`.
   - Exported `DenseRetriever` in `src/retrieval/__init__.py`.

---

## 2. TDD Evidence

### Step 1 & 2: RED Phase (Initial Test Failure)
Test file `tests/test_dense_retriever.py` was created before implementation. Running `pytest tests/test_dense_retriever.py -v` confirmed expected failure:
```
ModuleNotFoundError: No module named 'src.indexing.dense_index'
=========================== short test summary info ===========================
ERROR tests/test_dense_retriever.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 1.43s ===============================
```

### Step 3 & 4: GREEN Phase (Implementation & Verification)
Following implementation of `src/indexing/dense_index.py`, `src/retrieval/dense_retriever.py`, and exports, tests were executed:
```
tests/test_dense_retriever.py::test_dense_indexing_and_search_numpy PASSED [  8%]
tests/test_dense_retriever.py::test_dense_indexing_and_search_faiss PASSED [ 16%]
tests/test_dense_retriever.py::test_dense_instruction_prefix_support PASSED [ 25%]
tests/test_dense_retriever.py::test_dense_empty_language_partition PASSED [ 33%]
tests/test_dense_retriever.py::test_dense_document_aggregation_max_score PASSED [ 41%]
tests/test_dense_retriever.py::test_dense_multilingual_partitioning PASSED [ 50%]
tests/test_dense_retriever.py::test_dense_edge_cases PASSED              [ 58%]
tests/test_dense_retriever.py::test_dense_search_documents PASSED        [ 66%]
tests/test_dense_retriever.py::test_dense_backend_fallbacks PASSED       [ 75%]
tests/test_dense_retriever.py::test_dense_callable_encoder PASSED        [ 83%]
tests/test_dense_retriever.py::test_dense_string_model_name_mocked PASSED [ 91%]
tests/test_dense_retriever.py::test_dense_invalid_encoder PASSED         [100%]
======================= 12 passed, 3 warnings in 3.12s ========================
```
Full test suite across entire project:
```
======================== 83 passed, 1 warning in 1.65s ========================
```

---

## 3. Files Changed

- `src/indexing/dense_index.py` (Created: `DenseIndexer` and `_encode_texts` helper)
- `src/retrieval/dense_retriever.py` (Created: `DenseRetriever` with chunk search and document aggregation)
- `src/indexing/__init__.py` (Modified: exported `DenseIndexer`)
- `src/retrieval/__init__.py` (Modified: exported `DenseRetriever`)
- `tests/test_dense_retriever.py` (Created: 12 comprehensive unit tests)

---

## 4. Self-Review Findings

1. **Schema & Protocol Adherence**:
   - `SearchHit` records correctly output `source="dense"`, `item_type="chunk"` or `"document"`, `item_id`, `doc_id`, `language`, `raw_score`, and 1-indexed `rank`.
   - Document aggregation correctly performs max pooling over raw chunk scores for each distinct `doc_id`.
2. **Instruction Prefix Flexibility**:
   - Correctly prefixes `passage: ` on chunk texts during indexing and `query: ` on queries during search when configured.
   - Handles missing or empty prefixes gracefully without error.
3. **Backend Parity**:
   - Tested both `faiss` (using `IndexFlatIP`) and `numpy` (using inner product `@`).
   - Cross-fallback logic allows a numpy retriever to read from FAISS index and vice versa if one format is missing.
4. **Efficiency**:
   - Query vectors and chunk embeddings are pre-normalized, reducing search to fast matrix multiplication / dot product without per-query normalization of the corpus.

---

## 5. Concerns / Notes

- **Warning on FAISS import in NumPy 2.x environment**: A deprecation warning (`numpy.core._multiarray_umath is deprecated...`) is emitted by the installed `faiss` package loader on Python 3.12/NumPy 2.x. This is benign and does not affect functionality or numerical correctness.
