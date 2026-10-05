# Task 7 Report: Multilingual Tokenizers & BM25 Indexing / Retrieval (BM25s / Rank-BM25)

## 1. What was Implemented

### Multilingual Tokenizers (`src/indexing/tokenizers.py`)
- Factory function `get_tokenizer(lang: str, mode: str | None = None) -> Callable[[str], list[str]]`.
- Supported language modes:
  - **Vietnamese (`vi`)**:
    - `"syllable"` (default): Unicode word regex matching (`re.findall(r"\w+", text.lower(), flags=re.UNICODE)`) preserving Vietnamese diacritics and alphanumeric terms.
    - `"pyvi"`: Leverages `pyvi.ViTokenizer` if installed; falls back gracefully to `"syllable"` otherwise.
  - **English (`en`)**:
    - `"whitespace"` (default): Word extraction regex (`re.findall(r"\b\w+\b", text.lower())`).
    - `"nltk_stem"`: `nltk.stem.PorterStemmer` tokenization with fallback to whitespace tokens if `nltk` is absent.
  - **Chinese (`zh`)**:
    - `"jieba"` (default): Word boundary segmentation via `jieba.lcut` if installed; falls back gracefully to `"char"`.
    - `"char"`: Character and token regex (`r"[\u4e00-\u9fff]|[a-zA-Z0-9]+"`) suitable for Hanzi medical character segmentation.
  - **Generic fallback**: Unicode word tokenizer for any unsupported or unspecified language.
  - Robust handling of `None`, empty strings, whitespace, and punctuation-only inputs returning `[]`.

### Multilingual BM25 Indexer (`src/indexing/bm25_index.py`)
- Class `BM25Indexer(backend: str = "bm25s" | "rank_bm25", tokenizers_config: dict[str, str] | None = None)`:
  - Supports dual backends: `bm25s` and `rank_bm25`.
  - Partitions both `documents` and `chunks` per language (`vi`, `en`, `zh`, etc.).
  - Saves individual models and metadata files per partition:
    - For `rank_bm25`: `{partition}.pkl` storing serialized model, `item_ids`, `doc_ids`, `language`, `item_type`.
    - For `bm25s`: `{partition}/` model directory saved via `model.save()`, alongside `{partition}_meta.pkl`.
    - For human inspection: `{partition}_metadata.jsonl` containing line-delimited item metadata.
    - Generates a root `manifest.json` detailing partition counts, backend, and tokenization configurations.
  - Mitigates `BM25Okapi` IDF zero-division / zero-score anomaly on small corpora (`N <= 2`) and high-frequency terms by applying a smoothed positive IDF floor (`0.25` or `epsilon * average_idf`).

### Multilingual BM25 Retriever (`src/retrieval/bm25_retriever.py`, `src/retrieval/__init__.py`)
- Class `BM25Retriever(index_dir: Path | str, backend: str | None = None, tokenizers_config: dict | None = None)`:
  - Automatically discovers configuration and backend from `manifest.json` if not explicitly specified.
  - Implements:
    - `search_documents(query: str, language: str, top_k: int = 10) -> list[SearchHit]`
    - `search_chunks(query: str, language: str, top_k: int = 10) -> list[SearchHit]`
  - Produces standard `SearchHit` instances conforming to `src.data.schemas.SearchHit`:
    - `item_id`, `item_type` (`"document"` or `"chunk"`), `doc_id`, `language`, `source="bm25"`, `raw_score`, `rank` (1-indexed).
  - Graceful handling:
    - If language partition has 0 items or index does not exist: returns `[]` safely without throwing exceptions.
    - If query is empty, whitespace, or has no matches: returns `[]`.
    - Caches loaded partition models and metadata for high-throughput querying.

---

## 2. TDD Evidence

### RED Failure
Initial test run before implementation in `tests/test_bm25_retriever.py`:
```
=================================== ERRORS ====================================
________________ ERROR collecting tests/test_bm25_retriever.py ________________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\test_bm25_retriever.py'.
Traceback:
tests\test_bm25_retriever.py:4: in <module>
    from src.indexing.bm25_index import BM25Indexer
E   ModuleNotFoundError: No module named 'src.indexing.bm25_index'
=========================== short test summary info ===========================
ERROR tests/test_bm25_retriever.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.27s ===============================
```

### GREEN Pass
After implementing the tokenizers, indexer, and retriever with IDF floor smoothing:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-8.2.0, pluggy-1.6.0
collected 9 items

tests/test_bm25_retriever.py::test_bm25_indexing_and_search PASSED       [ 11%]
tests/test_bm25_retriever.py::test_bm25_search_documents PASSED          [ 22%]
tests/test_bm25_retriever.py::test_bm25_empty_partition_graceful PASSED  [ 33%]
tests/test_bm25_retriever.py::test_bm25s_backend_indexing_and_search PASSED [ 44%]
tests/test_bm25_retriever.py::test_bm25_multilingual_partitions PASSED   [ 55%]
tests/test_tokenizers_functionality PASSED                               [ 66%]
tests/test_tokenizers_edge_cases PASSED                                   [ 77%]
tests/test_bm25_edge_cases PASSED                                        [ 88%]
tests/test_bm25_invalid_backend PASSED                                   [100%]

============================== 9 passed in 1.48s ==============================
```

Full regression test across all 7 test suites:
```
============================= 71 passed in 1.78s ==============================
```

Type checking (`mypy`):
```
Success: no issues found in 5 source files
```

---

## 3. Files Changed

- `src/indexing/tokenizers.py` (NEW) - Multilingual tokenizers with fallback strategies (`vi`, `en`, `zh`, generic).
- `src/indexing/bm25_index.py` (NEW) - `BM25Indexer` with language partitioning, dual backend persistence, manifest generation, and IDF smoothing.
- `src/indexing/__init__.py` (MODIFIED) - Re-exported `BM25Indexer` and `get_tokenizer`.
- `src/retrieval/__init__.py` (NEW) - Package initialization re-exporting `BM25Retriever`.
- `src/retrieval/bm25_retriever.py` (NEW) - `BM25Retriever` implementing `search_documents` and `search_chunks` returning `SearchHit` objects.
- `tests/test_bm25_retriever.py` (NEW) - 9 comprehensive unit and integration tests for indexing, retrieval, tokenizers, multilingual partitions, and edge cases.

---

## 4. Self-Review Findings

- **Architecture Conformance**: All classes adhere to the signatures and behavioral specifications in `task-7-brief.md` and configuration in `configs/r2ai_mvp.yaml`.
- **Backend Robustness**: Both `rank_bm25` and `bm25s` backends are fully supported and verified with unit tests.
- **Graceful Degradation**: Missing optional packages (`pyvi`, `jieba`, `nltk`) degrade gracefully to regex/char-level fallback tokenizers without crashing.
- **Empty & Non-Existent Partitions**: Empty corpora or missing language partitions safely yield empty search results (`[]`).
- **Correct Hit Schema**: `SearchHit` fields (`item_id`, `item_type`, `doc_id`, `language`, `source="bm25"`, `raw_score`, `rank`) are verified with 1-based indexing and descending score order.

---

## 5. Any Concerns

- None. Both `bm25s` and `rank_bm25` are installed and functioning properly, all 71 tests in the repository pass cleanly, and static typing passes without errors.
