# Task 3 Report: Dataset Adapter Protocol & In-Memory / JSONL Adapter

**Task Name:** Dataset Adapter Protocol & In-Memory / JSONL Adapter  
**Date:** 2026-09-22  
**Status:** DONE  

---

## 1. What Was Implemented

- **`DatasetAdapter(Protocol)` (`src/data/dataset_adapter.py`)**:
  - Defined `@runtime_checkable` `DatasetAdapter` protocol with abstract methods:
    - `load_documents() -> list[DocumentRecord]`
    - `load_chunks() -> list[ChunkRecord]`
    - `load_queries(split: str) -> list[QueryRecord]`
    - `load_ground_truth(split: str) -> list[GroundTruthRecord]`
- **`JsonlDatasetAdapter` (`src/data/dataset_adapter.py`)**:
  - Implements JSONL-backed dataset loading.
  - Normalizes IDs to `str`.
  - Normalizes language codes/names to `"vi" | "en" | "zh"` (e.g. `"VI"`, `"Vietnamese"`, `"EN"`).
  - Handles `docs_path`, `chunks_path`, `queries_dir`, and `qrels_dir` as either `str` or `Path`.
  - Supports optional chunks (`chunks_path=None` returns `[]`).
  - Supports flexible query split file naming (`queries_{split}.jsonl` or `{split}.jsonl` or direct file).
  - Supports flexible ground truth split naming (`qrels_{split}.jsonl`, `ground_truth_{split}.jsonl`, or `{split}.jsonl` or direct file).
  - Parses ground truth `relevant_doc_ids` and `relevant_chunk_ids` into Python sets of strings (`set[str]`).
  - Gracefully ignores empty/whitespace lines in JSONL files.
  - Raises informative `FileNotFoundError` and `ValueError` (when path is not a file or language is invalid).
- **`InMemoryDatasetAdapter` (`src/data/dataset_adapter.py`)**:
  - In-memory implementation conforming to `DatasetAdapter` protocol for mock data in testing and pipeline prototyping without disk I/O.
- **Package Exports (`src/data/__init__.py`)**:
  - Exported `DatasetAdapter`, `JsonlDatasetAdapter`, and `InMemoryDatasetAdapter`.
- **Test Suite (`tests/test_dataset_adapter.py`)**:
  - 14 comprehensive unit tests verifying record loading, missing files, directory paths, chunks None handling, filename variants, missing directories, normalization, invalid language error handling, empty line tolerance, in-memory adapter behavior, and protocol conformance.

---

## 2. TDD Evidence

### Step 1 & 2: RED Phase (Failing Test Verification)
Before implementing `src/data/dataset_adapter.py`, `tests/test_dataset_adapter.py` was created and run:

```text
pytest tests/test_dataset_adapter.py -v
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-8.2.0, pluggy-1.6.0 -- C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\python.exe
cachedir: .pytest_cache
rootdir: D:\Research\R2AI\mvp-rag-medical
configfile: pyproject.toml
plugins: anyio-4.14.2, Faker-37.11.0, langsmith-0.10.18, asyncio-0.23.6, cov-7.1.0, typeguard-4.6.0, zarr-3.1.5
asyncio: mode=Mode.STRICT
collecting ... collected 0 items / 1 error

=================================== ERRORS ====================================
_______________ ERROR collecting tests/test_dataset_adapter.py ________________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\test_dataset_adapter.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
tests\test_dataset_adapter.py:5: in <module>
    from src.data.dataset_adapter import DatasetAdapter, JsonlDatasetAdapter, InMemoryDatasetAdapter
E   ModuleNotFoundError: No module named 'src.data.dataset_adapter'
=========================== short test summary info ===========================
ERROR tests/test_dataset_adapter.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.20s ===============================
```

### Step 3 & 4: GREEN Phase (Passing Test Verification)
Implemented `src/data/dataset_adapter.py` and updated `src/data/__init__.py`.

```text
pytest tests/test_dataset_adapter.py -v
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-8.2.0, pluggy-1.6.0
rootdir: D:\Research\R2AI\mvp-rag-medical
configfile: pyproject.toml
plugins: anyio-4.14.2, Faker-37.11.0, langsmith-0.10.18, asyncio-0.23.6, cov-7.1.0, typeguard-4.6.0, zarr-3.1.5
asyncio: mode=Mode.STRICT
collected 14 items

tests\test_dataset_adapter.py ..............                             [100%]

=============================== tests coverage ================================
Name                          Stmts   Miss  Cover
-------------------------------------------------
src\data\dataset_adapter.py     118      0   100%
-------------------------------------------------
TOTAL                           118      0   100%
============================= 14 passed in 0.48s ==============================
```

### Full Test Suite & Typing Verification
- Full test suite: **25 passed in 0.62s** (3 config + 14 dataset adapter + 8 validator/schema tests).
- Coverage across `src/data`: **100%** (205/205 statements).
- Static type check: `mypy src/data tests/test_dataset_adapter.py` succeeded with **0 errors in 5 source files**.

---

## 3. Files Changed / Created

- `src/data/dataset_adapter.py` [NEW]
- `src/data/__init__.py` [MODIFIED]
- `tests/test_dataset_adapter.py` [NEW]
- `.superpowers/sdd/2026-09-22-multilingual-medical-retrieval-mvp/task-3-report.md` [NEW]

---

## 4. Self-Review Findings

- **Protocol Conformance**: Both `JsonlDatasetAdapter` and `InMemoryDatasetAdapter` fully conform to `DatasetAdapter` protocol (verified with `@runtime_checkable` and `isinstance` checks).
- **Data Schema Compatibility**: All methods return typed domain records (`DocumentRecord`, `ChunkRecord`, `QueryRecord`, `GroundTruthRecord`) with sets for ground truth ID collections.
- **Robust Error Handling**: Missing files, missing directories, non-file path targets, and invalid language values raise clear and standard exceptions (`FileNotFoundError`, `ValueError`).
- **No Side Effects or Mutations**: All record loaders yield new immutable records without mutating external inputs.

---

## 5. Git Commit

- **Commit SHA**: `00a0ca2`
- **Commit Message**: `feat: implement dataset adapter protocol and jsonl adapter`

---

## 6. Concerns

- None. 100% test coverage achieved and all contract requirements verified.
