# Task 8 Report: Official JSON/ZIP exporter and strict validator

## 1. Summary of Implementation
Implemented the ViBioMIR Phase 1 submission export pipeline and strict offline validation suite conforming to official competition rules and section 9 of the implementation plan:
- **`src/submission/validators.py`**:
  - `validate_predictions`: Performs strict structural and semantic validation on submission rows:
    - Signed int64 verification for query IDs, document IDs, and chunk doc IDs (explicitly rejecting booleans, floats, strings, and out-of-range integers).
    - Exact query coverage against official queries (detects missing, extra, and duplicate queries; supports expected count check).
    - Registry/snapshot membership verification via `SourceStore.contains_official_doc(doc_id)`.
    - Deduplication checking for document IDs in `relevant_docs` and `(doc_id, chunk_text)` pairs in `relevant_chunks`.
    - Exact source substring provenance check verifying chunk text appears verbatim in document `source_text`.
    - Acceptance of valid empty predictions (`relevant_docs: []`, `relevant_chunks: []`) and chunk doc IDs not present in `relevant_docs`.
    - Strict validation of optional non-negative integer `chunk_order`.
  - `validate_zip`: Inspects submission ZIP archives:
    - Enforces exactly one root entry named `results.json` without directories, paths, extra files, or encryption flags.
    - Decompresses and parses UTF-8 JSON array, delegating to `validate_predictions`.
  - `SubmissionValidationError`: Structured exception retaining detailed `ValidationReport`.
- **`src/submission/vibiomir_json_zip.py`**:
  - `VibiomirExporter`: Translates core `FinalPrediction` pipeline outputs into official submission format:
    - Resolves internal string IDs to signed int64 official IDs.
    - Level A pre-serialization provenance check: verifies chunk existence in store, non-empty text, source version/hash equality with parent document, and exact character span slice match (`doc.source_text[start:end] == chunk.text`).
    - Sorts queries deterministically by integer ID ascending.
    - Atomic writes: writes JSON, deterministic ZIP (fixed timestamp `1980-01-01 00:00:00`, compression level 9, standard file attributes), and validation report via temporary files and `os.replace`.
    - Produces `ExportResult(json_path, zip_path, validation_report_path, json_sha256, zip_sha256, validation_report)`.
- **`src/data/vibiomir_contracts.py` & `src/submission/__init__.py`**:
  - Updated `ValidationReport` dataclass with `valid`, `query_count`, `document_count`, `chunk_count`, `errors`, `warnings`, `snapshot_id`, and `is_valid` property.
  - Added `ExportResult` dataclass.
  - Exported relevant symbols in `src/submission`.

## 2. TDD Evidence

### RED Test Phase
Command:
```powershell
uv run pytest tests/submission/test_vibiomir_export.py -q
```
Output:
```
=================================== ERRORS ====================================
__________ ERROR collecting tests/submission/test_vibiomir_export.py __________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\submission\test_vibiomir_export.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\submission\test_vibiomir_export.py:20: in <module>
    from src.submission.validators import (
E   ModuleNotFoundError: No module named 'src.submission.validators'
=========================== short test summary info ===========================
ERROR tests/submission/test_vibiomir_export.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.16s
```

### GREEN Test Phase
Command:
```powershell
uv run pytest tests/submission/test_vibiomir_export.py -v
```
Output:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0 -- D:\Research\R2AI\mvp-rag-medical\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Research\R2AI\mvp-rag-medical
configfile: pyproject.toml
plugins: anyio-4.15.1, asyncio-1.4.0, cov-7.1.0
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 40 items

tests/submission/test_vibiomir_export.py::test_source_store_protocol_compliance PASSED [  2%]
tests/submission/test_vibiomir_export.py::test_sparse_int64_ids_pass PASSED [  5%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_query_id[True] PASSED [  7%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_query_id[False] PASSED [ 10%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_query_id[1] PASSED [ 12%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_query_id[1.0] PASSED [ 15%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_query_id[9223372036854775808] PASSED [ 17%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_query_id[-9223372036854775809] PASSED [ 20%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_doc_id[True] PASSED [ 22%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_doc_id[False] PASSED [ 25%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_doc_id[583] PASSED [ 27%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_doc_id[583.0] PASSED [ 30%]
tests/submission/test_vibiomir_export.py::test_reject_non_int64_or_bool_doc_id[9223372036854775808] PASSED [ 32%]
tests/submission/test_vibiomir_export.py::test_reject_missing_query PASSED [ 35%]
tests/submission/test_vibiomir_export.py::test_reject_extra_query PASSED [ 37%]
tests/submission/test_vibiomir_export.py::test_reject_duplicate_query PASSED [ 40%]
tests/submission/test_vibiomir_export.py::test_expected_count_check PASSED [ 42%]
tests/submission/test_vibiomir_export.py::test_reject_foreign_doc_id PASSED [ 45%]
tests/submission/test_vibiomir_export.py::test_reject_foreign_chunk_doc_id PASSED [ 47%]
tests/submission/test_vibiomir_export.py::test_reject_duplicate_doc_ids_in_same_query PASSED [ 50%]
tests/submission/test_vibiomir_export.py::test_reject_duplicate_chunk_in_same_query PASSED [ 52%]
tests/submission/test_vibiomir_export.py::test_missing_lists_rejected PASSED [ 55%]
tests/submission/test_vibiomir_export.py::test_valid_empty_predictions_pass PASSED [ 57%]
tests/submission/test_vibiomir_export.py::test_reject_modified_chunk_text PASSED [ 60%]
tests/submission/test_vibiomir_export.py::test_chunk_with_official_doc_not_in_relevant_docs_accepted PASSED [ 62%]
tests/submission/test_vibiomir_export.py::test_valid_chunk_order PASSED  [ 65%]
tests/submission/test_vibiomir_export.py::test_reject_invalid_chunk_order[-1] PASSED [ 67%]
tests/submission/test_vibiomir_export.py::test_reject_invalid_chunk_order[True] PASSED [ 70%]
tests/submission/test_vibiomir_export.py::test_reject_invalid_chunk_order[False] PASSED [ 72%]
tests/submission/test_vibiomir_export.py::test_reject_invalid_chunk_order[1.5] PASSED [ 75%]
tests/submission/test_vibiomir_export.py::test_reject_invalid_chunk_order[0] PASSED [ 77%]
tests/submission/test_vibiomir_export.py::test_validate_zip_valid PASSED [ 80%]
tests/submission/test_vibiomir_export.py::test_reject_zip_with_extra_files PASSED [ 82%]
tests/submission/test_vibiomir_export.py::test_reject_zip_with_subdirectories PASSED [ 85%]
tests/submission/test_vibiomir_export.py::test_reject_zip_with_wrong_filename PASSED [ 87%]
tests/submission/test_vibiomir_export.py::test_reject_corrupt_or_non_zip PASSED [ 90%]
tests/submission/test_vibiomir_export.py::test_reject_encrypted_zip PASSED [ 92%]
tests/submission/test_vibiomir_export.py::test_vibiomir_exporter_end_to_end_and_determinism PASSED [ 95%]
tests/submission/test_vibiomir_export.py::test_exporter_rejects_orphan_chunk_id PASSED [ 97%]
tests/submission/test_vibiomir_export.py::test_exporter_rejects_tampered_chunk_provenance PASSED [100%]

============================= 40 passed in 0.22s ==============================
```

## 3. Full Regression Suite Result
Command:
```powershell
uv run pytest -q
```
Output:
```
........................................................................ [ 21%]
........................................................................ [ 43%]
........................................................................ [ 65%]
........................................................................ [ 87%]
..........................................                               [100%]
330 passed in 8.16s
```
Regressions: 0.

## 4. Files Created and Modified
- `src/submission/validators.py`: Created.
- `src/submission/vibiomir_json_zip.py`: Created.
- `tests/submission/test_vibiomir_export.py`: Created.
- `src/data/vibiomir_contracts.py`: Updated `ValidationReport` and added `ExportResult`.
- `src/submission/__init__.py`: Exported new validation and exporter interfaces.

## 5. Commit SHA and Message
- Commit: `4a59f22`
- Message: `feat: export validated ViBioMIR JSON submissions (Task 8)`

## 6. Self-Review Checklist
- [x] Sparse int64 validation for query IDs and document IDs strictly implemented (rejects bool, string, float).
- [x] Exact query coverage validated (no missing, extra, or duplicate queries).
- [x] Foreign doc IDs outside official snapshot/registry rejected.
- [x] Duplicates in `relevant_docs` and `relevant_chunks` rejected.
- [x] Valid empty predictions pass; missing list fields rejected.
- [x] Pre-serialization Level A provenance checks verify exact character spans, source hash, and version.
- [x] Level B validator verifies exact substring containment in document text.
- [x] Chunk with valid official doc not in `relevant_docs` is accepted.
- [x] Negative, bool, or float `chunk_order` rejected.
- [x] Single-file `results.json` ZIP requirement strictly enforced (rejects directories, subdirectories, extra entries, and encrypted archives).
- [x] Exporter is deterministic across runs (identical SHA-256 for JSON and ZIP).
- [x] Atomic file writes implemented using temporary files and atomic replace.
- [x] Zero silent mutations: invalid rows trigger explicit validation errors.
