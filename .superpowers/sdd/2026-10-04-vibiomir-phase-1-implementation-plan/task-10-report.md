# Task 10 Report: Dev Annotation và Error Analysis có Provenance

## Summary of Implementation
Implemented the development annotation protocol, failure drop funnel analysis with strict provenance, and aggregate diagnostics:

1. **Contracts and Provenance (`src/data/vibiomir_contracts.py`)**:
   - Added `ErrorBreakdown` dataclass tracking the entire retrieval funnel and chunk evidence localization:
     * `query_id: int`
     * `missing_source_docs: tuple[int, ...]` (Gold docs missing from SourceStore snapshot)
     * `missing_candidate_docs: tuple[int, ...]` (Gold docs in snapshot but dropped before first-stage candidate retrieval)
     * `lost_after_rerank_docs: tuple[int, ...]` (Gold docs in candidates but dropped during/after cross-encoder reranking)
     * `lost_after_cutoff_docs: tuple[int, ...] `(Gold docs in reranked set but excluded by final cutoff threshold)
     * `missing_chunk_refs: tuple[ReferenceChunk, ...]` (Gold reference chunks not matched by predicted chunks)
     * `evidence_missing_given_doc: tuple[ReferenceChunk, ...]` (Gold reference chunks missed where parent doc was retrieved in `final_doc_ids`)
   - Enhanced `ReferenceChunk` to support exact character span offsets `char_start: int | None = None` and `char_end: int | None = None` with non-negative validation and backwards compatibility.
   - Enhanced `DevQueryMetadata` with list-to-tuple normalization in `__post_init__`.

2. **Provenance Error Breakdown Engine (`src/evaluation/error_analysis.py`)**:
   - `build_error_breakdown`:
     * Validates that `gold.query_id == trace.query_id`.
     * Identifies exactly where gold documents drop out across the 4 retrieval stages.
     * Extracts predicted chunks from either `store.get_chunk(cid)` via `trace.final_chunk_ids` or `trace.metadata.get("predicted_chunks")` (typed dicts or objects).
     * Configures `TextOverlapEvaluator` with `profile` (defaults to `PROVISIONAL_SPEC_V1_PROFILE`).
     * Matches predicted chunks against reference chunks requiring matching `doc_id` and normalized token LCS ratio $\ge 40\%$.
     * Categorizes chunk evidence misses into overall `missing_chunk_refs` and conditionally into `evidence_missing_given_doc` if parent doc is in `final_doc_ids`.
   - `generate_error_breakdown_summary`:
     * Aggregates failure counts across all queries: totals for each stage (`total_missing_source_docs`, `total_missing_candidate_docs`, `total_lost_after_rerank_docs`, `total_lost_after_cutoff_docs`, `total_missing_chunk_refs`, `total_evidence_missing_given_doc`).
     * Computes distinct query counts affected by each error mode (`queries_with_missing_source`, `queries_with_missing_candidates`, etc.).
     * Handles empty breakdown sequences safely.

3. **Development Annotation Guide (`docs/vibiomir/dev_annotation_guide.md`)**:
   - Documented namespace separation: dev query IDs reserved $\ge 9001$, verified anti-leakage against official $1 \dots 1{,}200$.
   - Explicit `DevQueryMetadata` specification and provenance tracking.
   - Multi-retriever pooling protocol across sparse BM25 and dense E5, tri-state judgments (positives/negatives/unjudged), and strict rule: *Never convert unjudged into negative by default*.
   - Exact verbatim reference chunk span extraction guidelines (`[char_start, char_end]`).
   - Phased annotation plan: 30-query pilot for IAA calibration $\rightarrow$ 100–200 query production set stratified across topics, languages, and lengths.
   - Anti-leakage group splitting by source domain/near-duplicate cluster between fit and holdout.

4. **Package Exports (`src/evaluation/__init__.py`)**:
   - Exported `build_error_breakdown` and `generate_error_breakdown_summary`.

---

## TDD Evidence

### RED Test Run
Command: `uv run pytest tests/evaluation/test_error_breakdown.py -q`
Output:
```text
=================================== ERRORS ====================================
__________ ERROR collecting tests/evaluation/test_error_breakdown.py __________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\evaluation\test_error_breakdown.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\evaluation\test_error_breakdown.py:16: in <module>
    from src.data.vibiomir_contracts import (
E   ImportError: cannot import name 'ErrorBreakdown' from 'src.data.vibiomir_contracts' (D:\Research\R2AI\mvp-rag-medical\src\data\vibiomir_contracts.py)
=========================== short test summary info ===========================
ERROR tests/evaluation/test_error_breakdown.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.15s
```

### GREEN Test Run
Command: `uv run pytest tests/evaluation/test_error_breakdown.py -q`
Output:
```text
.............                                                            [100%]
13 passed in 0.05s
```

---

## Full Regression Suite Result
Command: `uv run pytest -q`
Output:
```text
........................................................................ [ 20%]
........................................................................ [ 40%]
........................................................................ [ 61%]
........................................................................ [ 81%]
..................................................................       [100%]
354 passed in 15.34s
```

---

## Files Created/Modified
- `src/data/vibiomir_contracts.py`: Added `ErrorBreakdown` dataclass, enhanced `ReferenceChunk` with `char_start`/`char_end`, added `__post_init__` to `DevQueryMetadata`.
- `src/evaluation/error_analysis.py`: Added `build_error_breakdown` and `generate_error_breakdown_summary`.
- `src/evaluation/__init__.py`: Exported `build_error_breakdown` and `generate_error_breakdown_summary`.
- `docs/vibiomir/dev_annotation_guide.md`: Comprehensive annotation, pooling, stratification, and error funnel documentation.
- `tests/evaluation/test_error_breakdown.py`: 13 comprehensive unit tests covering all funnel drop locations, chunk evidence conditions, and aggregate summaries.

---

## Commit Details
- Commit SHA: `6a2a23e`
- Commit Message: `feat: add dev annotation guide and provenance error breakdown (Task 10)`

---

## Self-Review Checklist
- [x] All 4 document drop stages tested and verified (`missing_source_docs`, `missing_candidate_docs`, `lost_after_rerank_docs`, `lost_after_cutoff_docs`).
- [x] Evidence chunk condition verified: in `missing_chunk_refs` always when missed; in `evidence_missing_given_doc` only if parent doc retrieved.
- [x] Chunk matching strictly enforces identical `doc_id` and normalized token LCS ratio $\ge 40\%$.
- [x] Both predicted chunk input paths supported: `store.get_chunk` and `trace.metadata["predicted_chunks"]`.
- [x] Default profile correctly sets `PROVISIONAL_SPEC_V1_PROFILE`.
- [x] Dataset aggregate summary helper implemented and tested.
- [x] Comprehensive documentation in `docs/vibiomir/dev_annotation_guide.md`.
- [x] 0 regressions across full 354-test suite.
