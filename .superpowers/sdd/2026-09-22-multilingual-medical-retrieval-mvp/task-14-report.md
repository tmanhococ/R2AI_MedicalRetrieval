# Task 14 Report: Submission Exporter & Error Analysis Reporting

## Summary
Implemented the submission exporting system and comprehensive error analysis reporting module in accordance with Task 14 specifications.

## What Was Implemented
1. **Submission Exporters (`src/submission/`)**:
   - `SubmissionExporter` (`src/submission/base.py`): A `@runtime_checkable` Python `Protocol` defining `export(predictions: Sequence[FinalPrediction], output_path: Path | str) -> Path`.
   - `InternalJsonlExporter` (`src/submission/internal_jsonl.py`): Formats and exports predictions to internal JSONL with exact schema keys (`query_id`, `document_ids`, `chunk_ids`, `document_scores`, `chunk_scores`). Handles deterministic ordering by `query_id` and automatic parent directory creation.
   - `CompetitionCsvExporter` (`src/submission/competition_csv.py`): Exports predictions to CSV format with columns `query_id`, `document_ids`, `chunk_ids`. Supports configurable ID delimiters (`separator`, default `" "`) and CSV delimiters (`delimiter`, default `","`). Automatically handles directory creation and query sorting.
   - `create_exporter(exporter_type: str, **kwargs)` factory (`src/submission/base.py` & re-exported in `src/submission/__init__.py`): Instantiates exporters (`internal_jsonl`, `competition_csv`) or raises `ValueError` on invalid types.
   - `src/submission/__init__.py`: Cleanly exposes all protocol, implementation, and factory interfaces.

2. **Error Analysis Reporting (`src/evaluation/error_analysis.py`)**:
   - `generate_error_analysis_report(predictions, ground_truths, output_dir, queries=None, beta=2.0) -> dict[str, Path]`:
     - Creates `output_dir` automatically if it does not exist.
     - `per_query_metrics.csv`: Per-query breakdown of doc and chunk precision, recall, and F2 scores, alongside prediction and ground truth counts. Deterministically sorted by `query_id`.
     - `error_cases.csv`: Isolated audit log for imperfect queries ($F_2 < 1.0$) cataloging error cases (`doc_fn`, `doc_fp`, `chunk_fn`, `chunk_fp`), missing IDs, and spurious IDs. Queries with perfect retrieval ($F_2 = 1.0$) are omitted. Optionally displays query text when `queries` are supplied.
     - `summary.json`: Aggregated metrics reporting `macro_doc_f2`, `macro_chunk_f2`, `combined_f2` (average of doc and chunk macro F2), `zero_f2_query_count`, `perfect_query_count`, and `total_queries`.
     - Returns dictionary mapping `"per_query_metrics"`, `"error_cases"`, and `"summary"` to their respective `Path` locations.
   - Updated `src/evaluation/__init__.py` to re-export `generate_error_analysis_report`.

3. **Comprehensive Unit Tests**:
   - `tests/test_submission.py`: Tests Protocol conformance, JSONL output schema & deterministic ordering, CSV delimiter configuration, directory auto-creation, factory creation, and error handling for unknown exporter types.
   - `tests/test_error_analysis.py`: Tests output artifact generation, CSV headers & numerical computations, identification and categorization of false negatives and false positives, query text inclusion, handling of empty inputs and missing predictions, custom beta weighting, and package re-exports.

## TDD Evidence

### 1. RED Phase (Failing Tests)
Before implementing the components, tests were executed and failed as expected:
```
tests/test_submission.py:9: in <module>
    from src.submission.base import SubmissionExporter, create_exporter
E   ModuleNotFoundError: No module named 'src.submission'

tests/test_error_analysis.py:9: in <module>
    from src.evaluation.error_analysis import generate_error_analysis_report
E   ModuleNotFoundError: No module named 'src.evaluation.error_analysis'

============================== 2 errors in 0.18s ==============================
```

### 2. GREEN Phase (Passing Tests)
After implementing all modules:
```
python -m pytest tests/test_submission.py tests/test_error_analysis.py -v
============================= 18 passed in 0.34s ==============================
```
Full suite verification:
```
python -m pytest
======================= 199 passed, 1 warning in 6.27s ========================
```

## Files Changed
- `src/submission/__init__.py` (Created)
- `src/submission/base.py` (Created)
- `src/submission/internal_jsonl.py` (Created)
- `src/submission/competition_csv.py` (Created)
- `src/evaluation/error_analysis.py` (Created)
- `src/evaluation/__init__.py` (Modified)
- `tests/test_submission.py` (Created)
- `tests/test_error_analysis.py` (Created)

## Git Commit
Commit hash: `1acd9eb`
Message: `feat(submission): implement submission exporters and error analysis reporting (Task 14)`

## Self-Review Findings
- All specifications from `task-14-brief.md` are completely met.
- Deterministic ordering by `query_id` ensures reproducible runs and stable diffs across environments.
- Edge cases including empty inputs, missing prediction records, and missing ground-truth records are handled without crashing or raising unexpected exceptions.
- Protocol checking works at runtime via `@runtime_checkable`.

## Concerns
None. The implementation is clean, robust, and passes all 199 tests in the repository without regression.
