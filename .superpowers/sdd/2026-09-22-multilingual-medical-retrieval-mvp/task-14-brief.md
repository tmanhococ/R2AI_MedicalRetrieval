### Task 14: Submission Exporter & Error Analysis Reporting

**Files:**
- Create: `src/submission/__init__.py`
- Create: `src/submission/base.py`
- Create: `src/submission/internal_jsonl.py`
- Create: `src/submission/competition_csv.py`
- Create: `src/evaluation/error_analysis.py`
- Modify: `src/evaluation/__init__.py`
- Test: `tests/test_submission.py`
- Test: `tests/test_error_analysis.py`

**Interfaces:**
- Consumes: `FinalPrediction`, `GroundTruthRecord`, `QueryRecord` (from `src.data.schemas`)
- Produces:
  - `SubmissionExporter(Protocol)` (in `src/submission/base.py`)
  - `InternalJsonlExporter` (in `src/submission/internal_jsonl.py`)
  - `CompetitionCsvExporter` (in `src/submission/competition_csv.py`)
  - `create_exporter(exporter_type: str) -> SubmissionExporter`
  - `generate_error_analysis_report(predictions, ground_truths, output_dir, queries=None) -> dict[str, Path]` (in `src/evaluation/error_analysis.py`)

**Submission Exporter Specifications:**
1. **Protocol:**
   ```python
   class SubmissionExporter(Protocol):
       def export(self, predictions: Sequence[FinalPrediction], output_path: Path | str) -> Path: ...
   ```
2. **InternalJsonlExporter:**
   - Writes one JSON object per line with exact keys:
     - `query_id: str`
     - `document_ids: list[str]`
     - `chunk_ids: list[str]`
     - `document_scores: dict[str, float]`
     - `chunk_scores: dict[str, float]`
   - Handles parent directory creation automatically.
   - Deterministic query ordering by `query_id`.
3. **CompetitionCsvExporter:**
   - Supports CSV export format with columns: `query_id`, `document_ids` (space- or semicolon-separated), `chunk_ids` (space- or semicolon-separated).
4. **Exporter Factory (`create_exporter`):**
   - Maps `"internal_jsonl"` -> `InternalJsonlExporter`.
   - Maps `"competition_csv"` -> `CompetitionCsvExporter`.

**Error Analysis Reporter Specifications (`src/evaluation/error_analysis.py`):**
1. **Function:**
   ```python
   def generate_error_analysis_report(
       predictions: Sequence[FinalPrediction],
       ground_truths: Sequence[GroundTruthRecord],
       output_dir: Path | str,
       queries: Sequence[QueryRecord] | None = None,
       beta: float = 2.0,
   ) -> dict[str, Path]: ...
   ```
2. **Generated Artifacts in `output_dir`:**
   - `per_query_metrics.csv`:
     Columns: `query_id`, `doc_precision`, `doc_recall`, `doc_f2`, `chunk_precision`, `chunk_recall`, `chunk_f2`, `num_pred_docs`, `num_gt_docs`, `num_pred_chunks`, `num_gt_chunks`.
   - `error_cases.csv`:
     Focuses on queries with $F_2 < 1.0$, detailing:
     - `query_id`, `query_text` (if queries provided), `error_type` (e.g. `doc_fn`, `doc_fp`, `chunk_fn`, `chunk_fp`), `missing_ids`, `spurious_ids`.
   - `summary.json`:
     Macro metrics (`macro_doc_f2`, `macro_chunk_f2`, `combined_f2`), zero-f2 query count, perfect query count, total queries.
3. **Returns:**
   Dictionary mapping artifact names to their `Path` objects (`"per_query_metrics"`, `"error_cases"`, `"summary"`).

**TDD Steps:**
- Step 1: Write failing tests in `tests/test_submission.py` and `tests/test_error_analysis.py`.
- Step 2: Run pytest to verify failure (Verify RED).
- Step 3: Implement exporters in `src/submission/` and error analysis in `src/evaluation/error_analysis.py`.
- Step 4: Run pytest to verify all pass (Verify GREEN).
- Step 5: Commit changes.
