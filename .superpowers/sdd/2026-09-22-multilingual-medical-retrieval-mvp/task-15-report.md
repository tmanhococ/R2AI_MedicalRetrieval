# Task 15 Report: Pipeline CLI Entrypoints & End-to-End Integration Smoke Test

## Implementation Overview
Task 15 ties together all components implemented across Tasks 1–14 into cohesive, production-ready CLI entrypoints and programmatic workflow functions:

1. **`src/cli/index.py` (`run_index`)**:
   - Implements automated dataset ingestion via `JsonlDatasetAdapter`.
   - Validates documents and chunk foreign keys using `validate_corpus()`.
   - Automatically generates chunks using `create_chunker` if missing or mode != "prechunked", persisting generated chunks to `chunks.jsonl` in the index directory for downstream consistency.
   - Builds multilingual partitioned BM25 index via `BM25Indexer` across VI, EN, ZH.
   - Builds dense vector index via `DenseIndexer` across VI, EN, ZH.
   - Supports CLI flags (`--config`, `--dataset`, `--output-dir`).

2. **`src/cli/calibrate.py` (`run_calibrate`)**:
   - Strictly enforces the Anti-Leakage principle by fitting cutoff parameters exclusively on the validation split.
   - Runs the multi-stream retrieval pipeline (routing, expansion, retrieval, weighted RRF, reranking, joint scoring).
   - Fits `CutoffCalibrator` independently for documents and chunks.
   - Serializes optimal policy and parameter choices to `calibration.json`.
   - Supports CLI flags (`--config`, `--dataset`, `--split`, `--output-dir`, `--index-dir`).

3. **`src/cli/evaluate.py` (`run_evaluate`)**:
   - Evaluates predictions on any split against ground truth.
   - Runs retrieval pipeline through joint scoring.
   - Applies calibrated cutoffs (or fallback top-k) and structural consistency (`apply_structural_consistency`).
   - Computes Macro F2 across queries via `evaluate_predictions`.
   - Generates comprehensive error analysis artifacts (`per_query_metrics.csv`, `error_cases.csv`, `summary.json`) via `generate_error_analysis_report`.
   - Exports evaluation metrics to `metrics.json`.
   - Supports CLI flags (`--config`, `--dataset`, `--split`, `--output-dir`, `--index-dir`, `--calibration-path`).

4. **`src/cli/predict.py` (`run_predict`)**:
   - Strictly enforces Anti-Leakage: loads test queries without loading ground truth.
   - Executes multi-stream candidate retrieval, fusion, reranking, and joint scoring.
   - Applies pre-fitted `calibration.json` cutoffs.
   - Applies structural consistency post-processing (`apply_structural_consistency`).
   - Exports final predictions to `submission.jsonl` (or `submission.csv`) via `create_exporter`.
   - Supports CLI flags (`--config`, `--dataset`, `--split`, `--calibration-path`, `--output-file`, `--exporter`, `--index-dir`).

5. **`src/cli/common.py`**:
   - Reusable pipeline execution engine avoiding code duplication across evaluate, calibrate, and predict.
   - Handles text lookup dictionaries, component initialization, and per-query execution orchestration.

6. **`tests/test_e2e_pipeline.py`**:
   - Comprehensive synthetic multilingual integration smoke test with 6 documents (2 VI, 2 EN, 2 ZH), 6 chunks, 2 validation queries + ground truth, and 2 test queries (unlabeled).
   - Validates end-to-end full cycle: `run_index` -> `run_calibrate` -> `run_evaluate` -> `run_predict`.
   - Verifies all generated artifacts (`manifest.json`, `calibration.json`, `metrics.json`, `summary.json`, `submission.jsonl`).
   - Tests CLI `--help` invocation across all 4 entrypoint modules.

---

## TDD Evidence

### RED Phase (Initial Test Failure)
```
tests/test_e2e_pipeline.py::test_full_pipeline_cycle FAILED              [ 50%]
tests/test_e2e_pipeline.py::test_cli_help_flags FAILED                   [100%]

================================== FAILURES ===================================
__________________________ test_full_pipeline_cycle ___________________________
>       from src.cli.index import run_index
E       ModuleNotFoundError: No module named 'src.cli'

tests\test_e2e_pipeline.py:245: ModuleNotFoundError
_____________________________ test_cli_help_flags _____________________________
>           assert result.returncode == 0, f"Failed --help on {ep}: {result.stderr}"
E           AssertionError: Failed --help on src.cli.index: Error while finding module specification for 'src.cli.index' (ModuleNotFoundError: No module named 'src.cli')
======================== 2 failed, 1 warning in 2.54s =========================
```

### GREEN Phase (Implementation Pass)
```
tests/test_e2e_pipeline.py::test_full_pipeline_cycle PASSED              [ 50%]
tests/test_e2e_pipeline.py::test_cli_help_flags PASSED                   [100%]

======================== 2 passed, 1 warning in 19.60s ========================
```

### Full Repository Regression Verification
```
======================= 201 passed, 1 warning in 22.04s =======================
```

---

## Files Changed

- `src/cli/__init__.py`: Exported all 4 entrypoint runner functions.
- `src/cli/common.py`: Shared pipeline orchestration helpers (`load_corpus_and_texts`, `build_retrievers`, `build_query_components`, `build_reranker`, `run_pipeline_for_queries`).
- `src/cli/index.py`: Indexing CLI and `run_index` function.
- `src/cli/calibrate.py`: Calibration CLI and `run_calibrate` function.
- `src/cli/evaluate.py`: Evaluation CLI and `run_evaluate` function.
- `src/cli/predict.py`: Prediction submission CLI and `run_predict` function.
- `src/data/dataset_adapter.py`: Enhanced `JsonlDatasetAdapter` to support directory initialization path seamlessly.
- `src/data/validators.py`: Added `validate_corpus` alias.
- `src/data/__init__.py`: Exported `validate_corpus`.
- `tests/test_e2e_pipeline.py`: Comprehensive end-to-end integration and CLI test suite.

---

## Self-Review Findings
1. **Anti-Leakage Safeguard**:
   - `predict.py` strictly calls `adapter.load_queries(split)` and never invokes `load_ground_truth`.
   - Calibration is fitted exclusively on the validation split in `calibrate.py`.
2. **Determinism and Reproducibility**:
   - All synthetic datasets and pipeline tests use deterministic scoring and ranking.
3. **Robustness**:
   - Chunks generated during indexing are persisted to `output_dir / chunks.jsonl` so downstream stages can reload them without re-chunking.
   - CLI help flags verified for all entrypoints via subprocess invocation.

---

## Concerns
None. All 201 unit and integration tests pass with zero regressions.
