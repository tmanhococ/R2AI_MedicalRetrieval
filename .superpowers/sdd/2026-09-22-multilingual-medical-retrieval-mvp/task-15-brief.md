### Task 15: Pipeline CLI Entrypoints & End-to-End Integration Smoke Test

**Files:**
- Create: `src/cli/__init__.py`
- Create: `src/cli/index.py`
- Create: `src/cli/evaluate.py`
- Create: `src/cli/calibrate.py`
- Create: `src/cli/predict.py`
- Test: `tests/test_e2e_pipeline.py`

**Interfaces:**
- Consumes: All modules implemented across Tasks 1–14
- Produces:
  - `src.cli.index`: `run_index(config, dataset_dir, output_dir=None)` + CLI parser
  - `src.cli.evaluate`: `run_evaluate(config, split, dataset_dir, output_dir=None, ...)` + CLI parser
  - `src.cli.calibrate`: `run_calibrate(config, split, dataset_dir, output_dir=None, ...)` + CLI parser
  - `src.cli.predict`: `run_predict(config, split, dataset_dir, calibration_path=None, output_file=None, exporter=None, ...)` + CLI parser
  - `tests/test_e2e_pipeline.py`: Comprehensive end-to-end integration smoke test running Index -> Evaluate -> Calibrate -> Predict on a synthetic multilingual corpus.

**CLI Specifications & Requirements:**

1. **`src/cli/index.py`**:
   - CLI flags: `--config` (default `"configs/r2ai_mvp.yaml"`), `--dataset` (required), `--output-dir` (optional).
   - Python entrypoint: `python -m src.cli.index --config configs/r2ai_mvp.yaml --dataset <path>`.
   - Programmatic function: `run_index(config: dict, dataset_dir: Path | str, output_dir: Path | str | None = None) -> dict[str, Any]`.
   - Workflow:
     1. Load corpus via `JsonlDatasetAdapter(dataset_dir)`.
     2. Validate documents and chunks via `validate_corpus()`.
     3. Apply `create_chunker(config.get("indexing", {}).get("chunking", {}))` if chunks missing or mode != "prechunked".
     4. Build BM25 index via `BM25Indexer` across languages.
     5. Build Dense index via `DenseIndexer` across languages.
     6. Returns dictionary with index paths and stats.

2. **`src/cli/evaluate.py`**:
   - CLI flags: `--config`, `--dataset`, `--split` (default `"validation"`), `--output-dir` (optional), `--index-dir` (optional).
   - Python entrypoint: `python -m src.cli.evaluate --config configs/r2ai_mvp.yaml --split validation --dataset <path>`.
   - Programmatic function: `run_evaluate(config: dict, split: str = "validation", dataset_dir: Path | str = ..., output_dir: Path | str | None = None, index_dir: Path | str | None = None) -> dict[str, Any]`.
   - Workflow:
     1. Load queries and ground truth for `split` via `JsonlDatasetAdapter`.
     2. Process queries via `AdaptiveQueryRouter` and `MultilingualQueryBuilder`.
     3. Load `BM25Retriever` and `DenseRetriever` from index directories.
     4. Execute 6 retrieval streams via `RetrievalOrchestrator`.
     5. Fuse streams via `fuse_stream_results` with weights from config.
     6. Rerank via `CrossEncoderReranker` (or `DummyReranker`) with chunk aggregation.
     7. Compute joint scores via `JointScorer`.
     8. Apply cutoff threshold/top-k.
     9. Apply structural consistency via `apply_structural_consistency`.
     10. Compute metrics via `evaluate_predictions`.
     11. Generate error analysis report via `generate_error_analysis_report`.
     12. Save `metrics.json` and return evaluation metrics dictionary.

3. **`src/cli/calibrate.py`**:
   - CLI flags: `--config`, `--dataset`, `--split` (default `"validation"`), `--output-dir` (optional), `--index-dir` (optional).
   - Python entrypoint: `python -m src.cli.calibrate --config configs/r2ai_mvp.yaml --split validation --dataset <path>`.
   - Programmatic function: `run_calibrate(config: dict, split: str = "validation", dataset_dir: Path | str = ..., output_dir: Path | str | None = None, index_dir: Path | str | None = None) -> dict[str, Any]`.
   - Workflow:
     1. Load queries and ground truth for validation split.
     2. Run retrieval pipeline up to joint scoring.
     3. Fit `CutoffCalibrator` independently for documents and chunks.
     4. Save `calibration.json` (containing document and chunk states).
     5. Returns calibration summary.

4. **`src/cli/predict.py`**:
   - CLI flags: `--config`, `--dataset`, `--split` (default `"test"`), `--calibration-path` (optional), `--output-file` (optional), `--exporter` (default `"internal_jsonl"`), `--index-dir` (optional).
   - Python entrypoint: `python -m src.cli.predict --config configs/r2ai_mvp.yaml --split test --dataset <path> --calibration-path <path> --output-file <path>`.
   - Programmatic function: `run_predict(config: dict, split: str = "test", dataset_dir: Path | str = ..., calibration_path: Path | str | None = None, output_file: Path | str | None = None, exporter_type: str = "internal_jsonl", index_dir: Path | str | None = None) -> Path`.
   - Workflow:
     1. Load test queries (Anti-Leakage: do NOT load ground truth).
     2. Run retrieval pipeline up to joint scoring.
     3. Load `calibration.json` if provided, else use default config cutoff.
     4. Filter documents and chunks with calibrated cutoff.
     5. Apply structural consistency post-processing.
     6. Export predictions via `create_exporter(exporter_type).export(predictions, output_file)`.
     7. Return output file `Path`.

5. **`tests/test_e2e_pipeline.py`**:
   - Builds a complete synthetic dataset:
     - 2 VI documents, 2 EN documents, 2 ZH documents with chunks.
     - 2 validation queries with ground truth.
     - 2 test queries (unlabeled).
   - Runs full cycle:
     1. `run_index`: creates indexes in temp dir.
     2. `run_calibrate`: fits calibration on validation split and saves `calibration.json`.
     3. `run_evaluate`: verifies Macro F2 calculation on validation split.
     4. `run_predict`: generates predictions for test split using calibration, exports `submission.jsonl`, and asserts file exists with valid JSONL lines.
   - Verifies CLI argument parsing for `--help` or invocation across all 4 entrypoints.

**TDD Steps:**
- Step 1: Write failing integration test in `tests/test_e2e_pipeline.py`.
- Step 2: Run pytest to verify failure (Verify RED).
- Step 3: Implement `src/cli/` modules (`index.py`, `evaluate.py`, `calibrate.py`, `predict.py`, `__init__.py`).
- Step 4: Run pytest to verify all pass (Verify GREEN).
- Step 5: Commit changes.
