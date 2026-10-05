# Task 9 Report: CLI End-to-End, Inference Checkpoint và Reports

## Summary of Implementation
Implemented the end-to-end CLI workflow, per-query inference checkpointing with crash-safe resume, anti-leakage guards, config validation with hierarchical precedence, and reporting:
1. **Config Validation and Precedence (`src/utils/config.py`)**:
   - `merge_configs`: Deep recursive merge supporting overlay configurations and runtime CLI overrides.
   - `validate_config`: Strict schema whitelist checking ensuring unknown top-level sections or unrecognized subkeys raise `ValueError`.
   - Precedence: CLI overrides > YAML config > code defaults.
2. **Inference Checkpoint Manager (`src/utils/checkpoint.py`)**:
   - `compute_run_fingerprint`: Deterministic SHA-256 fingerprint encompassing query IDs, corpus membership hash, index manifests, model revisions, and core pipeline parameters.
   - `InferenceCheckpointManager`: Per-query transactional checkpoints written to `predictions_checkpoint.jsonl`.
   - Safe resume skips already completed queries with matching fingerprint.
   - Rejects resume attempts when `run_fingerprint` mismatches.
   - Failure tracking: Inference failures are explicitly recorded as `status="failed"` and strictly BLOCK final export (never silently converted to empty predictions).
3. **CLI Commands Implementation & Wiring (`src/cli/`)**:
   - `src/cli/ingest_vibiomir.py`: Added `--config` support, overriding dataset paths and expected query counts.
   - `src/cli/crawl.py`: Added `--config`, `--pilot-size`, `--resume`, `--seed` support and generates `coverage.json`.
   - `src/cli/extract.py`: Created offline document extraction utility checking raw SHA-256, quality classification, and reporting `ExtractionReport`.
   - `src/cli/build_snapshot.py`: Created CLI entrypoint to freeze corpus into immutable Parquet shards and `lookup.sqlite`.
   - `src/cli/validate_submission.py`: Created offline validation utility verifying submission ZIP archives against frozen snapshot and emitting `validation_report.json`.
   - `src/cli/index.py`: Validated dual index building from immutable snapshot directories.
   - `src/cli/evaluate.py`: Wired `--snapshot`, `--split dev`, loading ground truth via `dev_records.py` and calculating official metrics via `vibiomir_metrics.py`. Enforces anti-leakage: refuses evaluation on unlabeled test split.
   - `src/cli/predict.py`: Integrated per-query checkpointing, snapshot lookups, `run_manifest.json`, `retrieval_debug.jsonl`, and `--export-submission` via `VibiomirExporter`.

---

## TDD Evidence

### RED Test Run
Command: `uv run pytest tests/integration/test_vibiomir_phase1.py -q`
Output:
```text
=================================== ERRORS ====================================
_________ ERROR collecting tests/integration/test_vibiomir_phase1.py __________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\integration\test_vibiomir_phase1.py'.
...
E   ImportError: cannot import name 'merge_configs' from 'src.utils.config' (D:\Research\R2AI\mvp-rag-medical\src\utils\config.py)
=========================== short test summary info ===========================
ERROR tests/integration/test_vibiomir_phase1.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.47s
```

### GREEN Test Run
Command: `uv run pytest tests/integration/test_vibiomir_phase1.py -v`
Output:
```text
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-9.1.1, pluggy-1.6.0
collected 10 items

tests/integration/test_vibiomir_phase1.py::test_config_merge_cli_yaml_defaults PASSED [ 10%]
tests/integration/test_vibiomir_phase1.py::test_unknown_yaml_key_rejected PASSED [ 20%]
tests/integration/test_vibiomir_phase1.py::test_unit_e2e_pipeline_with_aliases_and_fetch_failure PASSED [ 30%]
tests/integration/test_vibiomir_phase1.py::test_inference_kill_and_resume_skips_completed_queries PASSED [ 40%]
tests/integration/test_vibiomir_phase1.py::test_inference_failure_records_failed_state_and_blocks_export PASSED [ 50%]
tests/integration/test_vibiomir_phase1.py::test_resume_with_mismatched_run_fingerprint_rejected PASSED [ 60%]
tests/integration/test_vibiomir_phase1.py::test_test_set_without_labels_does_not_generate_fake_zero_metrics PASSED [ 70%]
tests/integration/test_vibiomir_phase1.py::test_missing_language_stream_does_not_crash PASSED [ 80%]
tests/integration/test_vibiomir_phase1.py::test_no_candidate_query_returns_clean_empty PASSED [ 90%]
tests/integration/test_vibiomir_phase1.py::test_evaluate_on_dev_split_with_snapshot_and_anti_leakage PASSED [100%]

============================= 10 passed in 3.10s ==============================
```

---

## Full Regression Suite Result
Command: `uv run pytest -q`
Output:
```text
........................................................................ [ 21%]
........................................................................ [ 42%]
........................................................................ [ 63%]
........................................................................ [ 84%]
....................................................                     [100%]
340 passed in 10.78s
```
Zero regressions across the existing suite (330 passed -> 340 passed).

---

## Files Created / Modified
- `src/utils/config.py` (Modified): Added `ALLOWED_CONFIG_SECTIONS`, `merge_configs`, `validate_config`, and overlay base merge logic in `load_config`.
- `src/utils/checkpoint.py` (Created): Added `compute_run_fingerprint` and `InferenceCheckpointManager` with resume and failure blocking.
- `src/cli/ingest_vibiomir.py` (Modified): Added `--config`, CLI overrides, and `run_ingest`.
- `src/cli/crawl.py` (Modified): Added `--config`, `--pilot-size`, `--resume`, `--seed`, `run_crawl_cli`, and `coverage.json` generator.
- `src/cli/extract.py` (Created): Offline document extraction and quality classification CLI.
- `src/cli/build_snapshot.py` (Created): Snapshot freezing CLI.
- `src/cli/validate_submission.py` (Created): Offline submission ZIP validation and `validation_report.json` CLI.
- `src/cli/index.py` (Modified): Verified snapshot indexing and fixed typing imports.
- `src/cli/evaluate.py` (Modified): Added snapshot and `dev_records` support, official metrics computation, and anti-leakage checks.
- `src/cli/predict.py` (Modified): Added per-query checkpointing with `run_fingerprint`, snapshot retrieval, and submission export.
- `tests/integration/test_vibiomir_phase1.py` (Created): End-to-end integration tests (unit fixture with 3 queries, 2 aliases, 1 failure, checkpoints, anti-leakage, edge cases).

---

## Commit Details
- Commit: `0624030`
- Message: `feat: wire Phase 1 ingestion prediction and submission workflow (Task 9)`

---

## Self-Review Checklist
- [x] Strict TDD followed: tests written first, verified RED failure, implemented code, verified GREEN.
- [x] Pipeline handles 3 queries (1, 4, 9), 2 URL aliases, 1 fetch failure in < 3s without GPU or network.
- [x] Checkpoint kill and resume skips completed queries with identical `run_fingerprint`.
- [x] Checkpoint mismatch on resume is rejected with `ValueError`.
- [x] Query inference failures are recorded and strictly BLOCK export.
- [x] Config merge precedence verified: CLI overrides > YAML config > code defaults. Unknown YAML keys rejected.
- [x] Anti-leakage principle enforced: test set without labels does not produce fake "0" metrics.
- [x] Full regression test suite passed with 0 regressions (340 passed).
