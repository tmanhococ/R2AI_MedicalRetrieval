# Task 0 Implementation Report: Audit Repo Thật và Cố Định Integration Map

- **Task Name:** Task 0: Audit repo thật và cố định integration map
- **Branch:** `feature/vibiomir-phase-1`
- **Base Commit:** `7e7ed6b406a41c96a60298a4c358503103b73979`
- **Commit Created:** `0963a6a` (`docs: create ViBioMIR phase 1 integration map (Task 0)`)
- **Status:** DONE

---

## 1. Scope & Audited Codebase

1. **Repo Entrypoints & CLI:**
   - Existing CLI scripts in `src/cli/`: `index.py`, `calibrate.py`, `evaluate.py`, `predict.py`, and `common.py`.
   - Proposed new CLI scripts in Section 11: `ingest_vibiomir.py`, `crawl.py`, `extract.py`, `build_snapshot.py`, `validate_submission.py`.
2. **Configuration & Dependencies:**
   - Current configuration: `configs/r2ai_mvp.yaml` parsed via `src/utils/config.py`.
   - Proposed Phase 1 overlay: `configs/vibiomir_phase1.yaml`.
   - Dependency management: Managed by `uv` (`pyproject.toml` and `uv.lock`). Virtual environment in `.venv` (Python 3.12).
   - Audited heavy packages: `faiss`, `sentence_transformers`, `torch`, `transformers`, `bm25s`, `rank_bm25` are guarded with try/except fallbacks and tested using mocks (`DummyEncoder`, `DummyReranker`, NumPy fallback).
3. **Existing Schemas & Adapters:**
   - `src/data/schemas.py`: `QueryRecord`, `DocumentRecord`, `ChunkRecord`, `GroundTruthRecord`, `SearchHit`, `FusedHit`, `ScoredHit`, `FinalPrediction`.
   - `src/data/dataset_adapter.py`: Protocol `DatasetAdapter`, implementations `JsonlDatasetAdapter`, `InMemoryDatasetAdapter`.
4. **AGENTS.md Audit:**
   - Explicitly searched entire repo: no `AGENTS.md` exists.
5. **Section 11 Mapping:**
   - Mapped all 29 files/modules listed in Section 11 of the plan to exact paths in the repository.

---

## 2. Verification Results

- **Test Command:** `uv run pytest -q`
- **Exit Code:** `0`
- **Output:**
  ```
  ........................................................................ [ 35%]
  ........................................................................ [ 71%]
  .........................................................                [100%]
  ============================== warnings summary ===============================
  C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\site-packages\faiss\loader.py:49
    C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\site-packages\faiss\loader.py:49: DeprecationWarning: numpy.core._multiarray_umath is deprecated and has been renamed to numpy._core._multiarray_umath. The numpy._core namespace contains private NumPy internals and its use is discouraged, as NumPy internals can change without warning in any release. In practice, most real-world usage of numpy.core is to access functionality in the public NumPy API. If that is the case, use the public NumPy API. If not, you are using NumPy internals. If you would still like to access an internal attribute, use numpy._core._multiarray_umath.__cpu_features__.
      from numpy.core._multiarray_umath import __cpu_features__

  -- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
  201 passed, 1 warning in 17.20s
  ```
- **Failures / Regressions:** 0 failures across all 201 existing tests.

---

## 3. Files Changed / Created

- **Created & Committed:**
  - `docs/vibiomir/phase1_integration_map.md` (Detailed 8-section audit and mapping document)
- **Report File:**
  - `.superpowers/sdd/2026-10-04-vibiomir-phase-1-implementation-plan/task-0-report.md`

---

## 4. Self-Review Findings

1. **Commit Integrity:** Document was committed directly into the actual working git tree on `feature/vibiomir-phase-1`.
2. **Interface Boundaries:** Core schemas (`schemas.py`) and dataset adapters (`dataset_adapter.py`) remain untouched, ensuring all existing tests pass without disruption.
3. **ID Representation Alignment:** Documented deterministic mapping between internal strings and official signed int64 integers (`583` <-> `"583"`).
4. **Architectural Gaps:** Explicitly isolated the 8 capability gaps (Parquet batch ingestion, SQLite registry, crawl scheduler, Trafilatura/PyMuPDF extraction, exact source spans, immutable snapshots, BTC LCS evaluator, and official ZIP exporter) to be implemented incrementally in Tasks 1-11.
