# Task 7 Implementation Report: Many-to-many Chunk Counts, Macro Metrics, and Dev Compatibility

## 1. Summary of Implementation

Implemented ViBioMIR evaluation metrics, dev records loader, and cutoff calibrator integration adhering strictly to sections 7.2 and 7.3 of the implementation plan:
1. **ViBioMIR Metrics Engine (`src/evaluation/vibiomir_metrics.py`)**:
   - `evaluate_query`: Computes independent row and column counts using a many-to-many boolean matching matrix (no 1-to-1 or Hungarian matching). Correct predictions are rows with at least one match; found references are columns with at least one match. Evaluates documents using unique ID sets. Handles empty predictions and gold edge cases per evaluation profile (`provisional_spec_v1` and `empty_both_zero`).
   - `evaluate_dataset`: Evaluates all queries in `query_ids`. Retains missing queries in predictions as empty submissions (|P| == 0) so difficult cases are not dropped from the macro average. Computes macro metrics as the arithmetic mean of per-query F2 values (NOT F2 of macro P and macro R). Calculates `inferred_final_mean = (macro_documents.f2 + macro_chunks.f2) / 2.0`.
   - Profiles: `PROVISIONAL_SPEC_V1_PROFILE` (empty/empty F2=1, unverified scorer) and `EMPTY_BOTH_ZERO_PROFILE` (empty/empty F2=0).
2. **Contracts & Validation (`src/data/vibiomir_contracts.py`)**:
   - Added validation to `ReferenceChunk` and `EvaluationRecord` to enforce signed int64 ranges and strictly reject empty chunk text.
   - Defined `MacroMetrics` and `PerQueryMetrics` (supporting both sequence indexing/iteration and query_id dict lookup).
   - Added `profile_fingerprint` property to `EvaluationProfile` and updated `EvaluationSummary` with diagnostic metadata.
3. **Dev Records Loader (`src/evaluation/dev_records.py`)**:
   - `load_dev_records`: Validates dev query ID namespace separation (rejects collisions with official 1,200 IDs), rejects forbidden test/public/private splits to avoid evaluation data leakage, validates store existence for all referenced doc IDs via `SourceStore.contains_official_doc`, and validates non-empty reference chunks.
4. **Calibrator Hardening (`src/evaluation/calibrator.py`)**:
   - Updated `CutoffCalibrator`: Rejects fitting when `fit_enabled=False` or when ground truth records contain test/public/private splits.
   - Updated `CalibrationState`: Stores `profile_fingerprint`, `dataset_fingerprint`, and `model_fingerprint`. Rejects calibration application when fingerprints mismatch.
5. **Addressed Deferred Minor**:
   - In `src/evaluation/vibiomir_text.py`: Handled float conversion cleanly via `Fraction(str(threshold))` in `overlap_matches`.

---

## 2. TDD Evidence

### Step 1 & 2: RED Phase (Failing Tests Before Implementation)

**Command:**
```bash
uv run pytest tests/evaluation/test_vibiomir_metrics.py tests/evaluation/test_dev_provenance.py -q
```

**Output:**
```
=================================== ERRORS ====================================
_________ ERROR collecting tests/evaluation/test_vibiomir_metrics.py __________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\evaluation\test_vibiomir_metrics.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\evaluation\test_vibiomir_metrics.py:6: in <module>
    from src.data.vibiomir_contracts import (
E   ImportError: cannot import name 'MacroMetrics' from 'src.data.vibiomir_contracts' (D:\Research\R2AI\mvp-rag-medical\src\data\vibiomir_contracts.py)
__________ ERROR collecting tests/evaluation/test_dev_provenance.py ___________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\evaluation\test_dev_provenance.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
C:\Users\Lenovo\AppData\Local\Programs\Python\Python312\Lib\importlib\__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests\evaluation\test_dev_provenance.py:16: in <module>
    from src.evaluation.dev_records import load_dev_records
E   ModuleNotFoundError: No module named 'src.evaluation.dev_records'
=========================== short test summary info ===========================
ERROR tests/evaluation/test_vibiomir_metrics.py
ERROR tests/evaluation/test_dev_provenance.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
2 errors in 0.18s
```

### Step 6: GREEN Phase (Passing Tests After Implementation)

**Command:**
```bash
uv run pytest tests/evaluation/test_vibiomir_metrics.py tests/evaluation/test_dev_provenance.py -q
```

**Output:**
```
..............                                                           [100%]
14 passed in 0.16s
```

---

## 3. Full Regression Suite Result

**Command:**
```bash
uv run pytest -q
```

**Output:**
```
........................................................................ [ 24%]
........................................................................ [ 49%]
........................................................................ [ 74%]
........................................................................ [ 99%]
..                                                                       [100%]
290 passed in 8.23s
```
*(Baseline was 276 passed; 14 new tests added, 0 regressions)*

---

## 4. Files Created / Modified

- `src/evaluation/vibiomir_metrics.py` (Created): Many-to-many chunk matching, query & dataset macro metrics, profiles.
- `src/evaluation/dev_records.py` (Created): Provenance-aware loader rejecting official ID collisions and test splits.
- `tests/evaluation/test_vibiomir_metrics.py` (Created): Contract and behavioral tests for metrics calculation.
- `tests/evaluation/test_dev_provenance.py` (Created): Tests for dev namespace separation, store existence, and calibrator anti-leakage.
- `src/data/vibiomir_contracts.py` (Modified): Added validations, `MacroMetrics`, `PerQueryMetrics`, `profile_fingerprint`.
- `src/evaluation/calibrator.py` (Modified): Anti-leakage checks on splits, fingerprint validation, `fit_enabled` guard.
- `src/evaluation/vibiomir_text.py` (Modified): `Fraction(str(threshold))` in `overlap_matches`.
- `src/evaluation/__init__.py` (Modified): Package exports for new evaluation functions.

---

## 5. Commit SHA and Message

- **Commit SHA:** `86abaecaf2bc1a8fdada5243ab94afd8247cd41c` (short: `86abaec`)
- **Message:** `fix: evaluate chunks with independent prediction and reference counts (Task 7)`

---

## 6. Self-Review Checklist

- [x] Many-to-many chunk matching matrix used (no Hungarian matching, no 1-to-1 forcing).
- [x] Correct predicted chunks = count of prediction rows with >= 1 match.
- [x] Found reference chunks = count of reference columns with >= 1 match.
- [x] Chunk P = correct_pred / total_pred; Chunk R = found_ref / total_ref.
- [x] Macro F2 is arithmetic mean of per-query F2 values, not F2 of macro P and macro R.
- [x] Missing queries in predictions evaluated as empty submissions without dropping from macro average.
- [x] Reference chunks with empty text strictly rejected.
- [x] Dev query namespace collision with official 1,200 IDs rejected.
- [x] Calibrator rejects test/public/private splits to prevent evaluation leakage.
- [x] Calibrator validates profile and dataset fingerprints.
- [x] All 290 tests pass with 0 regressions.
