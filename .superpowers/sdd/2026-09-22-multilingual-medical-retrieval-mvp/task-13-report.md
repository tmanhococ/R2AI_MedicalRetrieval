# Task 13 Report: Macro F2 Metrics & Calibration Engine (Anti-Leakage)

## What Was Implemented

1. **`src/evaluation/metrics.py`**:
   - `calculate_f2(precision, recall, beta=2.0) -> float`:
     - Implements exact $F_\beta$ formulation favoring recall:
       $$F_\beta = \frac{(1 + \beta^2) \cdot P \cdot R}{\beta^2 \cdot P + R}$$
       For $\beta = 2.0$:
       $$F_2 = \frac{5 \cdot P \cdot R}{4 \cdot P + R}$$
     - Handles zero cases: returns 0.0 when precision <= 0.0, recall <= 0.0, or denominator <= 0.0.
   - `calculate_precision_recall_fbeta(predicted, ground_truth, beta=2.0) -> dict[str, float]`:
     - Evaluates precision, recall, and $F_\beta$ using set semantics.
     - Implements strict edge-case guards:
       - $|GT| == 0$ and $|P| == 0 \implies F_2 = 1.0, P = 1.0, R = 1.0$
       - $|GT| == 0$ and $|P| > 0 \implies F_2 = 0.0, P = 0.0, R = 0.0$
       - $|GT| > 0$ and $|P| == 0 \implies F_2 = 0.0, P = 0.0, R = 0.0$
   - `evaluate_query(prediction, ground_truth, predicted_chunks=None, ground_truth_chunks=None, beta=2.0) -> dict[str, dict[str, float]]`:
     - Evaluates a single query for both documents and chunks.
     - Accepts typed objects (`FinalPrediction`, `GroundTruthRecord`), dictionaries, or raw ID collections.
   - `evaluate_predictions(predictions, ground_truth, beta=2.0) -> dict[str, Any]`:
     - Aggregates metrics across queries:
       $$\text{Macro } F_2 = \frac{1}{|Q|} \sum_{q \in Q} F_2(q)$$
     - Edge case guard: returns 0.0 for macro metrics when $|Q| == 0$.
     - Missing predictions for a query in ground truth are handled cleanly as empty predictions ($|P| == 0$).
     - Returns document and chunk macro statistics alongside detailed `per_query` breakdowns.

2. **`src/evaluation/calibrator.py`**:
   - **Anti-Leakage Architecture**:
     - Prominently documented at the module and method level: calibration parameters must be fitted strictly on validation queries and NEVER on test evaluation queries to prevent optimistic leakage.
   - `CalibrationState`:
     - Dataclass storing: `item_type`, `best_policy`, `best_params`, `best_f2`, and `all_policy_results`.
     - Methods: `to_dict()`, `from_dict(data)`, `save(filepath)`, and `load(filepath)` supporting JSON serialization roundtrips.
   - `CutoffCalibrator`:
     - Supports 4 cutoff policies:
       1. `fixed_top_k`: Selects fixed number of top items $K \in [K_{\min}, K_{\max}]$.
       2. `global_threshold`: Keeps items with $\text{score} \ge T$, where $T \in [0.0, 1.0]$.
       3. `relative_threshold`: Keeps items with $\text{score} \ge \lambda \cdot \text{score}_{\text{top 1}}$, where $\lambda \in [0.1, 1.0]$.
       4. `score_gap`: Identifies the largest score drop $s_i - s_{i+1}$ within bounds $N_{\min} \le i \le N_{\max}$, cutting off at position $i$.
     - `fit(query_scores, ground_truths, item_type="document") -> CalibrationState`:
       - Grid search across policy parameter candidates to maximize Macro F2.
       - Deterministic tie-breaking across policies: higher Macro F2, then higher Recall, then policy order in candidate policies, then smaller parameter value.
     - `apply(ranked_items, state) -> int`:
       - Returns cutoff index $K$ (number of items to keep).
       - Supports input items as `ScoredHit` objects, `(item_id, score)` tuples, dictionaries, or float scores.
     - `filter_items(ranked_items, state) -> list`:
       - Slices ranked candidate items up to calibrated cutoff index $K$.

3. **`src/evaluation/__init__.py`**:
   - Clean package exports: `calculate_f2`, `calculate_precision_recall_fbeta`, `evaluate_query`, `evaluate_predictions`, `CutoffCalibrator`, `CalibrationState`.

## TDD Evidence

### Step 1 & 2: RED Phase (Failing Tests)
Created comprehensive unit tests in `tests/test_metrics.py` and `tests/test_calibration.py`.
Verification command:
```powershell
python -m pytest tests/test_metrics.py tests/test_calibration.py -v
```
Output:
```
ModuleNotFoundError: No module named 'src.evaluation'
=========================== short test summary info ===========================
ERROR tests/test_metrics.py
ERROR tests/test_calibration.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
============================== 2 errors in 0.18s ==============================
```

### Step 3 & 4: GREEN Phase (Passing Implementation)
Implemented `src/evaluation/metrics.py`, `src/evaluation/calibrator.py`, and `src/evaluation/__init__.py`.
Verification command:
```powershell
python -m pytest tests/test_metrics.py tests/test_calibration.py -v
```
Output:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-8.2.0, pluggy-1.6.0
collected 23 items

tests/test_metrics.py::test_calculate_f2_standard PASSED                 [  4%]
tests/test_metrics.py::test_calculate_f2_perfect PASSED                  [  8%]
tests/test_metrics.py::test_calculate_f2_zero_cases PASSED               [ 13%]
tests/test_metrics.py::test_calculate_f2_custom_beta PASSED              [ 17%]
tests/test_metrics.py::test_calculate_precision_recall_fbeta_standard PASSED [ 21%]
tests/test_metrics.py::test_calculate_precision_recall_fbeta_edge_cases PASSED [ 26%]
tests/test_metrics.py::test_calculate_precision_recall_fbeta_duplicates_handled PASSED [ 30%]
tests/test_metrics.py::test_evaluate_query_with_schemas PASSED           [ 34%]
tests/test_metrics.py::test_evaluate_query_with_raw_collections PASSED   [ 39%]
tests/test_metrics.py::test_evaluate_predictions_empty_ground_truth_guard PASSED [ 43%]
tests/test_metrics.py::test_evaluate_predictions_empty_query_set PASSED  [ 47%]
tests/test_metrics.py::test_evaluate_predictions_macro_averaging PASSED  [ 52%]
tests/test_metrics.py::test_evaluate_predictions_missing_prediction_handled_as_empty PASSED [ 56%]
tests/test_metrics.py::test_evaluate_predictions_dict_inputs PASSED      [ 60%]
tests/test_calibration.py::test_calibrator_selects_best_policy PASSED    [ 65%]
tests/test_calibration.py::test_cutoff_policies_apply_logic PASSED       [ 69%]
tests/test_calibration.py::test_calibrator_with_scored_hit_objects PASSED [ 73%]
tests/test_calibration.py::test_calibration_state_serialization_roundtrip PASSED [ 78%]
tests/test_calibration.py::test_calibrator_chunk_item_type PASSED        [ 82%]
tests/test_anti_leakage_documentation_present PASSED                     [ 86%]
tests/test_calibration.py::test_calibrator_custom_bounds_support PASSED  [ 91%]
tests/test_calibration.py::test_calibrator_empty_and_single_edge_cases PASSED [ 95%]
tests/test_calibration.py::test_calibrator_deterministic_tie_breaking PASSED [100%]

============================= 23 passed in 0.13s ==============================
```

Full test suite regression check:
```powershell
python -m pytest -v
```
Output:
```
======================= 174 passed, 1 warning in 5.22s ========================
```

## Files Changed
- `src/evaluation/__init__.py` (Created)
- `src/evaluation/metrics.py` (Created)
- `src/evaluation/calibrator.py` (Created)
- `tests/test_metrics.py` (Created)
- `tests/test_calibration.py` (Created)

## Git Commit
Commit hash: `7c44712`
Message: `feat: implement macro f2 evaluation and anti leakage cutoff calibrator`

## Self-Review Findings
- **Mathematical Correctness**: Checked formula $\frac{5 \cdot P \cdot R}{4 \cdot P + R}$ against standard manual test points ($P=0.5, R=1.0 \implies 5/6 = 0.8333$).
- **Edge Case Adherence**: Verified the exact specified behavior for empty ground truth and empty prediction ($F_2 = 1.0$), empty ground truth with predictions ($F_2 = 0.0$), non-empty ground truth with empty predictions ($F_2 = 0.0$), and empty evaluation query set ($|Q| = 0 \implies 0.0$).
- **Anti-Leakage Safeguards**: Rigorously documented that fitting should occur strictly on validation queries.
- **Type Interoperability**: Handled `ScoredHit` objects, `(id, score)` tuples, dictionaries, and raw floats transparently across both `metrics.py` and `calibrator.py`.
- **Deterministic Grid & Tie-Breaking**: Deterministically sorted candidates by `(-macro_f2, -macro_recall, policy_index, param_sort_key)`.

## Concerns
None. All 174 tests across the entire project pass with 0 regressions.
