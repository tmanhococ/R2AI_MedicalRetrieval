### Task 13: Macro F2 Metrics & Calibration Engine (Anti-Leakage)

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/metrics.py`
- Create: `src/evaluation/calibrator.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `FinalPrediction`, `GroundTruthRecord`, `ScoredHit` (from `src.data.schemas`)
- Produces:
  - In `src/evaluation/metrics.py`: `calculate_f2`, `calculate_precision_recall_fbeta`, `evaluate_query`, `evaluate_predictions`
  - In `src/evaluation/calibrator.py`: `CutoffCalibrator`, `CalibrationState`
  - In `src/evaluation/__init__.py`: Package exports

**Metric Formulation & Edge Case Guards:**
$$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}, \quad \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
$$F_2 = \frac{(1 + 2^2) \cdot P \cdot R}{2^2 \cdot P + R} = \frac{5 \cdot P \cdot R}{4 \cdot P + R}$$
$$\text{Macro F}_2 = \frac{1}{|Q|} \sum_{q \in Q} F_2(q)$$

Edge Case Rules:
- If ground truth is empty ($|GT| == 0$) and prediction is empty ($|P| == 0$): $F_2 = 1.0, P = 1.0, R = 1.0$.
- If ground truth is empty ($|GT| == 0$) but prediction is non-empty ($|P| > 0$): $F_2 = 0.0, P = 0.0, R = 0.0$.
- If ground truth is non-empty ($|GT| > 0$) but prediction is empty ($|P| == 0$): $F_2 = 0.0, P = 0.0, R = 0.0$.
- If $|Q| == 0$: returns 0.0.

**Cutoff Policies (in `CutoffCalibrator`):**
1. `fixed_top_k`: Selects fixed number of top items $K \in [K_{\min}, K_{\max}]$.
2. `global_threshold`: Keeps items with $\text{score} \ge T$, where $T \in [0.0, 1.0]$.
3. `relative_threshold`: Keeps items with $\text{score} \ge \lambda \cdot \text{score}_{\text{top 1}}$, where $\lambda \in [0.1, 1.0]$.
4. `score_gap`: Identifies the largest score drop $s_i - s_{i+1}$ within bounds $N_{\min} \le i \le N_{\max}$, cutting off at position $i$.

**Calibration Requirements:**
- `CutoffCalibrator(beta=2.0, policies=..., bounds=...)`:
  - `fit(query_scores, ground_truths, item_type="document") -> CalibrationState`
    - Evaluates all candidate policies on validation queries using grid search to find the parameter maximizing Macro F2.
    - Deterministic tie-breaking across policies: higher Macro F2, then higher Recall, then smaller parameter value.
  - `apply(ranked_items, state) -> int`:
    - Returns cutoff index $K$ (number of items to keep) based on the fitted policy.
  - `filter_items(ranked_items, state) -> list`:
    - Returns sliced top items.
- `CalibrationState`:
  - Fields: `item_type`, `best_policy`, `best_params`, `best_f2`, `all_policy_results`.
  - Methods: `to_dict()`, `from_dict()`, `save(filepath)`, `load(filepath)` for JSON serialization.
- Anti-Leakage:
  - Document that fitting must occur strictly on validation data.

**TDD Steps:**
- Step 1: Write failing tests in `tests/test_metrics.py` and `tests/test_calibration.py`.
- Step 2: Run pytest to verify failure (Verify RED).
- Step 3: Implement `src/evaluation/metrics.py`, `src/evaluation/calibrator.py`, and `src/evaluation/__init__.py`.
- Step 4: Run pytest to verify all pass (Verify GREEN).
- Step 5: Commit changes.
