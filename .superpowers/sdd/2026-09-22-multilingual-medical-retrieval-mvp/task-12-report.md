# Task 12 Report: Joint Scoring & Structural Consistency Post-Processing

## What Was Implemented

1. **`src/scoring/joint_scoring.py`**:
   - `compute_joint_scores(doc_hits, chunk_hits, ...)`:
     - **Base Score Normalization**: Handled via `_normalize_or_pass_scores` utilizing `min_max_normalize` from `src.reranking`. Supports `"auto"`, `True`, and `False`.
     - **Final Document Score**:
       $$\text{final\_doc\_score}(d) = \alpha \cdot \text{base\_doc\_score}(d) + \beta \cdot \max_{c \in d}(\text{base\_chunk\_score}(c)) + \gamma \cdot \text{mean}(\text{top\_m\_chunk\_scores}(d))$$
       Defaults: $\alpha=0.70$ (`base_weight`), $\beta=0.20$ (`best_chunk_weight`), $\gamma=0.10$ (`top_chunks_mean_weight`), $m=3$ (`top_m_chunks`).
       Handles edge cases: documents with no matching chunk candidates have 0.0 chunk evidence; documents with $k < m$ chunks average over the $k$ available chunks.
     - **Final Chunk Score**:
       $$\text{final\_chunk\_score}(c) = \delta \cdot \text{base\_chunk\_score}(c) + \epsilon \cdot \text{final\_doc\_score}(\text{parent\_doc}(c))$$
       Defaults: $\delta=0.85$ (`base_weight`), $\epsilon=0.15$ (`parent_document_weight`).
       Handles edge cases: missing parent document in candidates defaults to 0.0 (`parent_doc_fallback`).
     - **Output Dataclass & Sorting**: Returns `tuple[list[ScoredHit], list[ScoredHit]]`. Output hits preserve `item_id`, `item_type`, `doc_id`, `language`, `rerank_score`, and `raw_fusion_score`, while `normalized_score` stores the final joint score. Sorted descending by `(-normalized_score, -raw_fusion_score, item_id)`.
   - `JointScorer`:
     - Encapsulates hyperparameters.
     - `from_config(config_dict)` method for configuration-driven instantiation from YAML.
     - Callable `score(doc_hits, chunk_hits)` and `__call__(doc_hits, chunk_hits)`.

2. **`src/scoring/consistency.py`**:
   - `ConsistencyMode` enum: `INDEPENDENT = "independent"`, `ENFORCE_PARENT_DOC = "enforce_parent_doc"`, `FILTER_ORPHAN_CHUNKS = "filter_orphan_chunks"`.
   - `apply_structural_consistency(selected_docs, selected_chunks, chunk_to_doc=None, mode="independent", available_docs=None)`:
     - `"independent"`: Preserves input selections without modification.
     - `"enforce_parent_doc"`: Appends missing parent document IDs for any selected chunks to `selected_docs`, preserving original order and deterministically appending at the end. Supports both `list[str]` and `list[ScoredHit]` (restores from `available_docs` or synthesizes valid `ScoredHit`).
     - `"filter_orphan_chunks"`: Removes chunks whose parent document ID is not in `selected_docs`.
     - Supports both `list[str]` IDs and `list[ScoredHit]` objects.
     - Handles deduplication and preserves ordering.

3. **`src/scoring/__init__.py`**:
   - Clean module exports: `compute_joint_scores`, `JointScorer`, `apply_structural_consistency`, `ConsistencyMode`.

## TDD Evidence

### Step 1 & 2: RED Phase (Failing Tests)
Created comprehensive unit tests in `tests/test_joint_scoring.py` and `tests/test_consistency.py`.
Verification command:
```powershell
python -m pytest tests/test_joint_scoring.py tests/test_consistency.py -v
```
Output:
```
ModuleNotFoundError: No module named 'src.scoring'
=========================== short test summary info ===========================
ERROR tests/test_joint_scoring.py
ERROR tests/test_consistency.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
============================== 2 errors in 0.18s ==============================
```

### Step 3 & 4: GREEN Phase (Passing Implementation)
Implemented `src/scoring/joint_scoring.py`, `src/scoring/consistency.py`, and `src/scoring/__init__.py`.
Verification command:
```powershell
python -m pytest tests/test_joint_scoring.py tests/test_consistency.py -v
```
Output:
```
collected 21 items

tests/test_joint_scoring.py::test_joint_scoring_strengthens_evidence PASSED [  4%]
tests/test_joint_scoring.py::test_exact_formula_calculation PASSED       [  9%]
tests/test_joint_scoring.py::test_doc_without_matching_chunks PASSED     [ 14%]
tests/test_joint_scoring.py::test_doc_with_fewer_than_m_chunks PASSED    [ 19%]
tests/test_joint_scoring.py::test_chunk_with_missing_parent_doc PASSED   [ 23%]
tests/test_joint_scoring.py::test_unnormalized_raw_scores_auto_normalized PASSED [ 28%]
tests/test_joint_scoring.py::test_deterministic_tie_breaking_and_sorting PASSED [ 33%]
tests/test_joint_scoring.py::test_empty_inputs PASSED                    [ 38%]
tests/test_joint_scoring.py::test_joint_scorer_class_and_from_config PASSED [ 42%]
tests/test_joint_scoring.py::test_fields_preserved_in_scored_hits PASSED [ 47%]
tests/test_consistency.py::test_independent_mode_strings PASSED          [ 52%]
tests/test_consistency.py::test_independent_mode_scored_hits PASSED      [ 57%]
tests/test_consistency.py::test_enforce_parent_doc_mode_strings PASSED   [ 61%]
tests/test_consistency.py::test_enforce_parent_doc_mode_scored_hits PASSED [ 66%]
tests/test_consistency.py::test_enforce_parent_doc_synthesizes_hit_if_unavailable PASSED [ 71%]
tests/test_consistency.py::test_enforce_parent_doc_already_present PASSED [ 76%]
tests/test_filter_orphan_chunks_mode_strings PASSED                       [ 80%]
tests/test_consistency.py::test_filter_orphan_chunks_mode_scored_hits PASSED [ 85%]
tests/test_consistency.py::test_deduplication_preserves_order PASSED     [ 90%]
tests/test_consistency.py::test_invalid_mode_raises_error PASSED         [ 95%]
tests/test_consistency.py::test_empty_inputs_consistency PASSED          [100%]

============================= 21 passed in 0.09s ==============================
```

Full test suite regression check:
```powershell
python -m pytest
```
Output:
```
======================= 157 passed, 1 warning in 4.88s ========================
```

## Files Changed
- `src/scoring/__init__.py` (Created)
- `src/scoring/joint_scoring.py` (Created)
- `src/scoring/consistency.py` (Created)
- `tests/test_joint_scoring.py` (Created)
- `tests/test_consistency.py` (Created)

## Self-Review Findings
- **Formula Accuracy**: Validated against exact numeric examples for both document combination ($\alpha \cdot \text{doc} + \beta \cdot \max + \gamma \cdot \text{mean}$) and chunk boost ($\delta \cdot \text{chunk} + \epsilon \cdot \text{parent\_doc}$).
- **Bidirectional Information Flow**: Document scores are updated first with chunk evidence, and chunk scores properly incorporate the updated `final_doc_score`.
- **Tie-Breaking Determinism**: Sorted descending by `(-normalized_score, -raw_fusion_score, item_id)`.
- **Type Flexibility & Robustness**: `apply_structural_consistency` supports both `list[str]` and `list[ScoredHit]`, gracefully extracts `doc_id` when `chunk_to_doc` mapping is omitted, and restores hits from `available_docs` when appending parent docs.
- **Config-Driven**: `JointScorer.from_config` seamlessly reads the `joint_scoring` block from `configs/r2ai_mvp.yaml`.

## Concerns
None. All 157 tests in the project pass with 0 regressions.
