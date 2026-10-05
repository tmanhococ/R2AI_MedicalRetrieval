### Task 12: Joint Scoring & Structural Consistency Post-Processing

**Files:**
- Create: `src/scoring/__init__.py`
- Create: `src/scoring/joint_scoring.py`
- Create: `src/scoring/consistency.py`
- Test: `tests/test_joint_scoring.py`
- Test: `tests/test_consistency.py`

**Interfaces:**
- Consumes: `ScoredHit` (from `src.data.schemas`), `min_max_normalize` (from `src.reranking`)
- Produces:
  - `compute_joint_scores(doc_hits, chunk_hits, ...)`
  - `JointScorer` class
  - `apply_structural_consistency(selected_docs, selected_chunks, chunk_to_doc, mode=...)`
  - `ConsistencyMode` enum or Literal

**Mathematical Specifications:**

1. **Base Score Normalization:**
   Before joint combination, ensure `base_doc_score` and `base_chunk_score` are normalized to $[0, 1]$ across the query candidates (using `min_max_normalize`).

2. **Final Document Score:**
   $$\text{final\_doc\_score}(d) = \alpha \cdot \text{base\_doc\_score}(d) + \beta \cdot \max_{c \in d}(\text{base\_chunk\_score}(c)) + \gamma \cdot \text{mean}(\text{top\_m\_chunk\_scores}(d))$$
   - Default hyperparameters:
     $\alpha = 0.70$ (`base_weight`)
     $\beta = 0.20$ (`best_chunk_weight`)
     $\gamma = 0.10$ (`top_chunks_mean_weight`)
     $m = 3$ (`top_m_chunks`)
   - If document $d$ has no matching chunk candidates: chunk evidence is $0.0$.
   - If document $d$ has $k < m$ chunks: mean is computed across all $k$ available chunks.

3. **Final Chunk Score:**
   $$\text{final\_chunk\_score}(c) = \delta \cdot \text{base\_chunk\_score}(c) + \epsilon \cdot \text{final\_doc\_score}(\text{parent\_doc}(c))$$
   - Default hyperparameters:
     $\delta = 0.85$ (`base_weight`)
     $\epsilon = 0.15$ (`parent_document_weight`)
   - If parent document is not present in document candidates: parent doc score defaults to $0.0$ (or min doc score).

4. **Output Dataclass:**
   Returns `tuple[list[ScoredHit], list[ScoredHit]]` (joint documents, joint chunks).
   - In returned `ScoredHit`:
     - `normalized_score` holds the newly computed final joint score.
     - `rerank_score` preserves original rerank score.
     - `raw_fusion_score` preserves original fusion score.
   - Outputs sorted descending by `normalized_score` with deterministic tie-breaking `(-normalized_score, -raw_fusion_score, item_id)`.

5. **Structural Consistency Post-Processing (`src/scoring/consistency.py`):**
   ```python
   def apply_structural_consistency(
       selected_docs: list[str] | list[ScoredHit],
       selected_chunks: list[str] | list[ScoredHit],
       chunk_to_doc: dict[str, str] | None = None,
       mode: str = "independent", # "independent" | "enforce_parent_doc" | "filter_orphan_chunks"
   ) -> tuple[list[str] | list[ScoredHit], list[str] | list[ScoredHit]]: ...
   ```
   - Modes:
     - `"independent"`: No structural modification. Both sets remain unchanged.
     - `"enforce_parent_doc"`: For each chunk in `selected_chunks`, if its parent `doc_id` is missing from `selected_docs`, append parent `doc_id` to `selected_docs` (preserving existing order, adding missing parent docs deterministically at the end).
     - `"filter_orphan_chunks"`: For each chunk in `selected_chunks`, if its parent `doc_id` is NOT in `selected_docs`, filter out that chunk.
   - Robustness:
     - Supports both raw ID strings (`list[str]`) and `ScoredHit` objects.
     - If `chunk_to_doc` is not passed and `selected_chunks` consists of `ScoredHit`, extracts `doc_id` directly from `hit.doc_id`.
     - Preserves uniqueness (no duplicate doc_ids or chunk_ids).

6. **Package Exports (`src/scoring/__init__.py`):**
   - Export `compute_joint_scores`, `JointScorer`, `apply_structural_consistency`.

**TDD Steps:**
- Step 1: Write failing tests in `tests/test_joint_scoring.py` and `tests/test_consistency.py`.
- Step 2: Run pytest to verify failure (Verify RED).
- Step 3: Implement `src/scoring/joint_scoring.py`, `src/scoring/consistency.py`, and `src/scoring/__init__.py`.
- Step 4: Run pytest to verify all pass (Verify GREEN).
- Step 5: Commit changes.
