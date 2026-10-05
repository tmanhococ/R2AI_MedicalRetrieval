### Task 10: Weighted Reciprocal Rank Fusion (Weighted RRF)

**Files:**
- Create: `src/fusion/__init__.py`
- Create: `src/fusion/weighted_rrf.py`
- Test: `tests/test_weighted_rrf.py`

**Interfaces:**
- Consumes: `SearchHit`, `FusedHit` (from `src.data.schemas`), `StreamResults` (from `src.retrieval.orchestrator`)
- Produces: `weighted_rrf`, `WeightedRRF`, `fuse_stream_results`

**Formula & Algorithm:**
$$\text{RRF\_Score}(item) = \sum_{s \in S} \frac{w_s}{k + \text{rank}_s(item)}$$
Where:
- $S$ is the set of streams containing the candidate item.
- $w_s$ is the weight of stream $s$ (default 1.0 if not specified in `weights`).
- $k$ is the RRF smoothing constant (default `rrf_k = 60`).
- $\text{rank}_s(item)$ is the 1-indexed rank in stream $s$.

**Deterministic Tie-Breaking:**
When two items have identical fusion scores (within float tolerance $10^{-9}$):
1. Number of streams containing the item (descending — items found in more streams rank higher).
2. Average rank across streams where the item appeared (ascending — lower average rank is better).
3. String comparison on `item_id` (ascending lexicographical order, e.g. `"c1" < "c2"`).

**Requirements:**
1. **Core Function:**
   ```python
   def weighted_rrf(
       ranked_lists: dict[str, list[SearchHit]],
       weights: dict[str, float] | None = None,
       rrf_k: int = 60,
       top_k: int = 200,
   ) -> list[FusedHit]: ...
   ```
2. **Class Wrapper:**
   ```python
   class WeightedRRF:
       def __init__(self, weights: dict[str, float] | None = None, rrf_k: int = 60): ...
       def fuse(self, ranked_lists: dict[str, list[SearchHit]], top_k: int = 200) -> list[FusedHit]: ...
       def fuse_stream_results(self, stream_results: StreamResults, top_k_docs: int = 200, top_k_chunks: int = 200) -> tuple[list[FusedHit], list[FusedHit]]: ...
   ```
3. **Helper Function:**
   ```python
   def fuse_stream_results(
       stream_results: StreamResults,
       weights: dict[str, float] | None = None,
       rrf_k: int = 60,
       top_k_docs: int = 200,
       top_k_chunks: int = 200,
   ) -> tuple[list[FusedHit], list[FusedHit]]: ...
   ```
4. **Metadata Preservation:**
   - In `FusedHit`:
     - `item_id`: ID of the document or chunk.
     - `item_type`: `"document"` or `"chunk"`.
     - `doc_id`: parent document ID.
     - `language`: language of the item (take from the highest ranked hit for that item).
     - `fusion_score`: accumulated weighted RRF score.
     - `stream_ranks`: dict mapping `stream_name -> rank` (1-indexed).
5. **Edge Cases:**
   - Empty input dictionaries -> returns empty list.
   - Streams with empty hit lists -> safely ignored.
   - All zero weights -> handles gracefully without division by zero.
   - Negative ranks in corrupt hits -> clamped or skipped.
   - `top_k <= 0` -> returns empty list.
6. **Package Exports:**
   - Export `weighted_rrf`, `WeightedRRF`, and `fuse_stream_results` in `src/fusion/__init__.py`.

**TDD Steps:**
- Step 1: Write failing tests in `tests/test_weighted_rrf.py` verifying RRF calculations, weights, tie-breaking, metadata preservation, empty streams, and `fuse_stream_results`.
- Step 2: Run `pytest tests/test_weighted_rrf.py -v` (Verify RED).
- Step 3: Implement `src/fusion/weighted_rrf.py` and `src/fusion/__init__.py`.
- Step 4: Run `pytest tests/test_weighted_rrf.py -v` (Verify GREEN).
- Step 5: Commit changes.
