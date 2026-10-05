### Task 11: Cross-Encoder Reranker with Chunk Aggregation

**Files:**
- Create: `src/reranking/__init__.py`
- Create: `src/reranking/reranker.py`
- Test: `tests/test_reranker.py`

**Interfaces:**
- Consumes: `QueryVariants` (from `src.query.query_processor`), `FusedHit` (from `src.data.schemas`), `ScoredHit` (from `src.data.schemas`)
- Produces: `Reranker(Protocol)`, `CrossEncoderReranker`, `DummyReranker`, `min_max_normalize`

**Requirements & Specifications:**
1. **Language-Aware Cross-Encoder Pairing:**
   When scoring pairs `(query, candidate_text)`:
   - If candidate `language == "vi"`: use `query_variants.vi`.
   - If candidate `language == "en"`: use `query_variants.en or query_variants.vi`.
   - If candidate `language == "zh"`: use `query_variants.zh or query_variants.vi`.
   - Any missing translation safely falls back to `query_variants.vi`.

2. **Chunk Reranking (`rerank_chunks`):**
   - Inputs: `query_variants: QueryVariants`, `candidates: list[FusedHit]`, `chunk_texts: dict[str, str]`, `top_k: int = 100`.
   - Filters out candidates with missing chunk text (or treats missing text as empty string).
   - Computes cross-encoder logits/scores for each candidate chunk.
   - Calculates `normalized_score` in $[0, 1]$ via min-max normalization across the scored chunks for the current query:
     $$\text{norm\_score} = \frac{\text{score} - \min}{\max - \min + 10^{-9}}$$
     (If $\max == \min$, defaults to 1.0 or 0.5).
   - Preserves `raw_fusion_score = candidate.fusion_score`.
   - Returns `list[ScoredHit]` sorted descending by `rerank_score`, limited to `top_k`.

3. **Document Reranking via Chunk Aggregation (`rerank_documents_via_chunk_aggregation`):**
   - Inputs: `doc_candidates: list[FusedHit]`, `scored_chunks: list[ScoredHit]`, `top_k: int = 100`.
   - Groups `scored_chunks` by `doc_id`.
   - For each document candidate `d`:
     $$\text{base\_doc\_score}(d) = \max_{c \in d}(\text{rerank\_chunk\_score}(c))$$
   - If document $d$ has no matching chunks in `scored_chunks`, falls back to a min-floor score or its `raw_fusion_score` scaled to the minimum chunk score, ensuring it is not completely dropped.
   - Calculates `normalized_score` in $[0, 1]$ via min-max normalization across all scored documents.
   - Preserves `raw_fusion_score = doc_candidate.fusion_score`.
   - Returns `list[ScoredHit]` sorted descending by `rerank_score`, limited to `top_k`.

4. **Direct Document Truncation Strategy (`rerank_documents_direct`):**
   - Inputs: `query_variants: QueryVariants`, `candidates: list[FusedHit]`, `doc_texts: dict[str, str]`, `max_length: int = 512`, `top_k: int = 100`.
   - Truncates doc text and directly scores `(query, truncated_doc_text)`.

5. **CrossEncoderReranker Implementation:**
   - Supports:
     - Pre-instantiated `CrossEncoder` or mock object with `.predict(sentences)`
     - Callable `(pairs: list[tuple[str, str]]) -> list[float]`
     - Model name string (e.g. `"BAAI/bge-reranker-v2-m3"`), with lazy `sentence_transformers.CrossEncoder` instantiation and CPU/CUDA device selection.
   - Batch size and max length configuration.

6. **DummyReranker Implementation:**
   - Deterministic mock for unit testing and CPU-only environments without PyTorch.

7. **Package Exports:**
   - Export `CrossEncoderReranker`, `DummyReranker`, `Reranker`, and `min_max_normalize` in `src/reranking/__init__.py`.

**TDD Steps:**
- Step 1: Write failing tests in `tests/test_reranker.py` covering chunk reranking, chunk-to-doc max pooling aggregation, language routing, min-max normalization, direct document truncation, and dummy reranker.
- Step 2: Run `pytest tests/test_reranker.py -v` (Verify RED).
- Step 3: Implement `src/reranking/reranker.py` and update `src/reranking/__init__.py`.
- Step 4: Run `pytest tests/test_reranker.py -v` (Verify GREEN).
- Step 5: Commit changes.
