# Task 11 Report: Cross-Encoder Reranker with Chunk Aggregation

**Date:** 2026-09-22  
**Task:** Task 11 - Cross-Encoder Reranker with Chunk Aggregation  
**Status:** DONE  
**Branch / Commit:** `fe008bf` (`master`)

---

## 1. What Was Implemented

### 1.1 Core Components (`src/reranking/reranker.py` & `src/reranking/__init__.py`)
1. **Language-Aware Cross-Encoder Pairing (`resolve_query_for_language`):**
   - Candidate `language == "vi"` -> `query_variants.vi`.
   - Candidate `language == "en"` -> `query_variants.en or query_variants.vi`.
   - Candidate `language == "zh"` -> `query_variants.zh or query_variants.vi`.
   - Case-insensitive matching (`strip().lower()`) and safe fallback to `query_variants.vi` for any missing or unhandled translation.

2. **Min-Max Score Normalization (`min_max_normalize`):**
   - Normalizes arbitrary cross-encoder logit scores into $[0.0, 1.0]$.
   - Formula: $\text{norm\_score} = \frac{\text{score} - \min}{\max - \min + 10^{-9}}$.
   - Gracefully handles empty inputs (`[]`), single values, or uniform scores ($\max == \min$) returning `[default_val] * len(scores)`.

3. **Chunk Reranking (`rerank_chunks`):**
   - Inputs: `query_variants`, `candidates: list[FusedHit]`, `chunk_texts: dict[str, str]`, `top_k: int = 100`.
   - Filters out candidates with missing chunk text.
   - Computes cross-encoder logits for candidate chunks.
   - Normalizes scores via min-max normalization.
   - Preserves `raw_fusion_score = candidate.fusion_score`.
   - Returns `list[ScoredHit]` sorted descending by `rerank_score` with deterministic tie-breaking (fusion score descending, then item ID ascending), truncated to `top_k`.

4. **Document Reranking via Chunk Aggregation (`rerank_documents_via_chunk_aggregation`):**
   - Inputs: `doc_candidates: list[FusedHit]`, `scored_chunks: list[ScoredHit]`, `top_k: int = 100`.
   - Groups `scored_chunks` by `doc_id`.
   - Implements max-pooling aggregation: $\text{base\_doc\_score}(d) = \max_{c \in d}(\text{rerank\_chunk\_score}(c))$.
   - Documents without matching chunks fall back to a minimum floor score below `min_chunk_score` (or user-specified `fallback_floor`), preserving them from being dropped.
   - Supports synthesizing document candidates directly from scored chunks when `doc_candidates` is None.
   - Calculates document normalized scores in $[0, 1]$ and preserves `raw_fusion_score`.

5. **Direct Document Truncation Strategy (`rerank_documents_direct`):**
   - Truncates document text to `max_length` (default 512).
   - Directly scores `(query, truncated_doc_text)` and returns sorted `ScoredHit` list.

6. **CrossEncoderReranker (`CrossEncoderReranker`):**
   - Supports pre-instantiated model / mock objects with `.predict(sentences, ...)`.
   - Supports callables `(pairs: list[tuple[str, str]]) -> list[float]`.
   - Supports model name strings (e.g., `"BAAI/bge-reranker-v2-m3"`) with lazy `sentence_transformers.CrossEncoder` instantiation.
   - Configurable `device` ("auto", "cpu", "cuda"), `batch_size`, and `max_length`.

7. **DummyReranker (`DummyReranker`):**
   - Deterministic lexical token overlap reranker for unit testing and CPU-only / lightweight environments without PyTorch.
   - Supports custom scoring functions.

8. **Reranker Protocol & Abstract Base Class (`Reranker`, `BaseReranker`):**
   - `@runtime_checkable` protocol defining `score_pairs`, `rerank_chunks`, `rerank_documents_via_chunk_aggregation`, and `rerank_documents_direct`.
   - `BaseReranker` implements the shared workflows for chunk reranking, chunk aggregation, and direct document truncation.

9. **Package Exports (`src/reranking/__init__.py`):**
   - Exports `CrossEncoderReranker`, `DummyReranker`, `Reranker`, `min_max_normalize`, `rerank_chunks`, `rerank_documents_direct`, `rerank_documents_via_chunk_aggregation`, and `resolve_query_for_language`.

---

## 2. TDD Evidence

### Step 1: Initial RED Phase (Failing Test)
Ran `python -m pytest tests/test_reranker.py -v`:
```text
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\test_reranker.py'.
...
E   ModuleNotFoundError: No module named 'src.reranking'
=========================== short test summary info ===========================
ERROR tests/test_reranker.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
============================== 1 error in 0.18s ===============================
```

### Step 2: Implementation & GREEN Phase
After implementing `src/reranking/reranker.py` and `src/reranking/__init__.py`:
Ran `python -m pytest tests/test_reranker.py -v`:
```text
============================= test session starts =============================
collected 22 items

tests/test_reranker.py::test_package_exports PASSED                      [  4%]
tests/test_reranker.py::test_protocol_conformance PASSED                 [  9%]
tests/test_reranker.py::test_min_max_normalize_standard_range PASSED     [ 13%]
tests/test_reranker.py::test_min_max_normalize_empty_and_single PASSED   [ 18%]
tests/test_reranker.py::test_min_max_normalize_identical_scores PASSED   [ 22%]
tests/test_reranker.py::test_language_aware_pairing_routing PASSED       [ 27%]
tests/test_reranker.py::test_language_aware_pairing_fallback_when_translation_missing PASSED [ 31%]
tests/test_reranker.py::test_rerank_chunks_scoring_and_normalization PASSED [ 36%]
tests/test_reranker.py::test_rerank_chunks_filters_missing_chunk_texts PASSED [ 40%]
tests/test_reranker.py::test_rerank_documents_via_chunk_aggregation_max_pooling PASSED [ 45%]
tests/test_reranker.py::test_rerank_documents_aggregation_fallback_unmatched_doc PASSED [ 50%]
tests/test_reranker.py::test_rerank_documents_aggregation_empty_chunks PASSED [ 54%]
tests/test_reranker.py::test_rerank_documents_direct PASSED              [ 59%]
tests/test_reranker.py::test_dummy_reranker_lexical_overlap PASSED       [ 63%]
tests/test_reranker.py::test_dummy_reranker_custom_fn PASSED             [ 68%]
tests/test_reranker.py::test_cross_encoder_with_mock_predict PASSED      [ 72%]
tests/test_reranker.py::test_cross_encoder_lazy_instantiation PASSED     [ 77%]
tests/test_reranker.py::test_empty_inputs PASSED                         [ 81%]
tests/test_reranker.py::test_rerank_documents_synthesize_from_chunks PASSED [ 86%]
tests/test_reranker.py::test_standalone_rerank_functions PASSED          [ 90%]
tests/test_reranker.py::test_cross_encoder_auto_device_and_numpy_output PASSED [ 95%]
tests/test_reranker.py::test_candidate_keyed_by_doc_id_and_all_missing PASSED [100%]

============================= 22 passed in 0.17s ==============================
```

### Step 3: Full Suite Regression Verification
Ran `python -m pytest`:
```text
======================= 135 passed, 1 warning in 4.83s ========================
```
Statement coverage for `src/reranking`: **96%**.

---

## 3. Files Changed
- `src/reranking/__init__.py` (Created)
- `src/reranking/reranker.py` (Created)
- `tests/test_reranker.py` (Created)

---

## 4. Self-Review Findings
- **Windows C-Extension Safety:** Early in testing, dynamically loading `sentence_transformers` directly via `patch("sentence_transformers.CrossEncoder")` in test runs after FAISS caused a Windows OpenMP / torch DLL collision. This was addressed by mocking `sentence_transformers` via `patch.dict("sys.modules", ...)`, which executes in sub-millisecond time and isolates unit tests from heavy C-extension initialization while preserving runtime loading logic.
- **Protocol Conformance:** `@runtime_checkable` verified for `Reranker` on both `CrossEncoderReranker` and `DummyReranker`.
- **Tie-Breaking Determinism:** Explicit 3-tier tuple sorting `(-rerank_score, -raw_fusion_score, item_id)` ensures deterministic ranking across all platforms.
- **Preservation of Raw Fusion Score:** Verified that `raw_fusion_score` from `FusedHit.fusion_score` is properly propagated into `ScoredHit.raw_fusion_score`.

---

## 5. Concerns / Notes
- None. The reranking module is self-contained, adheres to all interface contracts, passes all unit tests, and integrates with the existing retrieval and fusion pipelines.
