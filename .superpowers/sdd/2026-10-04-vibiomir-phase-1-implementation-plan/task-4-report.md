# Task 4 Implementation Report: Provenance Source Spans và Model Token Budgets

## Summary of Implementation
Implemented exact source span preservation, dual tokenizer body budgeting, contextual header separation, and cross-encoder pair packing with windowed token budgeting:
1. **Provenance Slicing (`src/indexing/chunker.py`)**:
   - Implemented `chunk_source_document` with strict invariant: `chunk.text == document.source_text[chunk.start_char:chunk.end_char]`.
   - Dual-tokenizer token budgeting (`body_max_tokens = 320`) across dense and reranker tokenizers ("choosing the safer split").
   - Long paragraph fallback splitting using token offset mapping with overlap (`long_block_overlap_tokens = 64`).
   - Contextual headers (max 32 tokens) prepended solely to `index_text`, strictly never polluting `text` (source slice).
   - Paragraph packing honoring section boundaries and retaining short blocks (`drop_short_blocks=False`).
   - Legacy chunk migration helper (`migrate_legacy_chunk`, `migrate_legacy_chunks`) supporting exact single match resolution vs. quarantine/regeneration.
2. **Cross-Encoder Pair Token Budgeting (`src/reranking/reranker.py`)**:
   - Implemented `pack_reranker_inputs(query, chunk, tokenizer, config)` producing `list[PackedPair]`.
   - Sliding query token windows (128 tokens, overlap 16) ensuring 100% query token coverage including middle conditions.
   - Pair budgeting: query window + contextual header + body + special tokens <= 512. Contextual header is shrunk first if exceeding budget; chunk body is never silently truncated (`body_truncation="reject"` raises `ValueError` if body exceeds available budget).
   - Max score aggregation across query windows in `BaseReranker.rerank_chunks`.
3. **Query Processor Updates (`src/query/query_processor.py`)**:
   - `ProcessedQuery` updated to maintain full original query branch for BM25 (`bm25_original_text`).
   - Warning logged for `dense_query_overflow` when query tokens > 512.

---

## TDD Evidence

### RED Phase
Command:
```powershell
uv run pytest tests/indexing/test_source_spans.py tests/reranking/test_pair_budget.py -q
```
Output:
```text
=================================== ERRORS ====================================
____________ ERROR collecting tests/indexing/test_source_spans.py _____________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\indexing\test_source_spans.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
tests\indexing\test_source_spans.py:14: in <module>
    from src.indexing.chunker import (
E   ImportError: cannot import name 'chunk_source_document' from 'src.indexing.chunker' (D:\Research\R2AI\mvp-rag-medical\src\indexing\chunker.py)
____________ ERROR collecting tests/reranking/test_pair_budget.py _____________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\reranking\test_pair_budget.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
tests\reranking\test_pair_budget.py:17: in <module>
    from src.reranking.reranker import (
E   ImportError: cannot import name 'PackedPair' from 'src.reranking.reranker' (D:\Research\R2AI\mvp-rag-medical\src\reranking\reranker.py)
=========================== short test summary info ===========================
ERROR tests/indexing/test_source_spans.py
ERROR tests/reranking/test_pair_budget.py
!!!!!!!!!!!!!!!!!!! Interrupted: 2 errors during collection !!!!!!!!!!!!!!!!!!!
1 warning, 2 errors in 2.78s
```

### GREEN Phase
Command:
```powershell
uv run pytest tests/indexing/test_source_spans.py tests/reranking/test_pair_budget.py -q
```
Output:
```text
.................                                                        [100%]
17 passed, 1 warning in 2.16s
```

---

## Full Regression Suite Result
Command:
```powershell
uv run pytest -q
```
Output:
```text
........................................................................ [ 28%]
........................................................................ [ 56%]
........................................................................ [ 84%]
.......................................                                  [100%]
255 passed, 1 warning in 23.01s
```
Zero regressions across the entire suite (238 baseline tests + 17 new tests = 255 passed).

---

## Files Created / Modified
- Created: `tests/indexing/test_source_spans.py`
- Created: `tests/reranking/test_pair_budget.py`
- Modified: `src/data/vibiomir_contracts.py` (added `PackedPair` and `MigrationResult`)
- Modified: `src/indexing/chunker.py` (implemented `chunk_source_document`, token counting, long paragraph splitting, legacy migration)
- Modified: `src/query/query_processor.py` (added `dense_query_overflow` logging and `bm25_original_text`)
- Modified: `src/reranking/reranker.py` (implemented `pack_reranker_inputs`, query window max score aggregation)
- Modified: `src/reranking/__init__.py` (re-exported `PackedPair`, `pack_reranker_inputs`, `PairPackingConfig`)
- Modified: `pyproject.toml` (registered `integration` pytest marker)

---

## Commit Details
- Commit: `9d4950e`
- Subject: `fix: preserve source spans and enforce pair token budgets (Task 4)`

---

## Self-Review Checklist
- [x] Invariant `chunk.text == document.source_text[chunk.start_char:chunk.end_char]` strictly verified across combining Unicode, medical entities, and long paragraphs.
- [x] Invariant `chunk.source_sha256 == document.source_sha256` and `chunk_id == f"{doc_id}:{source_sha256}:{chunk_index:06d}"` verified.
- [x] Content blocks covered without gaps or deletions.
- [x] Additional title/headings present ONLY in `index_text`, never in `text`.
- [x] Dual-tokenizer body limit (320) and query window packing (128 / 16 overlap) enforced.
- [x] Pair token budget: query + header + body + special tokens <= 512; header shrunk first; body never silently truncated.
- [x] Score aggregation across query windows uses max pooling.
- [x] Legacy chunk migration preserves unique exact matches and quarantines/regenerates non-unique matches.
- [x] `dense_query_overflow` logged for queries exceeding 512 tokens while BM25 branch maintains full query.
- [x] 0 regressions across full test suite (255 passed).
