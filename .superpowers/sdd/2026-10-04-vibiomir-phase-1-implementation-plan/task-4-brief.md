### Task 4: Provenance source spans và model token budgets

**Files:** Modify indexing/chunker.py, query/query_processor.py, reranking/reranker.py; tests/indexing/test_source_spans.py, tests/reranking/test_pair_budget.py.

**Interfaces:**

- chunk_source_document(document: ExtractedDocument, dense_tokenizer: PreTrainedTokenizerFast, reranker_tokenizer: PreTrainedTokenizerFast, config: ChunkConfig) -> Iterator[SpanChunk]
- pack_reranker_inputs(query: str, chunk: SpanChunk, tokenizer: PreTrainedTokenizerFast, config: PairPackingConfig) -> list[PackedPair]
- PackedPair: query_text, passage_text, input_ids, attention_mask, query_window_index; final score max qua pairs.

- [ ] Viết tests exact-span identity với combining Unicode, lặp paragraph, ký hiệu và long paragraph; span union không bỏ content blocks; same input cho same IDs; title bổ sung chỉ xuất hiện trong index_text.
- [ ] Viết test long query with relevant condition in middle: query windows phải bao phủ condition; query+header+body+special tokens <=512; chunk body không bị truncation. Test tokenizers thật ở marked integration, fake offset tokenizer ở unit.
- [ ] Run: pytest tests/indexing/test_source_spans.py tests/reranking/test_pair_budget.py -q. Expected: FAIL spans hoặc pair budgeting trước sửa.
- [ ] Implement paragraph packing, dual-tokenizer body limit 320, offsets fallback long block, overlap64 và query windows128/16. Preserve raw scores; no query rewriting.
- [ ] Legacy chunks không có provenance chỉ migrate unique exact occurrence; otherwise regenerate. Log dense overflow; original BM25 branch phải dùng full query.
- [ ] Run tests; marked model-tokenizer test không tải weights lớn. Expected: PASS, source slices exact, không silent truncation body.
- [ ] Commit: fix: preserve source spans and enforce pair token budgets.

