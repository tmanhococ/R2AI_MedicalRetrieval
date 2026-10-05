### Task 10: Dev annotation và error analysis có provenance

**Files:** Create docs/vibiomir/dev_annotation_guide.md; annotation JSONL tại data/dev/ không commit nếu lớn/nhạy cảm; modify evaluation/error_analysis.py; tests/evaluation/test_error_breakdown.py.

**Interfaces:**

- DevQueryMetadata: query_id, query_text, origin, source_doc_ids, topic, question_style, split, pool_version, judgment_coverage.
- build_error_breakdown(trace: RetrievalTrace, gold: EvaluationRecord, store: SourceStore, profile: EvaluationProfile) -> ErrorBreakdown
- ErrorBreakdown: missing_source_docs, missing_candidate_docs, lost_after_rerank_docs, lost_after_cutoff_docs, missing_chunk_refs, evidence_missing_given_doc.

- [ ] Tạo dev queries độc lập với 1.200 query chính thức; dùng namespace riêng, lưu origin và reviewer. Không coi query sinh tự động từ bài viết là validation đại diện hoàn hảo.
- [ ] Pilot 30 queries để ổn định annotation; sau đó mục tiêu 100–200 queries, chủ đề/ngôn ngữ/độ dài đa dạng. Split theo source/near-duplicate groups để giảm leakage giữa fit và holdout.
- [ ] Pool documents từ nhiều retriever; label judged positives/negatives/unjudged rõ ràng; reference chunks là spans được người kiểm tra chọn từ source. Không biến unjudged thành negative mặc định.
- [ ] Tests cho từng vị trí rơi mất document và reference chunk đã có parent doc. Run: pytest tests/evaluation/test_error_breakdown.py -q; expected FAIL trước implement.
- [ ] Implement reports. Pooled qrels có thể thiếu relevant docs: ghi judgment coverage; chưa có labels đáng tin thì giữ fixed K và chỉ dùng dev để debug.
- [ ] Run tests và review 10 cases bằng raw/source text. Expected: PASS và trace giải thích được từng lỗi.
- [ ] Commit guide/report code; chỉ lưu annotation data theo quy tắc repo.

