### Task 7: Many-to-many chunk counts, macro metrics và dev compatibility

**Files:** Create evaluation/vibiomir_metrics.py, dev_records.py; modify metrics/calibrator integration; tests/evaluation/test_vibiomir_metrics.py, test_dev_provenance.py.

**Interfaces:**

- evaluate_query(query_id: int, prediction: SubmissionRow, gold: EvaluationRecord, profile: EvaluationProfile) -> QueryMetrics
- evaluate_dataset(predictions: Iterable[SubmissionRow], gold: Iterable[EvaluationRecord], query_ids: Sequence[int], profile: EvaluationProfile) -> EvaluationSummary
- EvaluationSummary: profile_fingerprint, scorer_parity_verified, query_count, macro_documents, macro_chunks, per_query_metrics, inferred_final_mean.
- load_dev_records(path: Path, official_query_ids: set[int], store: SourceStore) -> tuple[EvaluationRecord, ...]

- [ ] Test one_pred_two_refs: prediction "a b c d", references "a b" và "c d", cùng doc; chunk P=1/R=1. Test two_preds_same_ref: predictions "a b","a b", refs "a b","c d"; P=1/R=0.5, found_reference_count=1. Evaluator test trực tiếp giữ duplicates để xác minh semantics, dù exporter baseline từ chối duplicate predictions.
- [ ] Test same text wrong doc => không match; no predictions =>0 với nonempty gold; macro F2 khác F2(macro P, macro R); missing query không bị drop.
- [ ] Test empty/empty F2=1 profile tạm thời, =0 profile thay thế; labels có empty chunk text reject. Dev query namespace không dùng ID test chính thức.
- [ ] Run: pytest tests/evaluation/test_vibiomir_metrics.py tests/evaluation/test_dev_provenance.py -q. Expected: FAIL new matching/data contract.
- [ ] Implement EvaluationProfile, EvaluationSummary, DevQueryMetadata, row/column counts và per-query trace matched references; không one-to-one matching. Keep legacy chunk-ID evaluator riêng cho synthetic tests, không áp dụng vào BTC. load_dev_records giữ provenance metadata trên EvaluationRecord để calibrator không mất thông tin origin/split.
- [ ] Modify calibrator: reject test/public/private qrels để fit; reject profile/data/model fingerprints mismatch; fit disabled mặc định. Evaluation có nhãn dev được phép nhưng profile chưa verified vẫn trả diagnostics với warning metadata.
- [ ] Run tests và existing metrics regression. Expected: PASS, all query IDs included.
- [ ] Commit: fix: evaluate chunks with independent prediction and reference counts.

