### Task 9: CLI end-to-end, inference checkpoint và reports

**Files:** Create CLI commands mục 11; wire index/evaluate/predict; tests/integration/test_vibiomir_phase1.py; config schema tests.

**Interfaces:** CLI commands mục 13; prediction checkpoint key = run_fingerprint + query_id, run_fingerprint includes query/corpus/index/model/config hashes.

- [ ] Viết fixture 3 query IDs 1/4/9, hai URL aliases và một fetch failure; fake model/retriever để unit E2E không cần GPU/network. Pipeline phải ra đủ 3 rows và correct source chunks.
- [ ] Test kill/resume query checkpoint; query inference failure phải ghi failed và block final export, không âm thầm thay bằng lists rỗng. Resume khác run fingerprint bị từ chối.
- [ ] Test unknown YAML key reject, config merge precedence CLI > YAML > defaults, missing language stream không crash, no-candidate query trả empty rõ ràng.
- [ ] Run: pytest tests/integration/test_vibiomir_phase1.py -q. Expected: FAIL integration.
- [ ] Implement từng CLI nối functions ở task trước; no duplicate retrieval pipeline. Predict lưu checkpoint mỗi query hoàn tất; resume chỉ skip completed query cùng fingerprint.
- [ ] Ghi coverage.json, retrieval_debug.jsonl, per_query_metrics khi có dev labels, run_manifest.json, validation_report.json. Test set không labels: không tạo metrics “0” giả.
- [ ] Run integration + existing regression suite; chạy smoke model thật trên một pilot nhỏ được đánh dấu riêng. Expected: PASS, đủ outputs và ZIP.
- [ ] Commit: feat: wire Phase 1 ingestion prediction and submission workflow.

