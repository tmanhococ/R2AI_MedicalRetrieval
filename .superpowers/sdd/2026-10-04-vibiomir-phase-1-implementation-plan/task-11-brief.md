### Task 11: Pilot, freeze baseline A và bàn giao submission

**Files:** Implementation run artifacts; docs/vibiomir/phase1_runbook.md; báo cáo baseline_A.md.

**Interfaces:** Consumes toàn bộ outputs Tasks 1–10; produces submission_A.zip, manifests/checksums, validation report và baseline report.

- [ ] Chạy pilot 500 URLs stratified theo domain và URL type, seed42; sample IDs được lưu. Không chỉ lấy 500 rows đầu.
- [ ] Review mẫu 30 nguồn và 20 exported chunks; report lỗi QA extraction, soft-error, language, source spans và token overflow.
- [ ] Freeze snapshot pilot; chạy đủ 1.200 queries để đo total runtime, peak RAM/GPU và lỗi inference. Chưa submit pilot nếu data quá thiếu; xác định snapshot baseline sử dụng corpus sẵn có + phần đã crawl được.
- [ ] Tăng crawl theo batches/snapshots khi cần; benchmark coverage mỗi lần. Không đặt cam kết crawl đủ 4,39M trong vài ngày khi chưa có throughput đo được.
- [ ] Freeze corpus/index/config/model/cache fingerprints cho baseline A; chạy inference resume-safe cho đủ query set.
- [ ] Chạy toàn bộ checks ở mục 14; inspect ZIP listing và validation report, không chỉ file extension.
- [ ] Viết baseline_A.md: corpus coverage, metrics dev + profile status, cutoff K, failure counts, costs, known limitations, ZIP SHA256.
- [ ] Bàn giao file ZIP và report. Việc upload Dashboard là thao tác riêng sau khi người dùng chọn bài nộp; Phase 1 không tự tiêu lượt submit.
- [ ] Khi có điểm Dashboard, lưu 6 metrics cùng submission ID/phase và UTC timestamp; không suy per-query labels từ aggregate metrics.

## 13. Runbook commands dự kiến

Các commands sau là interface sẽ implement trong Task 9, chưa phải xác nhận có sẵn trong repo. Paths tương đối với repo root; --snapshot nhận snapshot ID/path thật sau build, không để literal placeholder khi chạy.

~~~bash
python -m src.cli.ingest_vibiomir --config configs/vibiomir_phase1.yaml
python -m src.cli.crawl --config configs/vibiomir_phase1.yaml --pilot-size 500 --seed 42
python -m src.cli.crawl --config configs/vibiomir_phase1.yaml --resume --limit 10000
python -m src.cli.extract --config configs/vibiomir_phase1.yaml --resume
python -m src.cli.build_snapshot --config configs/vibiomir_phase1.yaml --name phase1_A
python -m src.cli.index --config configs/vibiomir_phase1.yaml --snapshot artifacts/vibiomir/snapshots/phase1_A
python -m src.cli.evaluate --config configs/vibiomir_phase1.yaml --split dev --snapshot artifacts/vibiomir/snapshots/phase1_A
python -m src.cli.predict --config configs/vibiomir_phase1.yaml --split test --snapshot artifacts/vibiomir/snapshots/phase1_A --resume --export-submission
python -m src.cli.validate_submission --config configs/vibiomir_phase1.yaml --snapshot artifacts/vibiomir/snapshots/phase1_A --zip artifacts/experiments/vibiomir_phase1_A/submission_A.zip
~~~

Không chạy evaluate --split dev khi chưa tạo dev qrels. Prediction/export không phụ thuộc việc đã có dev.

Tests tối thiểu:

~~~bash
pytest tests/data/test_vibiomir_inputs.py tests/data/test_snapshot_adapter.py -q
pytest tests/ingestion/test_fetcher.py tests/ingestion/test_crawl_resume.py tests/ingestion/test_extraction.py -q
pytest tests/indexing/test_source_spans.py tests/reranking/test_pair_budget.py -q
pytest tests/evaluation/test_vibiomir_text.py tests/evaluation/test_vibiomir_metrics.py tests/evaluation/test_dev_provenance.py tests/evaluation/test_error_breakdown.py -q
pytest tests/submission/test_vibiomir_export.py tests/integration/test_vibiomir_phase1.py -q
~~~

Sau các tests này, chạy regression command của repo ghi tại Task 0. Không báo tests pass chỉ dựa trên danh sách test trong kế hoạch.

## 14. Acceptance gates trước submit A

| Gate | Bằng chứng cần có | Điều kiện pass |
|---|---|---|
| Input identity | Input manifest/registry | Đủ đúng 1.200 query; IDs hợp lệ; corpus row count đo từ file thật |
| Crawl resume | Tests + pilot report | Không nhân đôi jobs/IDs; lỗi được ghi trạng thái |
| Extraction | Fixtures + human sample | QA answer/body được giữ; error pages quarantine; chưa hỗ trợ được báo |
| Source provenance | SourceStore + validation report | 100% submitted chunks là exact spans đúng doc/version |
| Token budget | Pair tests + overflow report | Reranker body không bị cắt âm thầm; dense query overflow được khai báo |
| Snapshot consistency | Index/run manifests | Mọi snapshot/model/tokenizer/cache fingerprints đúng |
| Evaluator logic | Hand-calculated tests | LCS exact, same doc, row/column counts và macro đúng |
| Scorer parity | Profile status | Nếu chưa có BTC tokenizer, ghi false; không claim exact parity |
| Prediction coverage | Query checkpoints | 1.200 completed query, 0 inference failures chưa xử lý |
| Export | Strict validation + ZIP listing | Một root results.json, IDs integer, đủ fields, không extra files |
| Reproducibility | Two export hashes, run manifest | Deterministic export cùng runtime và predictions |
| Regression | Actual test command/output | Không regression chưa giải thích thuộc pipeline hiện có |

Scorer parity chưa verified không tự chặn submission hợp lệ; nó chặn việc gọi điểm nội bộ là điểm BTC chính xác. Dev annotation chưa hoàn tất cũng không chặn export baseline, nhưng không được tuyên bố cutoff đã calibrated.

Không đặt threshold score như Docs F2>=0,55 làm gate hoàn thành Phase 1. Mục tiêu Phase 1 là baseline đáng tin; quality score được đo để quyết định Phase 2.

## 15. Báo cáo cần giữ cho mỗi run

| Artifact | Nội dung |
|---|---|
| run_manifest.json | Seed, config hash, models/revisions, runtime, snapshot/index/cache/profile hashes |
| coverage.json | Official registry size, unique URLs, fetched/ready/quarantined/failures theo host/language |
| retrieval_debug.jsonl | Original/translated query, candidates, raw/fusion/rerank/joint scores, cutoff, windows/overflow |
| predictions.jsonl | Internal FinalPrediction, đủ query IDs; không phải file nộp BTC |
| metrics.json | Dev metrics thực nếu có labels, profile status; không gán số giả khi thiếu labels |
| per_query_metrics.csv | P/R/F2 và correct_prediction/found_reference counts |
| error_analysis.csv | Vị trí mất document/chunk và evidence recall có điều kiện parent doc |
| validation_report.json | Schema/coverage/provenance/ZIP checks và lỗi cụ thể |
| results.json | Official submission JSON array |
| submission_A.zip | Một file results.json ở root |
| baseline_A.md | Phân tích baseline, chi phí, limitations và checkpoint Dashboard sau khi nộp |

Coverage denominator phải ghi rõ là official rows hay unique URLs. “Fetched URL coverage” không phải “retrieval recall”. Recall chỉ tính khi có reference labels.

## 16. Trình tự triển khai và phần có thể trì hoãn

Đường chạy tối thiểu để có ZIP: Task 0 → 1 → 2 → 3 → 4 → 5 → 8 → 9 → 11.

Evaluator và diagnostics: Task 6 → 7 → 10, hoàn tất trước khi kết luận baseline đạt hiệu quả hoặc dùng dev để chọn cutoff.

Task 6/7 không phụ thuộc crawl toàn corpus; có thể thực hiện khi chờ pilot network bằng fixtures. Đây là thứ tự công việc, không yêu cầu tự tạo subagents.

| Mức ưu tiên | Hạng mục |
|---|---|
| Bắt buộc trước ZIP | Identity, source spans, snapshot, integration, exporter, query coverage |
| Bắt buộc trước kết luận về metric | LCS/macro evaluator, profile disclosures, reliable dev labels |
| Có thể mở rộng sau baseline | Nhiều domain parsers, OCR/JS sources, larger crawl coverage |
| Ngoài Phase 1 | Model upgrade, learned cutoff, multi-query, document-conditioned evidence selector |

Không ước lượng ngày hoàn thành cố định trước Task 0 và pilot. Sau pilot, ước lượng crawl từ throughput từng host, tỷ lệ retry và bytes/source; ước lượng inference từ pairs/query và hardware thật.

## 17. Checklist tự rà soát thiết kế

- [ ] Repo paths đã được đối chiếu hoặc giữ trạng thái proposed rõ ràng.
- [ ] Không có task nào yêu cầu sử dụng test labels để fit.
- [ ] Internal strings và official integer IDs có mapping xác định.
- [ ] URL/content dedup không làm mất official document IDs.
- [ ] source_text, index_text và evaluation normalization được tách riêng.
- [ ] LCS giữ thứ tự token; threshold chia cho reference length.
- [ ] Chunk P/R có hai counts độc lập; không ép matching một-một.
- [ ] Tokenizer/empty-label/FINAL SCORE chưa rõ được đánh dấu, không hard-code thành quy định BTC.
- [ ] Snapshot/index/model/profile changes invalidate artifacts cần thiết.
- [ ] Các functions/types/commands trong tasks nhất quán với contracts.
- [ ] 1.200 queries được validate theo exact IDs, không chỉ số lượng.
- [ ] Submission không có chunk được sinh mới hoặc cắt từ nguồn khác.
- [ ] Tests là kế hoạch thực thi; chưa bị mô tả như đã chạy và pass.

Self-review khi tạo tài liệu: đã đối chiếu scope với spec gốc, kiểm tra tên contracts/tasks, phân biệt dense sharding với sparse statistics, tách internal provenance validation khỏi JSON validation, và giữ các quy định BTC chưa xác nhận dưới dạng profile tạm thời. Checklist vận hành ở trên vẫn để trống vì repo thật và pipeline chưa được chạy trong workspace này.

## 18. Nguồn kỹ thuật và cách dùng

Nguồn gốc yêu cầu: spec(2).md và quy định BTC người dùng cung cấp; số liệu 4,39M/1.200 và Dashboard là snapshot ảnh ngày 2026-10-04, không phải leaderboard live đã xác minh.

Tài liệu primary đã đối chiếu ngày 2026-10-04:

- Dataset chính thức: https://huggingface.co/datasets/AIGuruTinix/ViBioMIR — khi triển khai, tải đúng release/checksum; không xem tên split train là labels.
- HTTPX timeouts: https://www.python-httpx.org/advanced/timeouts/ — read timeout là thời gian chờ chunk, cần total deadline riêng.
- Arrow Parquet streaming: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html — dùng iter_batches thay vì materialize toàn file.
- Trafilatura structured extraction: https://trafilatura.readthedocs.io/en/latest/corefunctions.html — kiểm tra API theo version pin tại repo.
- fastText language identification: https://fasttext.cc/docs/en/language-identification.html — model language ID, lưu checksum và tuân thủ điều kiện sử dụng của model.
- E5 model card: https://huggingface.co/intfloat/multilingual-e5-large — query/passage prefix, dimension 1024, max input 512.
- BGE reranker model card: https://huggingface.co/BAAI/bge-reranker-v2-m3 — input query/passage, score không mặc định là calibrated probability.
- FAISS index memory: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes — Flat float32 4*d bytes/vector; ANN có trade-offs riêng.
- Python HTML entity decoding: https://docs.python.org/3/library/html.html — html.unescape; evaluator vẫn phải giữ thứ tự normalize của BTC.

Khi triển khai, đọc API theo dependency version thật. Không copy install versions cũ trong ví dụ model card mà không kiểm tra compatibility của repo.
