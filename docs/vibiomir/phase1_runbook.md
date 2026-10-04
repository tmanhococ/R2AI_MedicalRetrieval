# ViBioMIR Phase 1 Runbook: Ingestion, Indexing, Prediction & Submission

Tài liệu hướng dẫn vận hành hoàn chỉnh cho hệ sinh thái ViBioMIR Phase 1 Retrieval. Toàn bộ các bước từ nhập liệu Parquet chính thức, thu thập dữ liệu có kiểm soát, trích xuất cấu trúc văn bản, đóng băng snapshot bất biến, lập chỉ mục BM25/Dense, dự đoán suy luận theo checkpoint giao dịch, và xuất gói nộp bài ZIP hợp lệ.

---

## 1. Yêu cầu Môi trường & Thiết lập

- **Python**: `>=3.10` (khuyến nghị Python 3.12).
- **Trình quản lý gói**: `uv` (hoặc `pip` trong virtualenv).
- **Thư viện chính**: `pyarrow`, `sqlite3`, `httpx`, `trafilatura`, `pymupdf`, `bm25s`, `rank-bm25`, `faiss-cpu`, `pytest`.
- **Dữ liệu chính thức**:
  - `data/official/query.parquet`: 1.200 câu hỏi đánh giá chính thức (`id` int64, `query` string).
  - `data/official/links_corpus.parquet`: 4.394.718 URL nguồn y tế (`id` int64, `url` string).

---

## 2. Quy trình Vận hành Chuẩn (CLI Runbook)

Tất cả các lệnh được thực thi từ thư mục gốc của repository (`d:/Research/R2AI/mvp-rag-medical`).

### Bước 1: Nhập Nguồn Dữ liệu vào Registry SQLite
Nhập toàn bộ danh mục 4.39M URL và 1.200 câu hỏi vào cơ sở dữ liệu SQLite định danh URL duy nhất:
```bash
uv run python -m src.cli.ingest_vibiomir --config configs/vibiomir_phase1.yaml
```
- **Input**: `data/official/links_corpus.parquet`, `data/official/query.parquet`
- **Output**: `artifacts/vibiomir/registry.sqlite` (`sources` và `fetch_jobs`)
- **Tùy chọn CLI**:
  - `--batch-size`: Số lượng hàng nạp mỗi batch giao dịch (mặc định: `20000`).
  - `--expected-query-count`: Số câu hỏi kiểm chứng bắt buộc (mặc định: `1200`).

---

### Bước 2: Thu thập Dữ liệu (Resumable Crawler)
Chạy crawler đa luồng có quản lý domain rate-limit, tuân thủ `robots.txt`, và checkpoint resume an toàn:
```bash
# Chạy pilot có kiểm soát 500 URLs
uv run python -m src.cli.crawl --config configs/vibiomir_phase1.yaml --pilot-size 500 --seed 42

# Tiếp tục tải theo lô tăng dần (resume từ trạng thái hiện tại)
uv run python -m src.cli.crawl --config configs/vibiomir_phase1.yaml --resume --limit 10000
```
- **Output**: 
  - `artifacts/vibiomir/raw/{hash_prefix}/{hash}.gz`: Dữ liệu nén thô HTML/PDF.
  - `artifacts/vibiomir/raw/coverage.json`: Báo cáo độ phủ theo host và trạng thái HTTP.
- **Tùy chọn CLI**:
  - `--concurrency`: Số workers đồng thời (mặc định: `16`).
  - `--per-domain-concurrency`: Giới hạn kết nối tối đa trên từng tên miền (mặc định: `2`).
  - `--min-domain-interval`: Thời gian chờ tối thiểu giữa 2 request cùng tên miền (mặc định: `1.0s`).

---

### Bước 3: Trích xuất Cấu trúc Văn bản Offline
Trích xuất tiêu đề, nội dung phân đoạn (`blocks`), xác thực ngôn ngữ và lọc rác (soft-404, quảng cáo):
```bash
uv run python -m src.cli.extract --config configs/vibiomir_phase1.yaml --resume
```
- **Backend trích xuất**: Trafilatura (HTML) và PyMuPDF (PDF).
- **Bộ lọc chất lượng**: Ngôn ngữ (`vi`, `en`, `zh`), độ dài tối thiểu $\ge 50$ ký tự, chặn trang lỗi.

---

### Bước 4: Đóng băng Snapshot Bất biến (Immutable Snapshot)
Tạo snapshot phân vùng Parquet sharded (`documents_*.parquet`, `chunks_*.parquet`) và chỉ mục tra cứu `lookup.sqlite`:
```bash
uv run python -m src.cli.build_snapshot --config configs/vibiomir_phase1.yaml --name phase1_A
```
- **Output**: `artifacts/vibiomir/snapshots/phase1_A/`
  - `manifest.json`: Chứa `source_membership_sha256`, `chunker_fingerprint`, số tài liệu ready, số chunks.
  - `documents_00000.parquet`: Tài liệu đã làm sạch kèm `source_sha256`.
  - `chunks_00000.parquet`: Các chunk đoạn văn bản kèm span `[start_char, end_char]` chính xác (`source_text[start:end] == chunk.text`).
  - `lookup.sqlite`: Chỉ mục B-tree tra cứu nhanh tài liệu theo `doc_id` và chunk theo `chunk_id`.

---

### Bước 5: Lập Chỉ mục Tìm kiếm (BM25 & Dense Indexing)
Xây dựng chỉ mục BM25S (syllable tokenizer tiếng Việt) và Dense vector embeddings:
```bash
uv run python -m src.cli.index --config configs/vibiomir_phase1.yaml --snapshot artifacts/vibiomir/snapshots/phase1_A --output-dir artifacts/vibiomir/indexes/phase1_A
```
- **Output**: `artifacts/vibiomir/indexes/phase1_A/`
  - `bm25/`: Chỉ mục sparse BM25S tài liệu và chunk.
  - `dense/`: Chỉ mục FAISS Flat L2/IP vector.
- **Đặc điểm**: Chỉ mục được xây dựng một lần duy nhất cho toàn bộ snapshot; không bao giờ rebuild chỉ mục theo từng query khi benchmark.

---

### Bước 6: Đánh giá Nội bộ (Chỉ chạy trên Split Dev)
> [!IMPORTANT]
> **Nguyên tắc chống rò rỉ (Anti-leakage Principle)**:
> Tuyệt đối KHÔNG chạy `--split test` với `evaluate`. Phân tách dev độc lập với 1.200 query chính thức (namespace `query_id >= 9001`). Lệnh `evaluate` sẽ lập tức báo lỗi `ValueError` nếu truyền split test không có nhãn ground truth.

```bash
uv run python -m src.cli.evaluate --config configs/vibiomir_phase1.yaml --split dev --snapshot artifacts/vibiomir/snapshots/phase1_A --dev-records-path data/dev/dev_qrels.jsonl --index-dir artifacts/vibiomir/indexes/phase1_A
```
- **Output**:
  - `metrics.json`: Macro Precision, Recall, $F_2$ cho tài liệu và chunk, inferred final mean score.
  - `per_query_metrics.csv`: Chi tiết P, R, $F_2$ của từng câu hỏi dev.

---

### Bước 7: Dự đoán và Xuất Gói Nộp bài (Predict & Export Submission)
Thực hiện truy xuất 1.200 câu hỏi chính thức với cơ chế checkpoint từng câu hỏi, cắt ngưỡng fixed top-K (`5` documents, `10` chunks) và xuất gói ZIP nộp bài chuẩn BTC:
```bash
uv run python -m src.cli.predict --config configs/vibiomir_phase1.yaml --split test --snapshot artifacts/vibiomir/snapshots/phase1_A --index-dir artifacts/vibiomir/indexes/phase1_A --output-dir artifacts/experiments/vibiomir_phase1_A --resume --export-submission
```
- **Output**:
  - `artifacts/experiments/vibiomir_phase1_A/predictions_checkpoint.jsonl`: Checkpoint từng câu hỏi kèm `run_fingerprint`.
  - `artifacts/experiments/vibiomir_phase1_A/retrieval_debug.jsonl`: Trace chi tiết điểm, variants, candidates, cutoff của từng câu hỏi.
  - `artifacts/experiments/vibiomir_phase1_A/predictions.jsonl`: Kết quả dự đoán nội bộ đầy đủ.
  - `artifacts/experiments/vibiomir_phase1_A/run_manifest.json`: Manifest phiên chạy (hash cấu hình, model, thời gian chạy).
  - `artifacts/experiments/vibiomir_phase1_A/results.json`: File JSON kết quả định dạng BTC.
  - `artifacts/experiments/vibiomir_phase1_A/submission_A.zip`: File ZIP đóng gói duy nhất một file `results.json` ở root.
  - `artifacts/experiments/vibiomir_phase1_A/validation_report.json`: Báo cáo nghiệm thu tự động trước khi đóng gói.

---

### Bước 8: Kiểm định Nghiêm ngặt Gói Nộp bài (Offline Validation)
Kiểm tra cấu trúc và tính hợp lệ của file ZIP nộp bài độc lập trước khi nộp lên hệ thống BTC:
```bash
uv run python -m src.cli.validate_submission --config configs/vibiomir_phase1.yaml --snapshot artifacts/vibiomir/snapshots/phase1_A --zip artifacts/experiments/vibiomir_phase1_A/submission_A.zip
```
- **Tiêu chí kiểm định**:
  1. File ZIP hợp lệ, chỉ chứa duy nhất 1 entry `results.json` ở root.
  2. Đầy đủ đúng 1.200 câu hỏi theo ID từ `data/official/query.parquet`.
  3. Định dạng JSON tuân thủ schema: `id` (int), `relevant_docs` (list[int]), `relevant_chunks` (list[dict]).
  4. 100% chunk text xuất hiện trong bài nộp phải là exact substring của tài liệu tương ứng trong frozen snapshot (`doc.source_text`).

---

## 3. Cơ chế Khôi phục Checkpoint (Kill & Resume)

1. **Transaction atomicity**: Mỗi câu hỏi hoàn thành được ghi một dòng `status: "completed"` hoặc `status: "failed"` vào `predictions_checkpoint.jsonl`.
2. **Fingerprint guard**: Khi chạy với cờ `--resume`, `InferenceCheckpointManager` tính toán mã băm `run_fingerprint` (kết hợp query IDs, snapshot hash, index manifest, model revisions, pipeline config). Nếu cấu hình hoặc snapshot thay đổi, hệ thống từ chối resume với ngoại lệ `ValueError`.
3. **Clean restart**: Khi chạy không có cờ `--resume`, file checkpoint cũ tự động được dọn dẹp sạch để đảm bảo không bị bỏ qua câu hỏi nào.
4. **Error blocking**: Bất kỳ câu hỏi nào gặp lỗi ngoại lệ trong suy luận đều được ghi nhận là `failed`. Hệ thống nghiêm cấm chuyển đổi âm thầm câu hỏi lỗi thành list rỗng và CHẶN hoàn toàn việc xuất gói nộp bài ZIP cho đến khi lỗi được khắc phục.

---

## 4. Kiểm thử Hồi quy Toàn diện (Regression Testing)

Chạy bộ kiểm thử tự động toàn diện để bảo đảm không có bất kỳ regression nào:
```bash
uv run pytest -q
```
- Toàn bộ 354+ tests kiểm thử đơn vị, kiểm thử tích hợp, kiểm thử spans, token budgets, metrics, và exporter phải PASS 100%.
