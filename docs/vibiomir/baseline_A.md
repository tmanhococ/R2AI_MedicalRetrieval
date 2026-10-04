# Báo cáo Nghiệm thu Baseline A: ViBioMIR Phase 1 Retrieval

## 1. Tóm tắt Tổng quan (Executive Summary)

Baseline A thiết lập đường chạy chuẩn nghiệm thu đầu tiên kết nối toàn diện hệ thống retrieval của Road to AI với dữ liệu và thể lệ chính thức của cuộc thi **ViBioMIR 2026** (Track Medical Retrieval).

Toàn bộ quy trình từ kiểm tra nguồn dữ liệu Parquet, nạp SQLite registry, thu thập pilot, trích xuất cấu trúc, đóng băng snapshot phân vùng Parquet, lập chỉ mục sparse BM25S, suy luận dự đoán có transactional checkpoint và đóng gói nộp bài đã hoàn tất và vượt qua 100% các tiêu chí nghiệm thu nghiêm ngặt.

- **Mã định danh gói nộp bài**: `submission_A.zip`
- **Mã băm SHA-256 gói nộp bài**: `7042f3f7f85df6aad857082b248492128ba14f72257ecc2477495cdf58de281f`
- **Độ phủ câu hỏi chính thức**: **1.200 / 1.200 queries** ($100{,}0\%$).
- **Số tài liệu dự đoán**: **6.000 documents** (chính xác 5 tài liệu / câu hỏi).
- **Số chunk dự đoán**: **12.000 chunks** (chính xác 10 chunks / câu hỏi).
- **Tỷ lệ truy vết nguồn chính xác**: **100,0%** (12.000/12.000 chunks có exact text span khớp hoàn hảo văn bản gốc trong snapshot).
- **Số lỗi / cảnh báo nghiệm thu**: **0 errors, 0 warnings**.
- **Thời gian suy luận toàn bộ 1.200 câu hỏi**: **2.80 giây** trên CPU.

---

## 2. Dữ liệu & Snapshot Cố định (Frozen Snapshot)

### 2.1 Định danh Dữ liệu Đầu vào
- `data/official/query.parquet`: 1.200 câu hỏi chính thức (`id`: int64, `query`: string tiếng Việt lâm sàng). Mã băm SHA-256: `ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08`.
- `data/official/links_corpus.parquet`: 4.394.718 liên kết nguồn (`id`: int64, `url`: string). Mã băm SHA-256: `dee1153f5b9d9010dea18e1b7724b4915ff847b711d036bec8b933486802a971`.

### 2.2 Thuộc tính Snapshot `phase1_A`
- **Snapshot ID**: `phase1_A`
- **Source Membership Hash**: `9a0823c8a2b2a0c5744ba9fd47990e1306d3d195959ba11765aa73e4c86151a8`
- **Extractor Fingerprint**: `18de584ab468e8f624b1b1dd748a4fba6348ff18b1de7e191b3a599a4b73a971`
- **Chunker Fingerprint**: `89bc0dcae1261557022449bd6f8098e3037f4e43a5474c0132f00205c6d69c50`
- **Số lượng tài liệu sẵn sàng**: 85 tài liệu đã trích xuất sạch (tiếng Việt).
- **Số lượng chunk đoạn văn**: 296 chunks (paragraph token spans, `body_max_tokens: 320`, `long_block_overlap: 64`).
- **Phân vùng Shards Parquet**:
  - `documents_00000.parquet`: 85 rows, 263.8 KB (SHA-256: `58b79062...`)
  - `chunks_00000.parquet`: 296 rows, 462.2 KB (SHA-256: `0440c58e...`)
  - `lookup.sqlite`: Chỉ mục B-tree tra cứu 4.394.718 official doc IDs và 296 chunk locations.

---

## 3. Kiến trúc Mô hình & Thiết lập Cắt ngưỡng (Baseline Architecture)

- **Sparse Retriever**: BM25S backend sử dụng syllable segmentation cho tiếng Việt.
- **Top-K Candidates**: 200 candidates per query stream.
- **Chiến lược Cắt ngưỡng (Cutoff Policy)**: `fixed_top_k` dự phòng an toàn theo mục 8 của kế hoạch:
  - Tài liệu: `documents = 5`
  - Đoạn văn: `chunks = 10`
- **Tính nhất quán cấu trúc (Structural Consistency)**: `independent` (giữ độc lập thứ hạng theo thể lệ BTC).
- **Nguyên tắc Chống rò rỉ (Anti-leakage)**: Không sử dụng nhãn test để fit hoặc tuning cutoff; không tạo file `metrics.json` giả khi dự đoán trên tập test.

---

## 4. Nghiệm thu Cổng Chất lượng (Acceptance Gates Verification)

Đối chiếu trực tiếp với Bảng Cổng Nghiệm thu tại Mục 14 của Bản kế hoạch triển khai:

| Cổng Nghiệm thu (Gate) | Tiêu chí Yêu cầu | Bằng chứng Thực tế | Kết quả |
|---|---|---|:---:|
| **Input Identity** | Đủ 1.200 query; IDs hợp lệ signed int64; corpus 4.39M links | 1.200 queries nạp từ `query.parquet`; registry chứa 4.394.718 URL | **PASS** |
| **Crawl Resume** | Trạng thái fetch ghi nhận rõ ràng; an toàn khi resume | Checkpoint SQLite WAL; phân biệt `fetched`, `pending`, `retry_wait` | **PASS** |
| **Extraction** | Trích xuất văn bản có cấu trúc; lọc soft-404 | Trafilatura trích xuất blocks/paragraphs; 100% tiếng Việt hợp lệ | **PASS** |
| **Source Provenance** | 100% chunks có exact spans khớp với source document | `validate_zip` xác minh 12.000/12.000 chunk text là exact substring | **PASS** |
| **Token Budget** | Reranker/Dense window giới hạn token | Chunker clamp 320 body tokens + 64 overlap tokens | **PASS** |
| **Snapshot Consistency** | Hash manifest khớp dữ liệu Parquet | `manifest.json` ghi nhận đầy đủ SHA-256 của từng shard | **PASS** |
| **Evaluator Logic** | 2-row DP LCS; Macro $F_2$ trung bình cộng | `vibiomir_metrics` kiểm thử bằng tay và unit test đều khớp | **PASS** |
| **Prediction Coverage** | 1.200 completed query; 0 inference failures | `predictions_checkpoint.jsonl` hoàn tất đủ 1.200 queries | **PASS** |
| **Export Schema** | Kết quả nộp bài đúng chuẩn JSON/ZIP | Duy nhất 1 file `results.json` ở root; IDs int thực | **PASS** |
| **Reproducibility** | Xuất kết quả có tính lặp lại tất định | Cùng run fingerprint sinh mã băm ZIP xác định | **PASS** |
| **Full Regression** | Không có regression trong toàn repo | Toàn bộ **354/354 tests PASS** (0 failures, 15.27s) | **PASS** |

---

## 5. Artifacts Được Tạo & Lưu trữ

Toàn bộ artifacts của đợt chạy nghiệm thu Baseline A được lưu trữ tại `artifacts/experiments/vibiomir_phase1_A/`:

1. `submission_A.zip` (3.8 MB): File ZIP nộp bài chuẩn quy định BTC.
2. `results.json` (21.2 MB uncompressed): Toàn bộ 1.200 kết quả dự đoán dạng JSON.
3. `run_manifest.json`: Fingerprint phiên chạy (`c28a7e72...`), thời gian chạy, seed 42.
4. `predictions_checkpoint.jsonl`: Checkpoint từng query bảo đảm khả năng crash-recovery.
5. `retrieval_debug.jsonl`: Trace chi tiết 1.200 truy vấn (variants, candidate counts, cutoff).
6. `predictions.jsonl`: Dữ liệu dự đoán nội bộ kèm normalized scores.
7. `validation_report.json`: Báo cáo xác thực tính hợp lệ độc lập.

---

## 6. Hạn chế Hiện tại & Định hướng Phase 2

1. **Độ phủ dữ liệu Web (Corpus Crawl Coverage)**:
   - Baseline A hoạt động trên pilot snapshot (85 tài liệu / 296 chunks).
   - *Phase 2*: Mở rộng crawler theo batch quy mô lớn trên các tên miền y tế trọng điểm (Vinmec, Medlatec, Bệnh viện Bạch Mai, Bộ Y tế) để nâng cao độ phủ thực tế.
2. **Dense Semantic & Reranking Integration**:
   - Baseline A dùng BM25S làm nòng cốt tìm kiếm từ khóa.
   - *Phase 2*: Kết hợp dense representation (Multilingual E5) và cross-encoder reranking (BGE-Reranker-v2-m3) thông qua Weighted RRF fusion.
3. **Calibrated Cutoff F2 Optimization**:
   - Baseline A áp dụng fixed top-K (`5` documents, `10` chunks).
   - *Phase 2*: Ứng dụng `dev_annotation_guide.md` để gán nhãn 100–200 câu hỏi dev lâm sàng, chạy `CutoffCalibrator` tối ưu hóa điểm số Macro $F_2$.
