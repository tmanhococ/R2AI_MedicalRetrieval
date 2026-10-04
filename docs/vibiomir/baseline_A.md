# Báo cáo Nghiệm thu Baseline A: ViBioMIR Phase 1 Retrieval

## 1. Tóm tắt Tổng quan (Executive Summary)

Baseline A thiết lập đường chạy chuẩn nghiệm thu đầu tiên kết nối toàn diện hệ thống retrieval của Road to AI với dữ liệu và thể lệ chính thức của cuộc thi **ViBioMIR 2026** (Track Medical Retrieval).

Toàn bộ quy trình từ kiểm tra nguồn dữ liệu Parquet, nạp SQLite registry, lấy mẫu pilot phân tầng 500 URLs (seed 42), thu thập và trích xuất cấu trúc, đóng băng snapshot Parquet, lập chỉ mục sparse BM25S, suy luận dự đoán có checkpoint phục hồi lỗi, đến đóng gói và thẩm định bài nộp đã hoàn tất và vượt qua 100% các tiêu chí nghiệm thu.

- **Mã định danh gói nộp bài**: `submission_A.zip`
- **Mã băm SHA-256 gói nộp bài**: `8ee757873b18ddec7775a333907f83778110766163a3ceddc2980e1c1a03e6ee`
- **Kích thước gói nộp**: **5,44 MB** (5.699.921 bytes).
- **Độ phủ câu hỏi chính thức**: **1.200 / 1.200 queries** ($100{,}0\%$).
- **Số tài liệu dự đoán**: **6.000 documents** (chính xác 5 tài liệu / câu hỏi).
- **Số chunk dự đoán**: **12.000 chunks** (chính xác 10 chunks / câu hỏi).
- **Tỷ lệ truy vết nguồn chính xác (Provenance)**: **100,0%** (12.000/12.000 chunks có exact text span khớp hoàn hảo văn bản gốc trong snapshot).
- **Số lỗi / cảnh báo nghiệm thu**: **0 errors, 0 warnings** (qua thẩm định ngoại tuyến `validate_submission`).
- **Thời gian suy luận (Runtime)**: **26,05 giây** trên CPU.
- **Tài nguyên tiêu thụ**:
  - **Peak RAM**: **10,07 MB** (đo đạc thực tế qua `tracemalloc`).
  - **GPU VRAM**: **0,0 MB** (BM25S chạy thuần CPU không phụ thuộc GPU).
  - **Lỗi suy luận (Inference Failures)**: **0**.

---

## 2. Dữ liệu & Snapshot Cố định (Frozen Snapshot)

### 2.1 Định danh Dữ liệu Đầu vào
- `data/official/query.parquet`: 1.200 câu hỏi chính thức (`id`: int64, `query`: string tiếng Việt lâm sàng). Mã băm SHA-256: `ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08`.
- `data/official/links_corpus.parquet`: 4.394.718 liên kết nguồn (`id`: int64, `url`: string). Mã băm SHA-256: `dee1153f5b9d9010dea18e1b7724b4915ff847b711d036bec8b933486802a971`.

### 2.2 Thuộc tính Snapshot `phase1_A`
- **Snapshot ID**: `phase1_A`
- **Source Membership Hash**: `42077515dcf25c4543c8302d335a81365edc8ff295b9f047fc150e212bcbfd8c`
- **Extractor Fingerprint**: `18de584ab468e8f624b1b1dd748a4fba6348ff18b1de7e191b3a599a4b73a971`
- **Chunker Fingerprint**: `89bc0dcae1261557022449bd6f8098e3037f4e43a5474c0132f00205c6d69c50`
- **Số lượng tài liệu sẵn sàng (Ready)**: 465 tài liệu sạch (197 tiếng Việt, 268 tiếng Trung).
- **Số lượng tài liệu cách ly (Quarantined)**: 3 tài liệu lỗi soft-error rỗng (`empty_text`).
- **Số lượng chunk đoạn văn**: 1.238 chunks (paragraph token spans, `body_max_tokens: 320`, `long_block_overlap: 64`).
- **Phân vùng Shards Parquet**:
  - `documents_00000.parquet`: 465 rows.
  - `chunks_00000.parquet`: 1.238 rows.
  - `lookup.sqlite`: Chỉ mục B-tree tra cứu 4.394.718 official doc IDs và 1.238 chunk locations.

---

## 3. Lấy Mẫu Phân Tầng Pilot & Thu Thập Dữ Liệu

Theo quy định tại Mục 11 & 14 của kế hoạch: *"Chạy pilot 500 URLs stratified theo domain và URL type, seed 42; sample IDs được lưu. Không chỉ lấy 500 rows đầu."*

1. **Thuật toán Phân tầng (Stratified Sampling)**:
   - 4.394.718 URLs được chia thành **178 tầng** theo cặp `(domain, url_type)` với các dạng URL: `.html`, `.pdf`, `.php/.aspx` (dynamic), và slug/REST.
   - Hạn ngạch (quota) được phân bổ theo phương pháp phần dư lớn nhất (Largest Remainder Method) đảm bảo mỗi tầng có đại diện (tối thiểu 1 mẫu) và phân bổ cân xứng cho các domain lớn.
   - Sử dụng `seed=42`, toàn bộ 500 job IDs được lưu trữ tại `artifacts/vibiomir/pilot_sample_ids.json`.
2. **Kết quả Thu thập Pilot**:
   - Số job được cấp phát: 500.
   - Tải thành công (`fetched`): 383 URLs.
   - Bị chặn bởi robots.txt (`blocked`): 93 URLs (ghi nhận mã lỗi `robots_disallowed`, tuân thủ chuẩn lịch sự).
   - Đang chờ thử lại (`retry_wait`): 24 URLs (kết nối chậm/timeout được lên lịch backoff).
   - Thất bại vĩnh viễn (`fetch_failed`): 0.
   - Dung lượng tải về: 6,30 MiB nén gzip lưu trữ content-addressed.
   - Tổng số URLs ở trạng thái `fetched` trong registry đạt 468 URLs.

---

## 4. Kiểm tra Định tính Mẫu 30 Nguồn & 20 Chunks Xuất (Qualitative Audit)

Để đảm bảo chất lượng trích xuất và tính toán span, quy trình đã tiến hành kiểm tra mẫu sâu:

### 4.1 Đánh giá 30 Nguồn Dữ liệu Đa dạng
- **Chất lượng Trích xuất Cấu trúc**: Parser Trafilatura loại bỏ triệt để các khối điều hướng, banner quảng cáo, thực đơn và chân trang rác; giữ trọn vẹn tiêu đề, danh sách triệu chứng và bảng biểu y khoa.
- **Bảo toàn Cặp Hỏi - Đáp (QA Extraction)**: Các bài viết tư vấn sức khỏe từ `ask.39.net`, `bachmai.gov.vn` bảo toàn đầy đủ cả phần câu hỏi của bệnh nhân và câu trả lời chỉ dẫn của bác sĩ.
- **Phát hiện Trang lỗi (Soft-404 / Empty Pages)**: Hệ thống phát hiện chính xác 3 trang rỗng không có nội dung hữu ích và tự động chuyển sang diện cách ly (`quarantine_counts: {'empty_text': 3}`), không để tài liệu rác lọt vào snapshot.
- **Nhận diện Ngôn ngữ (Language ID)**: Mô hình fastText LID gán nhãn chuẩn xác 15 bài tiếng Việt và 15 bài tiếng Trung trong mẫu ngẫu nhiên (độ tin cậy > 0.85); không xảy ra tình trạng ép sai ngôn ngữ tiếng Trung về tiếng Việt.
- **Kiểm chứng Spans Tuyệt đối**: Đối chiếu 100% chunks thuộc 30 tài liệu mẫu:
  $$\text{doc.source\_text}[\text{start\_char}:\text{end\_char}] == \text{chunk.text}$$
  Kết quả đạt **100% khớp tuyệt đối** (không lệch một ký tự Unicode, dấu câu hay ký tự xuống dòng).
- **Ngân sách Token (Token Budget)**: Toàn bộ chunk tuân thủ nghiêm ngặt giới hạn $\le 320$ body tokens (giá trị token cao nhất đo được là 312 tokens, 0 trường hợp tràn ngân sách).

### 4.2 Đánh giá 20 Chunks Xuất từ `results.json`
- Lấy mẫu 20 chunks xuất xưởng tương ứng với 20 câu hỏi rải đều từ query ID 1 đến 1200.
- **Tính xác thực nguồn**: 20/20 chunks ($100{,}0\%$) là chuỗi con nguyên bản (exact substring) của văn bản gốc trong Parquet shards.
- **Thứ tự xếp hạng**: `chunk_order` xuất đều đặn từ $0 \dots 9$, bảo toàn thứ tự rank từ bộ sắp xếp BM25S.
- **Không rò rỉ metadata**: Không xuất trường điểm số, chunk ID nội bộ, hash hay offset trong file nộp.

---

## 5. Nghiệm thu 12 Cổng Chất lượng (Acceptance Gates Verification)

Đối chiếu trực tiếp với Bảng Cổng Nghiệm thu tại Mục 14 của Bản kế hoạch triển khai:

| Cổng Nghiệm thu (Gate) | Tiêu chí Yêu cầu | Bằng chứng Thực tế | Kết quả |
|---|---|---|:---:|
| **1. Input Identity** | Đủ 1.200 query; IDs hợp lệ signed int64; corpus 4.39M links | 1.200 queries nạp từ `query.parquet`; registry chứa 4.394.718 URL | **PASS** |
| **2. Crawl Resume** | Trạng thái fetch ghi nhận rõ ràng; phân tầng seed 42; an toàn khi resume | Phân tầng 178 strata; lưu `pilot_sample_ids.json`; phân biệt `fetched`, `blocked`, `retry_wait` | **PASS** |
| **3. Extraction** | Trích xuất văn bản có cấu trúc; lọc soft-404; audit mẫu 30 nguồn | Trafilatura trích xuất cấu trúc sạch; cách ly 3 trang rỗng; audit 30 nguồn đạt 100% chuẩn | **PASS** |
| **4. Source Provenance** | 100% chunks có exact spans khớp với source document | `validate_submission` xác minh 12.000/12.000 chunk text là exact substring | **PASS** |
| **5. Token Budget** | Reranker/Dense window giới hạn token; không cắt âm thầm | Chunker clamp 320 body tokens + 64 overlap tokens; max token 312 | **PASS** |
| **6. Snapshot Consistency** | Hash manifest khớp dữ liệu Parquet | `manifest.json` ghi nhận đầy đủ SHA-256 của từng shard; membership SHA khớp | **PASS** |
| **7. Evaluator Logic** | 2-row DP LCS; Macro $F_2$ trung bình cộng; khớp công thức | `vibiomir_metrics` kiểm thử bằng tay và unit test đều khớp công thức BTC | **PASS** |
| **8. Scorer Parity** | Minh thị trạng thái parity với BTC; nếu chưa có tokenizer BTC ghi false | `false` - Ghi nhận minh thị: Profile `provisional_spec_v1` dùng vi syllable/regex LCS v1 do BTC chưa công bố official tokenizer | **PASS** (Disclosed) |
| **9. Prediction Coverage** | 1.200 completed query; đo runtime & RAM/GPU; 0 inference failures | 1.200 queries hoàn tất; runtime 26.05s; Peak RAM 10.07 MB; GPU 0.0 MB; 0 failures | **PASS** |
| **10. Export Schema** | Kết quả nộp bài đúng chuẩn JSON/ZIP; 1 results.json ở root | Duy nhất 1 file `results.json` ở root; IDs int thực; 0 duplicate doc/chunk | **PASS** |
| **11. Reproducibility** | Xuất kết quả có tính lặp lại tất định | Cùng run fingerprint `eea46199...` sinh mã băm ZIP xác định | **PASS** |
| **12. Full Regression** | Không có regression trong toàn repo | Toàn bộ **354/354 tests PASS** (0 failures, 15.33s) | **PASS** |

---

## 6. Danh mục Artifacts Được Tạo & Lưu trữ

Các artifacts được tổ chức chuẩn xác theo Mục 15 của bản kế hoạch:

| File Artifact | Vị trí Lưu trữ | Mô tả & Nội dung |
|---|---|---|
| `submission_A.zip` | `artifacts/experiments/vibiomir_phase1_A/` | File ZIP nộp bài chuẩn quy định BTC (5,44 MB, SHA-256: `8ee75787...`). |
| `results.json` | `artifacts/experiments/vibiomir_phase1_A/` | Toàn bộ 1.200 kết quả dự đoán dạng JSON mảng gốc. |
| `coverage.json` | `artifacts/experiments/vibiomir_phase1_A/` & `artifacts/vibiomir/raw/` | Độ phủ registry 4.39M, 468 fetched, 93 blocked, host breakdown và language counts. |
| `pilot_sample_ids.json` | `artifacts/vibiomir/` | Danh sách 500 job IDs phân tầng ngẫu nhiên (seed 42) trên 178 tầng. |
| `run_manifest.json` | `artifacts/experiments/vibiomir_phase1_A/` | Run fingerprint (`eea46199...`), runtime 26.05s, Peak RAM 10.07 MB, GPU 0.0 MB. |
| `predictions_checkpoint.jsonl` | `artifacts/experiments/vibiomir_phase1_A/` | Checkpoint từng query bảo đảm khả năng crash-recovery. |
| `retrieval_debug.jsonl` | `artifacts/experiments/vibiomir_phase1_A/` | Trace chi tiết 1.200 truy vấn (candidates, raw scores, cutoff). |
| `predictions.jsonl` | `artifacts/experiments/vibiomir_phase1_A/` | Dữ liệu dự đoán nội bộ kèm normalized scores. |
| `validation_report.json` | `artifacts/experiments/vibiomir_phase1_A/` | Báo cáo kiểm định ngoại tuyến xác nhận 0 lỗi. |

---

## 7. Hạn chế Hiện tại & Định hướng Phase 2

1. **Độ phủ dữ liệu Web (Corpus Crawl Coverage)**:
   - Baseline A hoạt động trên pilot snapshot (465 tài liệu sạch / 1.238 chunks).
   - *Phase 2*: Mở rộng crawler theo batch quy mô lớn trên các tên miền y tế trọng điểm (Vinmec, Medlatec, Bệnh viện Bạch Mai, Bộ Y tế, 120ask, cnkang) để nâng cao độ phủ thực tế.
2. **Dense Semantic & Reranking Integration**:
   - Baseline A dùng BM25S làm nòng cốt tìm kiếm từ khóa.
   - *Phase 2*: Kết hợp dense representation (Multilingual E5) và cross-encoder reranking (BGE-Reranker-v2-m3) thông qua Weighted RRF fusion.
3. **Calibrated Cutoff F2 Optimization**:
   - Baseline A áp dụng fixed top-K (`5` documents, `10` chunks).
   - *Phase 2*: Ứng dụng `dev_annotation_guide.md` để gán nhãn 100–200 câu hỏi dev lâm sàng, chạy `CutoffCalibrator` tối ưu hóa điểm số Macro $F_2$.
