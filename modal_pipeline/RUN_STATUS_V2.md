# Pipeline v2 — trạng thái sáng 25/10/2026

Nhánh `feature/modal-pipeline-v2`. Kế hoạch và lý do nằm trong `PLAN_V2.md`. Hướng dẫn đầy đủ nằm trong `README.md`.

## Thay đổi chính so với v1

| | v1 | v2 |
|---|---|---|
| Corpus dùng để index | 4,39 triệu link (danh sách crawl cũ) | **corpus chính thức hiện tại trên HF: 1.220.598 link** |
| Tài liệu ngoài corpus chính thức trong index | **74,6%** (trả về là chắc chắn sai) | **0** (lọc khi chuẩn bị dữ liệu, chặn thêm ở `finalize`) |
| Độ phủ corpus chính thức | 55,4% (676.190 doc) | **75,1%** (916.150 doc) |
| Tài liệu PubMed tiếng Anh (20% corpus) | 0 | **239.771 / 240.000** (Europe PMC) |
| Chunk cần embedding | 6.674.146 | **2.928.239** (−56%) |
| Token cần embedding (ước tính) | ≈ 2,0 tỷ | ≈ 0,9 tỷ |
| BM25 tiếng Anh | không stemming | Snowball stemming cho chunk và query tiếng Anh |
| Đã chạy thật end-to-end | chưa | **có, ở local với model và Lucene thật (xem bên dưới)** |

Dữ liệu đã nằm sẵn trên Modal: volume **`vibiomir-v2`**, 41 file, kích thước khớp với local (`artifacts/modal_upload_v2`, 942 MB).

## Bằng chứng "chạy được"

`local_smoke.py` chạy **mọi stage** của `app.py` trên máy (RTX 3050 Ti) với model thật: E5-base,
bge-reranker-v2-m3, NLLB (bản 600M vì GPU 4 GB), và Lucene thật qua Pyserini 2.4.0. Dữ liệu là
3.000 dòng tiếng Việt + 3.000 dòng PubMed, 10 query:

```
[meta] [translate] [encode ×2] [bm25 index] [dense search] [bm25 search] [pool] [rerank ×2] [final]
SMOKE PASS — 10 query, định dạng hợp lệ, 10 doc top-1 khác nhau
```

Smoke đã bắt được **3 lỗi sẽ làm crash trên Modal**. Cả 3 đã sửa, có test hoặc smoke xác nhận:
1. Pyserini 2.4 tạo client OpenAI ngay lúc import; openai ≥ 2 từ chối key rỗng, nên `bm25_search` crash. Sửa bằng biến giữ chỗ `OPENAI_API_KEY` trong image BM25 (không bao giờ được dùng).
2. `rerank_batch` crash khi section/title rỗng (pandas trả về NaN). Đã sửa, có test.
3. NLLB chỉ có `pytorch_model.bin`, mà transformers đòi torch ≥ 2.6 để load file này. Image GPU giờ pin `torch>=2.6`.

Ngoài ra, khi chạy `prepare_upload --official-corpus` trên dữ liệu thật đã lộ một lỗi `IndexError` (tra bảng đếm boilerplate với dòng ngoài corpus). Đã sửa, có test hồi quy.

Dịch query có tác dụng thấy được ngay trên smoke: median điểm reranker top-1 tăng từ −0,90 lên −0,32. Ví dụ query
"Kết quả của Beta-HCG là dưới 0.10 U/L…" lấy về abstract PubMed tiếng Anh về "Serum HCG-beta".

## Những gì không làm được và lý do

- **NCBI E-utilities chặn IP nhà bạn** ("temporarily blocked … possible misuse"). Có thể khối này đã có từ trước, vì tôi chỉ gửi 2 request. Tôi không lách khối mà lấy dữ liệu từ Europe PMC (EMBL-EBI), nguồn công khai độc lập phục vụ cùng nội dung MEDLINE. Muốn gỡ chặn: email info@ncbi.nlm.nih.gov.
- **304k tài liệu vi/zh chính thức vẫn thiếu**:
  - 160k bị robots chặn: tôn trọng robots, không crawl lại.
  - 76k "empty_text": HTML gốc thực chất là trang chống bot (laodong.vn: trang challenge; ask.39.net: captcha kéo-thả). Không có nội dung để khôi phục, và vượt captcha là không phù hợp.
  - 61k lỗi tải: có thể crawl lại sau bằng crawler sẵn có.
- **Chưa chạy trên Modal**: tài khoản chưa có thẻ, nên Modal không cho dùng GPU.

## Sáng nay chạy gì

PowerShell, tại `D:\Research\R2AI\mvp-rag-medical`:

```powershell
$env:PYTHONUTF8 = "1"
git checkout feature/modal-pipeline-v2
uv pip install --python .venv\Scripts\python.exe snowballstemmer

# 1. Thêm thẻ và đặt budget limit: https://modal.com/settings/billing

# 2. Chạy thử 1 part + 30 query (khoảng 0,3 USD; dữ liệu đã có trên vibiomir-v2, không cần upload)
.venv\Scripts\modal.exe run modal_pipeline\app.py --stage all --parts 1 --nq 30
.venv\Scripts\python.exe modal_pipeline\check_run.py artifacts\modal_runs\v1\p1_q30 --nq 30 --min-median-ce -3

# 3. Đo chi phí embedding thật (khoảng 0,1 USD)
.venv\Scripts\modal.exe run modal_pipeline\app.py --stage bench

# 4. Chạy full (ước tính 3–5 USD vì corpus nhỏ hơn v1 56%; khoảng 1,5–2,5 giờ), vẫn chạy khi tắt máy
.venv\Scripts\modal.exe run --detach modal_pipeline\app.py --stage all

# 5. Tải về (nếu terminal ở bước 4 đã đóng) và kiểm tra
.venv\Scripts\modal.exe volume get vibiomir-v2 runs/v1/full/submission.zip artifacts\modal_runs\v1\full\
.venv\Scripts\modal.exe volume get vibiomir-v2 runs/v1/full/scored_pool.parquet artifacts\modal_runs\v1\full\
.venv\Scripts\python.exe modal_pipeline\check_run.py artifacts\modal_runs\v1\full

# 6. Tạo biến thể số doc/chunk để nộp thử (miễn phí)
.venv\Scripts\python.exe modal_pipeline\select_export.py --pool artifacts\modal_runs\v1\full\scored_pool.parquet --queries data\official\query.parquet --out artifacts\modal_runs\v1\full\try_60 --doc-max 60 --chunk-max 60
```

Ước tính chi phí embedding: smoke đo được 62k token/giây trên GPU laptop; L4 nhanh hơn khoảng 3 lần, tức
≈ 1,3 giờ GPU L4 cho ≈ 0,9 tỷ token (≈ 1,3 USD). Bước `bench` sẽ cho con số thật.

## Lưu ý

- Volume cũ `vibiomir` không bị xoá và không còn được dùng.
- Muốn chạy lại smoke local: `..\.venv-gpu` đã cài sẵn. Lệnh ở đầu `local_smoke.py`.
