# ViBioMIR trên Modal (v2)

Chạy phần còn lại của pipeline (embedding, BM25, dịch query, truy vấn, rerank, xuất submission) trên Modal.
Mọi lệnh chạy từ thư mục `mvp-rag-medical/`. Lý do có bản v2 nằm trong `PLAN_V2.md`. Tóm tắt: corpus chính thức
hiện tại có 1.220.598 link. 75% tài liệu đã crawl nằm ngoài corpus này, và 240.000 link PubMed chưa hề được tải.

| File | Chạy ở đâu | Việc |
|---|---|---|
| `prepare_upload.py` | máy local | Chỉ giữ tài liệu thuộc corpus chính thức (`--official-corpus`); lọc chunk trùng ID và boilerplate; chia lại part theo ranh giới tài liệu |
| `fetch_pubmed.py` | máy local | Lấy tiêu đề và abstract của 240.000 link PubMed từ Europe PMC (chạy lại được, tối đa 2 request/giây) |
| `build_extra_parts.py` | máy local | Chuyển bản ghi PubMed thành `part_9xxxx.parquet` trong bundle |
| `upload.py` | máy local | Upload bundle lên volume `vibiomir-v2`, bỏ qua file đã có |
| `app.py` | Modal | Toàn bộ các stage trên cloud |
| `vb_common.py` | cả hai | Tokenizer BM25 (bigram tiếng Trung, stemming tiếng Anh), chọn kết quả, xuất `results.json` và zip |
| `local_smoke.py` | máy local (GPU) | Chạy mọi stage với model thật và Lucene thật trên một lát dữ liệu nhỏ |
| `select_export.py` | máy local | Chọn lại kết quả từ `scored_pool.parquet` mà không tốn credit |
| `check_run.py` | máy local | Kiểm tra một run đã tải về: định dạng zip, query rỗng, độ đa dạng doc top-1, điểm reranker |

## 0. Chuẩn bị (một lần)

```powershell
uv pip install --python .venv\Scripts\python.exe modal snowballstemmer   # venv do uv tạo nên không có pip
.venv\Scripts\modal.exe setup
```
Modal yêu cầu có thẻ thanh toán thì mới dùng được GPU. Gói Starter có 30 USD credit mỗi tháng và tối đa 10 GPU chạy cùng lúc.
Corpus chính thức: `https://huggingface.co/datasets/AIGuruTinix/ViBioMIR/resolve/main/links_corpus_vi_en_zh.parquet`
(đã tải sẵn về `..\hf_check\`).

## 1. Dựng bundle v2 (đã chạy sẵn đêm 25/10, kết quả ở `artifacts\modal_upload_v2`)

Snapshot 22 GB `phase2_full` đã được thu gọn thành `phase2_official` (2,9 GB), chỉ giữ tài liệu thuộc corpus chính thức,
bằng `slim_snapshot.py`. Đã kiểm chứng: dựng bundle từ bản thu gọn cho ra 34 part giống hệt bundle v2.
Bản 22 GB đã được xoá theo yêu cầu. Phải chạy `build_extra_parts.py` lại sau mỗi lần chạy `prepare_upload.py`,
vì `prepare_upload.py` xoá thư mục `chunks/`.

```powershell
.venv\Scripts\python.exe modal_pipeline\prepare_upload.py --snapshot artifacts\vibiomir\snapshots\phase2_official --queries data\official\query.parquet --official-corpus ..\hf_check\links_corpus_vi_en_zh.parquet --out artifacts\modal_upload_v2
.venv\Scripts\python.exe modal_pipeline\fetch_pubmed.py --official-corpus ..\hf_check\links_corpus_vi_en_zh.parquet --out artifacts\pubmed
.venv\Scripts\python.exe modal_pipeline\build_extra_parts.py --pubmed artifacts\pubmed --bundle artifacts\modal_upload_v2
.venv\Scripts\python.exe modal_pipeline\upload.py artifacts\modal_upload_v2
```

## 2. Đo chi phí embedding (khoảng 0,1 USD)

```powershell
.venv\Scripts\modal.exe run modal_pipeline\app.py --stage bench
```
Lệnh in ra `gpu_hours` và `est_usd_L4`. Nếu vượt ngân sách, đặt `"max_len": 256` trong file opts.

## 3. Chạy thử trên 1 part và 30 query (khoảng 0,3 USD)

```powershell
.venv\Scripts\modal.exe run modal_pipeline\app.py --stage all --parts 1 --nq 30
.venv\Scripts\python.exe modal_pipeline\check_run.py artifacts\modal_runs\v1\p1_q30 --nq 30 --min-median-ce -3
```

## 4. Chạy đầy đủ

```powershell
.venv\Scripts\modal.exe run --detach modal_pipeline\app.py --stage all
```
Với `--detach`, pipeline vẫn chạy nếu bạn tắt máy, vì bộ điều phối `pipeline` chạy trên Modal. Khi terminal còn mở,
`submission.zip`, `selection.json` và `scored_pool.parquet` được tải về `artifacts\modal_runs\v1\full\`.
Nếu terminal đã đóng thì tải bằng lệnh:

```powershell
.venv\Scripts\modal.exe volume get vibiomir-v2 runs/v1/full/submission.zip artifacts\modal_runs\v1\full\
.venv\Scripts\modal.exe volume get vibiomir-v2 runs/v1/full/scored_pool.parquet artifacts\modal_runs\v1\full\
.venv\Scripts\python.exe modal_pipeline\check_run.py artifacts\modal_runs\v1\full
```

Mỗi stage bỏ qua phần việc đã có kết quả trên volume. Nếu một stage lỗi, sửa rồi chạy lại đúng stage đó, ví dụ `--stage rerank`.
Để tính lại sau khi sửa code hoặc đổi tham số tìm kiếm, đặt `"version": "v2"` trong file opts; không cần xoá gì trên
volume. Nếu đổi tham số mà giữ nguyên version, pipeline dừng và báo lỗi thay vì âm thầm dùng kết quả cũ.

## 5. Tinh chỉnh số doc/chunk xuất ra (miễn phí, chạy local)

```powershell
.venv\Scripts\python.exe modal_pipeline\select_export.py --pool artifacts\modal_runs\v1\full\scored_pool.parquet --queries data\official\query.parquet --out artifacts\modal_runs\v1\full\try_60 --doc-max 60 --chunk-max 60
```
Các tham số: `--doc-max --doc-min --doc-min-logit --chunk-max --chunk-min --chunk-min-logit --chunk-per-doc`.
Nộp vài biến thể lên leaderboard public (tối đa 10 bài mỗi ngày) để chọn ngưỡng.

## Tham số pipeline (`--opts file.json`)

Xem `DEFAULT_OPTS` trong `app.py`. Ví dụ `opts.json`:

```json
{"version": "v2", "max_len": 384, "pool_size": 300, "selection": {"doc_max": 80, "chunk_max": 80}}
```
