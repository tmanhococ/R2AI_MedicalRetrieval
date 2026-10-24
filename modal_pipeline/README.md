# ViBioMIR trên Modal

Chạy phần còn lại của pipeline (embedding, BM25, dịch query, truy vấn, rerank, xuất submission) trên Modal.
Mọi lệnh chạy từ thư mục `mvp-rag-medical/`.

| File | Chạy ở đâu | Việc |
|---|---|---|
| `prepare_upload.py` | máy local | Lọc chunk trùng ID và boilerplate, giữ cột cần thiết, chia lại part theo ranh giới tài liệu (22 GB thành vài GB) |
| `upload.py` | máy local | Upload bundle lên volume `vibiomir`, bỏ qua file đã có (chạy lại được khi rớt mạng) |
| `app.py` | Modal | Toàn bộ các stage trên cloud |
| `vb_common.py` | cả hai | Tokenizer BM25, chọn kết quả, xuất `results.json` và zip |
| `select_export.py` | máy local | Chọn lại kết quả từ `scored_pool.parquet` mà không tốn credit |
| `check_run.py` | máy local | Kiểm tra một run đã tải về: định dạng zip, query rỗng, độ đa dạng doc top-1, điểm reranker |

## 0. Chuẩn bị (một lần)

```powershell
uv pip install --python .venv\Scripts\python.exe modal   # venv do uv tạo nên không có pip
.venv\Scripts\modal.exe setup        # mở trình duyệt để đăng nhập
```
Modal yêu cầu có thẻ thanh toán trong tài khoản để dùng GPU. Gói Starter cho 30 USD credit mỗi tháng và tối đa 10 GPU chạy cùng lúc.

## 1. Thu gọn snapshot và upload

```powershell
.venv\Scripts\python.exe modal_pipeline\prepare_upload.py --snapshot artifacts\vibiomir\snapshots\phase2_full --queries data\official\query.parquet --out artifacts\modal_upload
.venv\Scripts\python.exe modal_pipeline\upload.py artifacts\modal_upload
```

## 2. Đo chi phí embedding (khoảng 0,1 USD)

```powershell
.venv\Scripts\modal.exe run modal_pipeline\app.py --stage bench
```
Lệnh in ra `gpu_hours` và `est_usd_L4`. Nếu con số vượt ngân sách, đặt `"max_len": 256` trong file opts.

## 3. Chạy thử trên 1 part và 30 query (khoảng 0,3 USD)

```powershell
.venv\Scripts\modal.exe run modal_pipeline\app.py --stage all --parts 1 --nq 30
```
Lần chạy này kiểm tra image, Java/Pyserini, GPU và định dạng submission. Embedding của part đầu được giữ lại cho lần chạy đầy đủ.

## 4. Chạy đầy đủ

```powershell
.venv\Scripts\modal.exe run --detach modal_pipeline\app.py --stage all
```
Với `--detach`, pipeline vẫn chạy nếu bạn tắt máy, vì bộ điều phối `pipeline` chạy trên Modal. Theo dõi log trên dashboard Modal.
Khi terminal còn mở, `submission.zip`, `selection.json` và `scored_pool.parquet` được tải về `artifacts/modal_runs/v1/full/`.
Nếu terminal đã đóng thì tải bằng lệnh:

```powershell
.venv\Scripts\modal.exe volume get vibiomir runs/v1/full/submission.zip artifacts\modal_runs\v1\full\
.venv\Scripts\modal.exe volume get vibiomir runs/v1/full/scored_pool.parquet artifacts\modal_runs\v1\full\
```

Mỗi stage bỏ qua phần việc đã có kết quả trên volume. Nếu một stage lỗi, sửa rồi chạy lại đúng stage đó, ví dụ `--stage rerank`.
Để tính lại sau khi sửa code hoặc đổi tham số tìm kiếm, đặt `"version": "v2"` trong file opts; không cần xoá gì trên volume. Nếu đổi tham số tìm kiếm mà giữ nguyên version, pipeline dừng và báo lỗi thay vì âm thầm dùng kết quả cũ.

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
