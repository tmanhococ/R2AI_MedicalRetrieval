# Trạng thái chạy — đêm 24→25/10/2026

## 1. Kết quả

| Task | Kết quả | Bằng chứng |
|---|---|---|
| T1–T4 của `PLAN.md` (code, test local, commit, push) | PASS | commit `16f85b0`; `pytest modal_pipeline/tests` → `6 passed`; sha remote = sha local |
| Preflight | PASS | `modal profile current` → `mle29022`; volume `vibiomir` truy cập được; pytest `6 passed` (commit `4c63bdf`) |
| Upload tiny | PASS | Trên volume có `data/query.parquet`, `data/upload_manifest.json`, `data/chunks/part_00000.parquet`, kích thước khớp file local |
| Chạy tiny trên Modal | **BỊ CHẶN** | Modal báo `Please add a payment method to use L4 GPU functions.`; `EXIT=1`; chưa function nào chạy |

**Lý do bị chặn:** tài khoản chưa có thẻ thanh toán. Modal bắt buộc có thẻ mới cho dùng GPU, kể cả khi trả bằng credit miễn phí. Theo kế hoạch, tôi dừng ở đây và không tìm cách lách. Log đầy đủ: `artifacts/modal_runs/p1_q30/run.log` (lần chạy này diễn ra trước khi đường dẫn output có thêm thư mục version).

**Những gì đã được kiểm chứng trên Modal (trong lúc build image):**
- Image CPU build xong.
- Tải và cài JDK 21 (Adoptium) xong.
- `pyserini==2.4.0` resolve và build wheel được (kèm torch, pyjnius).
- Lớp tải model E5 và reranker vào image, và lớp cài Pyserini, đang chạy dở thì app bị dừng. Lần chạy sau sẽ build tiếp hai lớp này (vài phút).

## 2. Thời gian và chi phí

- Upload: 27 MB với tốc độ khoảng 2 MB/s. Upload phần còn lại (2,34 GB) sẽ mất **khoảng 20 phút**.
- Build image khoảng 5 phút. Không function nào chạy, nên **không có chi phí GPU/CPU cho pipeline**; chỉ có phí build image (rất nhỏ). Xem chính xác tại https://modal.com/settings/usage.

## 3. Lỗi đã gặp và cách sửa

| Lỗi | Cách xử lý |
|---|---|
| Chưa có phương thức thanh toán nên không dùng được GPU | Không sửa được bằng code. Bạn cần thêm thẻ (bước 1 bên dưới) |

Code không phát sinh lỗi nào trong lần chạy này.

**Sửa sau review toàn nhánh** (reviewer độc lập; 0 lỗi Critical; API Pyserini 2.4.0 đã được đối chiếu với mã nguồn):
- Danh sách shard cũ: sau khi upload thêm part, `meta` giờ tự build lại. Trước đây `bench` chỉ thấy 1 part nên báo chi phí thấp hơn thật khoảng 84 lần, và BM25 có thể chỉ index 1 part mà không báo gì. Marker `.done` của BM25 giờ lưu số dòng và bị kiểm tra lại.
- Output trên volume giờ nằm theo `version` (mặc định `v1`) và `max_len`: `emb/v1_len512/`, `bm25/v1/`, `runs/v1/`. Nếu đổi tham số tìm kiếm mà giữ nguyên version, pipeline dừng và báo lỗi thay vì âm thầm dùng kết quả cũ. Sau khi sửa code, đặt `"version": "v2"` để tính lại.
- Stage dịch: tải NLLB và commit cache trước khi load model (tránh crash do commit khi file đang mở).
- `retries` của stage GPU giảm từ 2 xuống 1 (lỗi lặp lại không còn tốn gấp 3).
- Thêm `check_run.py`: kiểm tra định dạng, query rỗng, số doc top-1 khác nhau, và median điểm reranker top-1 > 0.

## 4. Sáng mai chạy gì

Chạy trong PowerShell, tại `D:\Research\R2AI\mvp-rag-medical`:

```powershell
$env:PYTHONUTF8 = "1"

# 1. Thêm thẻ thanh toán: https://modal.com/settings/billing
#    Gói Starter vẫn tặng 30 USD credit/tháng; thẻ chỉ bị trừ khi dùng vượt credit.

# 2. Chạy lại bản tiny (khoảng 0,3–0,5 USD, 10–20 phút). Lần này cũng kiểm chứng GPU, Pyserini, model.
.venv\Scripts\modal.exe run modal_pipeline\app.py --stage all --parts 1 --nq 30
#    Pass khi: log có đủ các dòng meta, encode, bm25 index, translate, dense search, bm25 search,
#    pool, rerank, final; dòng final có "queries": 30; và lệnh dưới in ra PASS:
.venv\Scripts\python.exe modal_pipeline\check_run.py artifacts\modal_runs\v1\p1_q30 --nq 30

# 3. Upload phần còn lại (miễn phí, khoảng 20 phút, chạy lại được nếu rớt mạng)
.venv\Scripts\python.exe modal_pipeline\upload.py artifacts\modal_upload

# 4. Đo chi phí embedding thật trước khi chạy full (khoảng 0,1 USD)
.venv\Scripts\modal.exe run modal_pipeline\app.py --stage bench
#    Đọc các giá trị gpu_hours và est_usd_L4. Nếu quá ngân sách, tạo opts.json {"max_len": 256} và thêm --opts opts.json

# 5. Chạy full (khoảng 6–10 USD, 2–3 giờ). Vẫn chạy khi bạn tắt máy.
.venv\Scripts\modal.exe run --detach modal_pipeline\app.py --stage all

# 6. Tải kết quả (nếu terminal ở bước 5 đã đóng), rồi kiểm tra
.venv\Scripts\modal.exe volume get vibiomir runs/v1/full/submission.zip artifacts\modal_runs\v1\full\
.venv\Scripts\modal.exe volume get vibiomir runs/v1/full/scored_pool.parquet artifacts\modal_runs\v1\full\
.venv\Scripts\python.exe modal_pipeline\check_run.py artifacts\modal_runs\v1\full

# 7. Tạo thêm biến thể số doc/chunk để nộp thử (miễn phí, chạy local)
.venv\Scripts\python.exe modal_pipeline\select_export.py --pool artifacts\modal_runs\v1\full\scored_pool.parquet --queries data\official\query.parquet --out artifacts\modal_runs\v1\full\try_60 --doc-max 60 --chunk-max 60
```

## 5. Lưu ý

- Sau khi upload thêm part, `build_meta` tự chạy lại để nhận toàn bộ 84 part (cả với `--stage bench`). Embedding của `part_00000` từ bản tiny được dùng lại nếu cùng `version` và `max_len`, không tính tiền lần hai.
- Đặt giới hạn chi tiêu (budget limit) cho workspace trong https://modal.com/settings/billing trước khi chạy full. Khi đã có thẻ, phần vượt 30 USD credit sẽ bị trừ vào thẻ.
- Bản dịch của 30 query từ bản tiny được cache; bản full chỉ dịch thêm 1.170 query còn lại.
- Mỗi stage bỏ qua phần đã có kết quả. Nếu một stage lỗi giữa chừng, sửa lỗi rồi chạy lại `--stage <tên>` mà không mất phần đã xong.
- Theo dõi chi phí tại https://modal.com/settings/usage trong lúc chạy full. Muốn dừng ngay: `.venv\Scripts\modal.exe app stop vibiomir`.
