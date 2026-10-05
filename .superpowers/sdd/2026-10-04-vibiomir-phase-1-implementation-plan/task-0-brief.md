### Task 0: Audit repo thật và cố định integration map

**Files:** Read repo entrypoints/config/dependencies; create docs/vibiomir/phase1_integration_map.md khi triển khai.

**Interfaces:** Consumes spec gốc và source repo. Produces mapping từng file ở mục 11 → file thật, command regression suite hiện có, baseline config và model revisions.

- [ ] Xác định AGENTS.md, dependency manager, adapters, schemas, index APIs, prediction format và exporter hiện có. Không xem tên branch trong spec là bằng chứng branch đang tồn tại.
- [ ] Chạy regression/smoke test đang có; ghi command, exit code và failures hiện hữu. Không sửa lỗi không thuộc Phase 1 trừ khi chặn pipeline.
- [ ] Ghi public signatures cần giữ, migration legacy artifacts và những modules có thể reuse. Nếu một chức năng chưa có như spec mô tả, đánh dấu gap và thêm task cụ thể; không report là đã triển khai.
- [ ] Commit integration map sau khi xác minh; chỉ dùng git trong repo thật, không tạo repo rỗng để “commit kế hoạch”.

**Exit criterion:** Có code paths thật và một baseline có thể chạy hoặc danh sách blocker có thể tái hiện. Không bắt đầu sửa hàng loạt khi thiếu repo.

