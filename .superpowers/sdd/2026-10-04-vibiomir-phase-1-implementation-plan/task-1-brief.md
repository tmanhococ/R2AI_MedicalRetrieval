### Task 1: Official inputs, ID contracts và registry import

**Files:** Create vibiomir_contracts.py, vibiomir_registry.py, phần input của vibiomir_adapter.py; config; tests/data/test_vibiomir_inputs.py.

**Interfaces:**

- iter_official_sources(path: Path, batch_size: int = 10000) -> Iterator[OfficialSource]
- load_official_queries(path: Path, expected_count: int | None) -> tuple[OfficialQuery, ...]
- ingest_sources(rows: Iterable[OfficialSource], registry: SourceRegistry) -> IngestReport
- IngestReport: input_rows, inserted_sources, unique_urls, rejected_rows, input_sha256.

- [ ] Viết tests test_sparse_int64_ids_and_alias_urls, test_reject_bool_float_null_and_duplicate_ids, test_query_exact_set_and_count, test_ingest_resume_idempotent. Assert IDs 583 và 4294967297 được giữ; hai IDs cùng URL vẫn có hai rows; duplicate IDs làm ingest thất bại transactional.
- [ ] Run: pytest tests/data/test_vibiomir_inputs.py -q. Expected: FAIL do adapter/registry chưa có hoặc contract chưa đáp ứng; không chấp nhận FAIL do test fixture lỗi.
- [ ] Implement ParquetFile.iter_batches(columns=["id","url"]) và query loader id/query; định nghĩa validated config types và SubmissionRow/SubmissionChunk contracts dùng ở tasks sau. Validate actual schema, null/empty values, signed int64; không dùng index dataframe làm ID.
- [ ] Implement staging registry/cursor/complete flag theo mục 4.2; batch transaction và resume cùng checksum. Duplicate ID trong input không được che bằng upsert. Nếu file đổi, dùng registry version mới hoặc migration minh thị, không append như cùng dataset.
- [ ] Run tests. Expected: PASS, exact IDs và counts.
- [ ] Commit: feat: add ViBioMIR official inputs and source registry.

