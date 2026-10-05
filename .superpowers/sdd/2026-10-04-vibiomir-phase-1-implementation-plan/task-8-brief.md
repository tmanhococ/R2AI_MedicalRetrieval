### Task 8: Official JSON/ZIP exporter và strict validator

**Files:** Create submission/vibiomir_json_zip.py, validators.py; tests/submission/test_vibiomir_export.py.

**Interfaces:**

- VibiomirExporter.export(predictions: Sequence[FinalPrediction], queries: Sequence[OfficialQuery], store: SourceStore, output_dir: Path) -> ExportResult
- ExportResult: json_path, zip_path, validation_report_path, json_sha256, zip_sha256.
- validate_predictions(rows: Sequence[SubmissionRow], queries: Sequence[OfficialQuery], store: SourceStore, expected_count: int) -> ValidationReport
- validate_zip(path: Path, queries: Sequence[OfficialQuery], store: SourceStore, expected_count: int) -> ValidationReport
- ValidationReport: valid, query_count, document_count, chunk_count, errors, snapshot_id.

- [ ] Tests: exact sparse IDs/int64, reject bool/string/float IDs, missing/extra/duplicate queries, foreign doc IDs, duplicate docs/chunks, missing lists, valid empty predictions.
- [ ] Test chunk exact source span; reject modified text, mismatched source version/hash, synthetic heading, orphan internal chunk ID. Chunk with valid official doc not in relevant_docs phải được chấp nhận.
- [ ] Test negative/bool chunk_order reject; one root JSON entry only; reject extra file/subdirectory/encrypted ZIP; export twice same runtime gives same SHA256.
- [ ] Run: pytest tests/submission/test_vibiomir_export.py -q. Expected: FAIL exporter/validation chưa có.
- [ ] Implement resolve nội bộ sang official IDs, snapshot-bound provenance check, deterministic JSON/ZIP và atomic writes. Không tự bổ sung missing query hoặc tự bỏ malformed chunks.
- [ ] Run tests. Expected: PASS; decompress ZIP và parse đúng JSON array.
- [ ] Commit: feat: export validated ViBioMIR JSON submissions.

