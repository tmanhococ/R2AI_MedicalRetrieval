### Task 3: HTML/PDF extraction, language và quality

**Files:** Create ingestion/extractors.py, quality.py; fixtures HTML/PDF tại tests/fixtures/vibiomir/; tests/ingestion/test_extraction.py.

**Interfaces:**

- extract_document(doc_id: int, raw: bytes, mime_type: str, raw_sha256: str, config: ExtractionConfig) -> ExtractedDocument
- classify_quality(document: ExtractedDocument, detector: LanguageDetector) -> QualityResult
- QualityResult: state, language, language_score, flags.
- LanguageDetector.detect(text: str) -> tuple[str, float].

- [ ] Viết fixtures article with headings/list/table, QA with question+answer, HER2+/K+/TNF-α/0.10 U/L, Chinese text, PDF text, soft-404/captcha/navigation-only, short valid medical text. Assert source blocks/offsets reconstruct source_text và phần trả lời không bị bỏ.
- [ ] Run: pytest tests/ingestion/test_extraction.py -q. Expected: FAIL extraction contract.
- [ ] Implement Trafilatura structured parsing; create stable source text; dispatch PDF backend; inject detector trong tests. Không tải language model trong unit tests.
- [ ] Implement quality rules theo mục 5; language unsupported/low confidence được quarantine, không forced-map vào vi.
- [ ] Review thủ công output ít nhất 30 trang pilot thuộc nhiều domains/ngôn ngữ; nếu QA thiếu answer, thêm domain parser với fixture tái hiện.
- [ ] Run tests. Expected: PASS và report các nguồn cần OCR/JS/unknown language.
- [ ] Commit: feat: extract source documents with quality metadata.

