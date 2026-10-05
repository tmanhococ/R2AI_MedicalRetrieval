# Task 3 Report: HTML/PDF Extraction, Language and Quality

## 1. Summary of Implementation
Implemented document extraction and quality classification according to Sections 5.3 and 5.4 of the ViBioMIR Phase 1 implementation plan:
- **`src/ingestion/extractors.py`**:
  - `extract_document(doc_id, raw, mime_type, raw_sha256, config)` supporting HTML and PDF.
  - HTML extraction using Trafilatura structured XML parsing with fallback to `lxml.html` traversal.
  - Preserves paragraphs, headings, lists, tables, and Q&A (question + doctor's answer).
  - Normalizes and excludes non-content boilerplate: `<header>`, `<nav>`, `<footer>`, `<script>`, `<style>`, `<noscript>`.
  - PDF extraction via PyMuPDF (`pymupdf`), maintaining page and block reading order. Flags scanned PDFs with no text as `needs_ocr`.
  - Calculates exact Python character offsets on deterministic `\n\n`-joined `source_text`, verifying `source_text[block.start_char:block.end_char] == text`.
  - Computes `source_sha256` as `hashlib.sha256(source_text.encode("utf-8")).hexdigest()`.
- **`src/ingestion/quality.py`**:
  - `classify_quality(document, detector, config)` returning `QualityResult(state, language, language_score, flags)`.
  - `LanguageDetector` protocol and `FastTextLanguageDetector` wrapper with lazy loading and SHA-256 model checksum tracking.
  - Quality rules: checks for empty text, captcha signatures, soft-404 error pages, low language confidence (<0.70), unsupported languages (outside `vi`, `en`, `zh`), and short text flag.
  - `build_extraction_report(results)` aggregating outcomes into `ExtractionReport`.
- **Fixtures in `tests/fixtures/vibiomir/`**:
  - `article_medical.html`: article with headings, lists, tables, and medical terms (`HER2+`, `K+`, `TNF-α`, `0.10 U/L`).
  - `qa_medical.html`: medical Q&A preserving doctor's answer and question.
  - `captcha_page.html`: Cloudflare captcha challenge page.
  - `soft_404_page.html`: soft-404 error page.
  - `nav_only_page.html`: navigation-only page.
  - `chinese_article.html`: Chinese medical article.
- **Unit test suite in `tests/ingestion/test_extraction.py`**:
  - 16 unit tests covering exact block reconstruction, medical symbols and Vietnamese accents, QA preservation, Chinese text, PDF extraction, blank/scanned PDF `needs_ocr`, quality states, language confidence quarantining, and report aggregation using stubbed language detectors (no model downloads).

## 2. TDD Evidence

### RED Phase
- **Command**: `uv run pytest tests/ingestion/test_extraction.py -q`
- **Output**:
```
=================================== ERRORS ====================================
_____________ ERROR collecting tests/ingestion/test_extraction.py _____________
ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\ingestion\test_extraction.py'.
Traceback:
tests\ingestion\test_extraction.py:13: in <module>
    from src.ingestion.extractors import extract_document
E   ModuleNotFoundError: No module named 'src.ingestion.extractors'
=========================== short test summary info ===========================
ERROR tests/ingestion/test_extraction.py
!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
1 error in 0.66s
```

### GREEN Phase
- **Command**: `uv run pytest tests/ingestion/test_extraction.py -v`
- **Output**:
```
============================= test session starts =============================
platform win32 -- Python 3.12.10, pytest-8.2.0, pluggy-1.6.0
collecting ... collected 16 items

tests/ingestion/test_extraction.py::test_extract_html_article_reconstruction_and_medical_terms PASSED [  6%]
tests/ingestion/test_extraction.py::test_extract_html_qa_preserves_doctor_answer PASSED [ 12%]
tests/ingestion/test_chinese_article PASSED                             [ 18%]
tests/ingestion/test_extract_navigation_only_page PASSED                 [ 25%]
tests/ingestion/test_extract_pdf_preserves_reading_order_and_offsets PASSED [ 31%]
tests/ingestion/test_extract_scanned_pdf_flags_needs_ocr PASSED         [ 37%]
tests/ingestion/test_quality_ready_for_valid_article PASSED             [ 43%]
tests/ingestion/test_quality_ready_for_chinese_article PASSED           [ 50%]
tests/ingestion/test_quality_quarantine_for_captcha_page PASSED         [ 56%]
tests/ingestion/test_quality_quarantine_for_soft_404_page PASSED        [ 62%]
tests/ingestion/test_quality_quarantine_for_low_language_confidence PASSED [ 68%]
tests/ingestion/test_quality_quarantine_for_unsupported_language PASSED [ 75%]
tests/ingestion/test_quality_quarantine_for_needs_ocr PASSED            [ 81%]
tests/ingestion/test_quality_short_valid_medical_text_is_ready_with_flag PASSED [ 87%]
tests/ingestion/test_fasttext_wrapper_missing_file_error PASSED         [ 93%]
tests/ingestion/test_build_extraction_report PASSED                     [100%]

============================= 16 passed in 1.83s ==============================
```

## 3. Full Regression Suite Result
- **Command**: `uv run pytest -q`
- **Output**:
```
........................................................................ [ 30%]
........................................................................ [ 61%]
........................................................................ [ 91%]
....................                                                     [100%]
236 passed, 1 warning in 25.91s
```
- Baseline passed: 220
- Current passed: 236 (+16 new tests, 0 regressions)

## 4. Files Created / Modified
- Created: `src/ingestion/extractors.py`
- Created: `src/ingestion/quality.py`
- Modified: `src/ingestion/__init__.py`
- Created: `tests/fixtures/vibiomir/article_medical.html`
- Created: `tests/fixtures/vibiomir/qa_medical.html`
- Created: `tests/fixtures/vibiomir/captcha_page.html`
- Created: `tests/fixtures/vibiomir/soft_404_page.html`
- Created: `tests/fixtures/vibiomir/nav_only_page.html`
- Created: `tests/fixtures/vibiomir/chinese_article.html`
- Created: `tests/ingestion/test_extraction.py`

## 5. Git Commit
- **SHA**: `66ca27b`
- **Message**: `feat: extract source documents with quality metadata (Task 3)`

## 6. Self-Review Checklist
- [x] Invariant verified: `source_text[block.start_char:block.end_char] == text`
- [x] Medical symbols (`HER2+`, `K+`, `TNF-α`, `0.10 U/L`) and Vietnamese diacritics intact
- [x] Doctor answer in medical Q&A never dropped
- [x] Scanned PDF flagged `needs_ocr`
- [x] Captcha, soft-404, low language confidence quarantined
- [x] Short valid medical text flagged `short_text` but marked `ready` (not dropped)
- [x] Unit tests stub LanguageDetector without model download
- [x] 0 regressions across full test suite

## 7. Review Round 1 Fixes
- **Issue 1 (Header pruning)**: Refined `_clean_non_content_tags` in `src/ingestion/extractors.py` to only remove site-level boilerplate headers/footers outside `<article>` and `<main>` (or with `role="banner"` / `role="contentinfo"`). Semantic headers within articles (containing titles, lead paragraphs) are completely preserved.
- **Issue 2 (Soft-404 substring false-positives)**: Replaced substring search with compiled regex patterns `SOFT_404_PATTERNS` enforcing word and phrase boundaries (e.g. `\bkhông tìm thấy trang(\s*(này|web|bạn yêu cầu|yêu cầu|\?|\.|$)|[.,!?:;]|$)`). Phrases such as `"không tìm thấy trang thiết bị y tế"` are not flagged as soft-404. Scoped soft-404 body matching to titles, headings, and short error bodies (< 500 chars).
- **Added Tests in `tests/ingestion/test_extraction.py`**:
  - `test_extract_html_article_with_header_inside_article`: verifies semantic `<header>` inside `<article>` is preserved while top-level site header is stripped.
  - `test_quality_legitimate_medical_article_with_trang_thiet_bi_not_quarantined`: verifies legitimate medical article containing `"không tìm thấy trang thiết bị y tế"` remains `ready` without `soft_404` flag.
- **Commit**: `b48af26` (`fix: preserve semantic article headers and refine soft-404 regexes (Task 3 review)`)
- **Regression suite**: 238 passed (18 in `test_extraction.py`), 0 failures, 0 regressions.

