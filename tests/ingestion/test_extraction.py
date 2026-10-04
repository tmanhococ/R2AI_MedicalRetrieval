import hashlib
import io
from pathlib import Path
import pytest
import pymupdf

from src.data.vibiomir_contracts import (
    ExtractedBlock,
    ExtractedDocument,
    ExtractionConfig,
    ExtractionReport,
)
from src.ingestion.extractors import extract_document
from src.ingestion.quality import (
    LanguageDetector,
    QualityResult,
    classify_quality,
    FastTextLanguageDetector,
    build_extraction_report,
)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "vibiomir"


class StubLanguageDetector:
    """Mock detector for fast deterministic testing without downloading lid.176.ftz."""

    def __init__(self, language: str = "vi", score: float = 0.95):
        self.language = language
        self.score = score

    def detect(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return ("und", 0.0)
        return (self.language, self.score)


class PatternLanguageDetector:
    """Mock detector that infers language based on text patterns."""

    def detect(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return ("und", 0.0)
        # Check Chinese characters
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return ("zh", 0.98)
        # Check Vietnamese specific diacritics
        vi_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
        if any(c in vi_chars for c in text.lower()):
            return ("vi", 0.96)
        return ("en", 0.90)


def create_synthetic_pdf(paragraphs: list[str]) -> bytes:
    """Create in-memory PDF using pymupdf.Story to guarantee unicode diacritics support."""
    buf = io.BytesIO()
    writer = pymupdf.DocumentWriter(buf)
    html_content = "".join(f"<p>{p}</p>" for p in paragraphs)
    story = pymupdf.Story(html_content)
    while True:
        device = writer.begin_page(pymupdf.Rect(0, 0, 595, 842))
        more, _ = story.place(pymupdf.Rect(50, 50, 500, 750))
        story.draw(device)
        writer.end_page()
        if not more:
            break
    writer.close()
    return buf.getvalue()


def create_scanned_blank_pdf() -> bytes:
    """Create a blank PDF simulating a scanned document with no extractable text."""
    doc = pymupdf.open()
    doc.new_page()
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# =========================================================================
# 1. HTML Extraction Tests
# =========================================================================

def test_extract_html_article_reconstruction_and_medical_terms():
    """Verify HTML extraction preserves structure, medical symbols, and exact character offsets."""
    raw = (FIXTURES_DIR / "article_medical.html").read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    config = ExtractionConfig()

    doc = extract_document(
        doc_id=101,
        raw=raw,
        mime_type="text/html",
        raw_sha256=raw_sha256,
        config=config,
    )

    assert isinstance(doc, ExtractedDocument)
    assert doc.doc_id == 101
    assert doc.raw_sha256 == raw_sha256
    assert doc.source_sha256 == hashlib.sha256(doc.source_text.encode("utf-8")).hexdigest()
    assert len(doc.blocks) > 0

    # Invariant: source_text[block.start_char:block.end_char] == block_text
    reconstructed_blocks = []
    for block in doc.blocks:
        assert isinstance(block, ExtractedBlock)
        block_text = doc.source_text[block.start_char : block.end_char]
        assert len(block_text) > 0
        reconstructed_blocks.append(block_text)

    # Invariant: source_text is formed deterministically by joining block texts with \n\n
    assert doc.source_text == "\n\n".join(reconstructed_blocks)

    # Invariant: medical symbols and terms are strictly preserved intact
    assert "HER2+" in doc.source_text
    assert "K+" in doc.source_text
    assert "TNF-α" in doc.source_text
    assert "0.10 U/L" in doc.source_text

    # Invariant: Vietnamese diacritics are completely intact
    assert "Xét nghiệm HER2 và TNF-α trong điều trị ung thư" in doc.source_text
    assert "Bệnh nhân có kết quả HER2+ dương tính" in doc.source_text

    # Invariant: Table and list content are present
    assert "Giá trị tham chiếu" in doc.source_text
    assert "Chỉ số 1: Tải lượng" in doc.source_text or "Chỉ số TNF-α huyết tương" in doc.source_text


def test_extract_html_qa_preserves_doctor_answer():
    """Verify that medical Q&A pages preserve both question and doctor answer."""
    raw = (FIXTURES_DIR / "qa_medical.html").read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    config = ExtractionConfig()

    doc = extract_document(
        doc_id=202,
        raw=raw,
        mime_type="text/html",
        raw_sha256=raw_sha256,
        config=config,
    )

    # Ensure question text is present
    assert "đau tức ngực trái lan lên vai" in doc.source_text
    # CRITICAL INVARIANT: Doctor's answer is NEVER dropped
    assert "Bác sĩ Chuyên khoa Tim mạch trả lời" in doc.source_text or "Chào bạn, đau thắt ngực khi gắng sức" in doc.source_text
    assert "Troponin" in doc.source_text
    assert "0.10 U/L" in doc.source_text

    # Verify offset reconstruction
    for block in doc.blocks:
        assert doc.source_text[block.start_char : block.end_char] != ""


def test_extract_chinese_article():
    """Verify Chinese medical text extraction and offset reconstruction."""
    raw = (FIXTURES_DIR / "chinese_article.html").read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    config = ExtractionConfig()

    doc = extract_document(
        doc_id=303,
        raw=raw,
        mime_type="text/html",
        raw_sha256=raw_sha256,
        config=config,
    )

    assert "HER2阳性乳腺癌患者的靶向治疗指南" in doc.source_text
    assert "曲妥珠单抗" in doc.source_text
    assert "K+" in doc.source_text

    for block in doc.blocks:
        assert doc.source_text[block.start_char : block.end_char] != ""


def test_extract_navigation_only_page():
    """Verify that navigation-only page yields empty or quarantined text."""
    raw = (FIXTURES_DIR / "nav_only_page.html").read_bytes()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    config = ExtractionConfig()

    doc = extract_document(
        doc_id=404,
        raw=raw,
        mime_type="text/html",
        raw_sha256=raw_sha256,
        config=config,
    )

    # Nav menus and footers are excluded, resulting in empty body
    assert doc.source_text.strip() == "" or "empty_text" in doc.quality_flags


# =========================================================================
# 2. PDF Extraction Tests
# =========================================================================

def test_extract_pdf_preserves_reading_order_and_offsets():
    """Verify PDF extraction preserves block order, medical symbols, and offsets."""
    paragraphs = [
        "Phác đồ điều trị ung thư vú thể HER2+ giai đoạn tiến xa.",
        "Theo dõi nồng độ điện giải đồ: K+ máu và cytokine TNF-α.",
        "Chỉ số xét nghiệm đạt 0.10 U/L sau 4 chu kỳ hóa trị.",
    ]
    raw = create_synthetic_pdf(paragraphs)
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    config = ExtractionConfig()

    doc = extract_document(
        doc_id=505,
        raw=raw,
        mime_type="application/pdf",
        raw_sha256=raw_sha256,
        config=config,
    )

    assert doc.doc_id == 505
    assert len(doc.blocks) >= 3
    # Check terms
    assert "HER2+" in doc.source_text
    assert "K+" in doc.source_text
    assert "TNF-α" in doc.source_text
    assert "0.10 U/L" in doc.source_text

    # Verify offset integrity
    for block in doc.blocks:
        slice_text = doc.source_text[block.start_char : block.end_char]
        assert len(slice_text) > 0
        assert slice_text.strip() != ""


def test_extract_scanned_pdf_flags_needs_ocr():
    """Verify that a scanned PDF with no text blocks is flagged as needs_ocr."""
    raw = create_scanned_blank_pdf()
    raw_sha256 = hashlib.sha256(raw).hexdigest()
    config = ExtractionConfig()

    doc = extract_document(
        doc_id=606,
        raw=raw,
        mime_type="application/pdf",
        raw_sha256=raw_sha256,
        config=config,
    )

    assert doc.source_text.strip() == ""
    assert "needs_ocr" in doc.quality_flags


# =========================================================================
# 3. Quality Classification Tests
# =========================================================================

def test_quality_ready_for_valid_article():
    """Verify valid article produces state='ready' with high confidence."""
    raw = (FIXTURES_DIR / "article_medical.html").read_bytes()
    doc = extract_document(
        doc_id=1,
        raw=raw,
        mime_type="text/html",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        config=ExtractionConfig(),
    )
    detector = PatternLanguageDetector()
    qr = classify_quality(doc, detector)

    assert isinstance(qr, QualityResult)
    assert qr.state == "ready"
    assert qr.language == "vi"
    assert qr.language_score >= 0.70
    assert "captcha" not in qr.flags
    assert "soft_404" not in qr.flags


def test_quality_ready_for_chinese_article():
    """Verify Chinese article is accepted and marked ready with language='zh'."""
    raw = (FIXTURES_DIR / "chinese_article.html").read_bytes()
    doc = extract_document(
        doc_id=2,
        raw=raw,
        mime_type="text/html",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        config=ExtractionConfig(),
    )
    detector = PatternLanguageDetector()
    qr = classify_quality(doc, detector)

    assert qr.state == "ready"
    assert qr.language == "zh"
    assert qr.language_score >= 0.70


def test_quality_quarantine_for_captcha_page():
    """Verify captcha / cloudflare challenge pages are quarantined."""
    raw = (FIXTURES_DIR / "captcha_page.html").read_bytes()
    doc = extract_document(
        doc_id=3,
        raw=raw,
        mime_type="text/html",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        config=ExtractionConfig(),
    )
    detector = StubLanguageDetector(language="en", score=0.99)
    qr = classify_quality(doc, detector)

    assert qr.state == "quarantined"
    assert "captcha" in qr.flags


def test_quality_quarantine_for_soft_404_page():
    """Verify soft-404 error pages are quarantined."""
    raw = (FIXTURES_DIR / "soft_404_page.html").read_bytes()
    doc = extract_document(
        doc_id=4,
        raw=raw,
        mime_type="text/html",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        config=ExtractionConfig(),
    )
    detector = StubLanguageDetector(language="vi", score=0.95)
    qr = classify_quality(doc, detector)

    assert qr.state == "quarantined"
    assert "soft_404" in qr.flags


def test_quality_quarantine_for_low_language_confidence():
    """Verify low language confidence (< 0.70) causes quarantine."""
    raw = (FIXTURES_DIR / "article_medical.html").read_bytes()
    doc = extract_document(
        doc_id=5,
        raw=raw,
        mime_type="text/html",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        config=ExtractionConfig(),
    )
    # Stub detector with low confidence 0.55
    detector = StubLanguageDetector(language="vi", score=0.55)
    qr = classify_quality(doc, detector)

    assert qr.state == "quarantined"
    assert "low_language_confidence" in qr.flags


def test_quality_quarantine_for_unsupported_language():
    """Verify unsupported language (e.g. French, German) causes quarantine."""
    raw = (FIXTURES_DIR / "article_medical.html").read_bytes()
    doc = extract_document(
        doc_id=6,
        raw=raw,
        mime_type="text/html",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        config=ExtractionConfig(),
    )
    detector = StubLanguageDetector(language="fr", score=0.95)
    qr = classify_quality(doc, detector)

    assert qr.state == "quarantined"
    assert "unsupported_language" in qr.flags


def test_quality_quarantine_for_needs_ocr():
    """Verify scanned blank PDF with needs_ocr is quarantined."""
    raw = create_scanned_blank_pdf()
    doc = extract_document(
        doc_id=7,
        raw=raw,
        mime_type="application/pdf",
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        config=ExtractionConfig(),
    )
    detector = StubLanguageDetector()
    qr = classify_quality(doc, detector)

    assert qr.state == "quarantined"
    assert "needs_ocr" in qr.flags


def test_quality_short_valid_medical_text_is_ready_with_flag():
    """Verify short valid text has short_text flag but is NOT quarantined."""
    short_html = b"<html><body><p>Ch\xc4\xa9 \xc4\x91\xe1\xbb\x8bnh x\xc3\xa9t nghi\xe1\xbb\x87m HER2 cho b\xe1\xbb\x87nh nh\xc3\xa2n ung th\xc6\xb0 v\xc3\xba.</p></body></html>"
    doc = extract_document(
        doc_id=8,
        raw=short_html,
        mime_type="text/html",
        raw_sha256=hashlib.sha256(short_html).hexdigest(),
        config=ExtractionConfig(),
    )
    detector = StubLanguageDetector(language="vi", score=0.92)
    qr = classify_quality(doc, detector)

    # Document should be ready despite short length; not excluded purely on brevity
    assert qr.state == "ready"
    assert qr.language == "vi"
    assert "short_text" in qr.flags
    assert "low_language_confidence" not in qr.flags


def test_fasttext_wrapper_missing_file_error():
    """Verify FastTextLanguageDetector raises FileNotFoundError when model path is missing."""
    detector = FastTextLanguageDetector(model_path="nonexistent/model.ftz")
    with pytest.raises(FileNotFoundError):
        _ = detector.model_sha256


def test_build_extraction_report():
    """Verify build_extraction_report aggregates quality outcomes correctly."""
    results = [
        QualityResult(state="ready", language="vi", language_score=0.95),
        QualityResult(state="ready", language="en", language_score=0.92),
        QualityResult(state="quarantined", language="und", language_score=0.0, flags=("captcha",)),
        QualityResult(state="quarantined", language="fr", language_score=0.85, flags=("unsupported_language",)),
        QualityResult(state="quarantined", language="vi", language_score=0.50, flags=("low_language_confidence",)),
        QualityResult(state="extraction_failed", language="und", language_score=0.0, flags=("extraction_failed",)),
    ]
    report = build_extraction_report(results)
    assert isinstance(report, ExtractionReport)
    assert report.attempted == 6
    assert report.ready == 2
    assert report.quarantined == 3
    assert report.failed == 1
    assert report.flags_by_reason.get("captcha") == 1
    assert report.flags_by_reason.get("unsupported_language") == 1
    assert report.flags_by_reason.get("low_language_confidence") == 1
