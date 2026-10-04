from collections import Counter
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Protocol, Sequence, runtime_checkable

from src.data.vibiomir_contracts import ExtractedDocument, ExtractionConfig, ExtractionReport


@runtime_checkable
class LanguageDetector(Protocol):
    """Protocol for language detection engines."""

    def detect(self, text: str) -> tuple[str, float]:
        """Detect language of text. Returns (language_code, confidence_score)."""
        ...


@dataclass(frozen=True)
class QualityResult:
    """Document quality evaluation verdict and language metadata."""
    state: str
    language: str
    language_score: float
    flags: tuple[str, ...] = ()


CAPTCHA_SIGNATURES = [
    "cf-turnstile",
    "g-recaptcha",
    "hcaptcha",
    "ray id:",
    "verify you are human",
    "please verify that you are a human",
    "security check to access",
    "checking your browser before accessing",
    "bot detection",
    "attention required! | cloudflare",
    "xác minh bạn là người",
    "vui lòng xác minh không phải người máy",
    "bảo vệ bởi cloudflare",
    "protected by cloudflare",
    "ddos protection by cloudflare",
]

SOFT_404_PATTERNS = [
    re.compile(r"\b404\s+(?:not\s+found|error)\b", re.IGNORECASE),
    re.compile(r"\berror\s+404\b", re.IGNORECASE),
    re.compile(r"\b404\s*-\s*không\s+tìm\s+thấy\b", re.IGNORECASE),
    re.compile(r"\b404\s+không\s+tìm\s+thấy\b", re.IGNORECASE),
    re.compile(r"\bpage\s+not\s+found\b", re.IGNORECASE),
    re.compile(r"\bthe\s+page\s+you\s+are\s+looking\s+for\s+does\s+not\s+exist\b", re.IGNORECASE),
    re.compile(r"\bkhông\s+tìm\s+thấy\s+trang(\s*(này|web|bạn yêu cầu|yêu cầu|\?|\.|$)|[.,!?:;]|$)", re.IGNORECASE),
    re.compile(r"\btrang\s+(?:này\s+)?không\s+tìm\s+thấy\b", re.IGNORECASE),
    re.compile(r"\btrang\s+bạn\s+tìm\s+kiếm\s+không\s+tồn\s+tại\b", re.IGNORECASE),
    re.compile(r"\bnội\s+dung\s+không\s+tồn\s+tại\b", re.IGNORECASE),
    re.compile(r"\bbài\s+viết\s+không\s+tồn\s+tại\b", re.IGNORECASE),
]


class FastTextLanguageDetector:
    """Wrapper around fastText lid.176.ftz with lazy loading and checksum tracking."""

    def __init__(self, model_path: Path | str = "artifacts/models/lid.176.ftz"):
        self.model_path = Path(model_path)
        self._model = None
        self._model_sha256: str | None = None

    @property
    def model_sha256(self) -> str:
        if self._model_sha256 is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Language model file not found at {self.model_path}")
            self._model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        return self._model_sha256

    def _load_model(self):
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Language model file not found at {self.model_path}")
            try:
                import fasttext
            except ImportError as e:
                raise ImportError(
                    "fasttext is required for FastTextLanguageDetector. "
                    "Install fasttext or inject a mock detector in tests."
                ) from e
            # Silence fasttext warnings if possible
            fasttext.FastText.eprint = lambda *args, **kwargs: None
            self._model = fasttext.load_model(str(self.model_path))
        return self._model

    def detect(self, text: str) -> tuple[str, float]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return ("und", 0.0)

        model = self._load_model()
        labels, scores = model.predict(cleaned, k=1)
        if labels and len(labels) > 0:
            lang = str(labels[0]).replace("__label__", "").strip()
            score = float(scores[0])
            return (lang, score)
        return ("und", 0.0)


def classify_quality(
    document: ExtractedDocument,
    detector: LanguageDetector,
    config: ExtractionConfig | None = None,
) -> QualityResult:
    """Classify document quality, language, and provenance health according to Section 5.3."""
    if config is None:
        config = ExtractionConfig()

    flags: list[str] = list(document.quality_flags)

    # 1. Check if extraction itself failed or is unsupported
    if "extraction_failed" in flags:
        return QualityResult(
            state="extraction_failed",
            language="und",
            language_score=0.0,
            flags=tuple(sorted(set(flags))),
        )
    if "unsupported_format" in flags or "unsupported_mime_type" in flags:
        return QualityResult(
            state="unsupported",
            language="und",
            language_score=0.0,
            flags=tuple(sorted(set(flags))),
        )

    # 2. Check for empty text
    clean_text = " ".join(document.source_text.split())
    if not clean_text:
        if "needs_ocr" not in flags:
            flags.append("empty_text")
        return QualityResult(
            state="quarantined",
            language="und",
            language_score=0.0,
            flags=tuple(sorted(set(flags))),
        )

    # 3. Check for OCR needed
    if "needs_ocr" in flags:
        return QualityResult(
            state="quarantined",
            language="und",
            language_score=0.0,
            flags=tuple(sorted(set(flags))),
        )

    is_quarantined = False
    text_lower = clean_text.lower()
    title_lower = (document.title or "").lower()
    combined_lower = f"{title_lower} {text_lower}"

    # 4. Check for captcha signatures
    for sig in CAPTCHA_SIGNATURES:
        if sig in combined_lower:
            flags.append("captcha")
            is_quarantined = True
            break

    # 5. Check for soft-404 signatures
    headings_text = " ".join(
        document.source_text[b.start_char : b.end_char]
        for b in document.blocks
        if b.kind == "heading"
    ).lower()

    is_soft_404 = False
    for pat in SOFT_404_PATTERNS:
        if pat.search(title_lower) or pat.search(headings_text):
            is_soft_404 = True
            break
        if len(clean_text) < 500 and pat.search(text_lower):
            is_soft_404 = True
            break

    if is_soft_404:
        flags.append("soft_404")
        is_quarantined = True

    # 6. Check for short text (flag only, per plan section 5.3: not quarantined purely on brevity)
    if len(clean_text) < 150:
        flags.append("short_text")

    # 7. Language detection on body text
    lang, score = detector.detect(clean_text)

    if lang in ("und", "unknown", ""):
        flags.append("unknown_language")
        is_quarantined = True
    elif lang not in config.accepted_languages:
        flags.append("unsupported_language")
        is_quarantined = True

    if score < config.language_confidence_min:
        flags.append("low_language_confidence")
        is_quarantined = True

    state = "quarantined" if is_quarantined else "ready"

    return QualityResult(
        state=state,
        language=lang,
        language_score=score,
        flags=tuple(sorted(set(flags))),
    )


def build_extraction_report(results: Sequence[QualityResult]) -> ExtractionReport:
    """Aggregate quality results into an ExtractionReport."""
    attempted = len(results)
    ready = sum(1 for r in results if r.state == "ready")
    quarantined = sum(1 for r in results if r.state == "quarantined")
    failed = sum(1 for r in results if r.state in ("extraction_failed", "unsupported"))

    flags_counter = Counter()
    for r in results:
        for f in r.flags:
            flags_counter[f] += 1

    return ExtractionReport(
        attempted=attempted,
        ready=ready,
        quarantined=quarantined,
        failed=failed,
        flags_by_reason=dict(flags_counter),
    )
