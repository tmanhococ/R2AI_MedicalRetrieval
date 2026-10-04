from __future__ import annotations

from dataclasses import dataclass
import logging
import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.schemas import QueryRecord

logger = logging.getLogger(__name__)


CLINICAL_BOILERPLATE_PATTERNS: list[str] = [
    # Patient demographics boilerplate (e.g. "Bệnh nhân nam 54 tuổi", "BN nữ", "Bệnh nhân")
    r"\b(?:bệnh nhân|bn)\s+(?:nam|nữ)(?:\s*,?\s*\d+\s*tuổi)?\b",
    r"\b(?:bệnh nhân|bn)(?:\s*,?\s*\d+\s*tuổi)?\b",
    r"\bnăm nay\s+\d+\s*tuổi\b",
    # Admission reason boilerplate
    r"\b(?:vào viện|nhập viện|đến khám|khám)\s+(?:vì lý do|do)\b",
    # History boilerplate
    r"\btiền sử\s+ghi nhận\b",
    r"\btiền sử\s+(?:bệnh lý|bệnh tật|bản thân)\b",
    # Clinical examination and indications boilerplate
    r"\b(?:sau đó\s+)?(?:được\s+)?chỉ định(?:\s+thực hiện)?\b",
    r"\b(?:qua\s+)?(?:thăm khám|khám lâm sàng)\s+ghi nhận\b",
    r"\bkết quả\s+(?:thăm khám|xét nghiệm)\s+ghi nhận\b",
    r"\bhiện tại\s+ghi nhận\b",
    r"\btình trạng\s+ghi nhận\b",
    # Conversational prefixes
    r"\b(?:bác sĩ\s+)?cho\s+(?:em|tôi)\s+hỏi\b",
    r"\bxin hỏi bác sĩ\b",
    r"\bbác sĩ cho hỏi\b",
]


@dataclass(frozen=True)
class ProcessedQuery:
    query_id: str
    original_text: str
    normalized_text: str
    bm25_text: str
    dense_text: str
    is_clinical_case: bool
    word_count: int
    bm25_original_text: str = ""
    dense_query_overflow: bool = False

    @property
    def condensed_vi(self) -> str | None:
        return self.bm25_text if self.is_clinical_case else None


@dataclass(frozen=True)
class QueryVariants:
    vi: str
    vi_condensed: str | None = None
    en: str | None = None
    zh: str | None = None


def normalize_vi_query(text: str) -> str:
    """Normalize Vietnamese medical query text while preserving clinical entities.

    Operations:
    - Unicode NFC normalization.
    - Zero-width character removal.
    - Consecutive whitespace collapsing and trimming.
    - Preserves case, abbreviations (HA, ĐTĐ, THA, COPD, etc.), units (mg, ml, mmol/L, mmHg),
      diagnostic markers ((+), (-)), and numbers without destructive stemming or lowercasing.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFC", text)
    # Remove zero-width spaces and BOMs
    normalized = normalized.replace("\u200b", "").replace("\ufeff", "")
    # Collapse multiple whitespace characters into a single space
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def build_query_variants(
    processed: ProcessedQuery,
    en: str | None = None,
    zh: str | None = None,
) -> QueryVariants:
    """Construct QueryVariants from a ProcessedQuery and optional translations."""
    return QueryVariants(
        vi=processed.normalized_text,
        vi_condensed=processed.condensed_vi,
        en=en,
        zh=zh,
    )


class AdaptiveQueryRouter:
    """Adaptive query processor routing short queries and long clinical cases.

    Short queries (<= length_threshold) keep uniform text across BM25 and Dense streams.
    Long queries (> length_threshold) strip clinical boilerplate for BM25 and retain
    semantic context (or apply head-tail truncation) for Dense retrieval.
    """

    def __init__(
        self,
        length_threshold: int = 50,
        strip_clinical_boilerplate: bool = True,
        dense_strategy: str = "head_tail",
        max_dense_tokens: int = 512,
    ) -> None:
        if dense_strategy not in ("head_tail", "full"):
            raise ValueError(
                f"Unsupported dense_strategy '{dense_strategy}'. Must be 'head_tail' or 'full'."
            )
        self.length_threshold = length_threshold
        self.strip_clinical_boilerplate = strip_clinical_boilerplate
        self.dense_strategy = dense_strategy
        self.max_dense_tokens = max_dense_tokens

    def _strip_clinical_boilerplate(self, text: str) -> str:
        cleaned = text
        for pattern in CLINICAL_BOILERPLATE_PATTERNS:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        # Clean up orphan punctuation: e.g. ", , " -> ", "
        cleaned = re.sub(r"\s*,\s*(?:,\s*)+", ", ", cleaned)
        # Strip leading/trailing punctuation left behind
        cleaned = re.sub(r"^[\s,;.:\-–—]+", "", cleaned)
        cleaned = re.sub(r"[\s,;.:\-–—]+$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned if cleaned else text

    def route(self, query: QueryRecord | str, query_id: str | None = None) -> ProcessedQuery:
        """Process and route a query for multi-stream retrieval.

        Accepts either a QueryRecord schema instance or raw string text.
        """
        # Dynamic check for QueryRecord to avoid hard import issues while retaining typing
        if hasattr(query, "query_id") and hasattr(query, "text_vi"):
            qid = query.query_id
            raw_text = query.text_vi
        elif isinstance(query, str):
            qid = query_id or "q0"
            raw_text = query
        else:
            raise TypeError(f"Expected QueryRecord or str, got {type(query).__name__}")

        normalized = normalize_vi_query(raw_text)
        words = normalized.split()
        word_count = len(words)
        is_clinical_case = word_count > self.length_threshold

        if not is_clinical_case:
            bm25_text = normalized
            dense_text = normalized
        else:
            # BM25 branch: strip clinical boilerplate if enabled
            if self.strip_clinical_boilerplate:
                bm25_text = self._strip_clinical_boilerplate(normalized)
            else:
                bm25_text = normalized

            # Dense branch: preserve context or head-tail truncation if > max_dense_tokens
            if word_count <= self.max_dense_tokens:
                dense_text = normalized
            else:
                if self.dense_strategy == "head_tail":
                    head_len = min(128, self.max_dense_tokens // 4)
                    tail_len = self.max_dense_tokens - head_len
                    dense_text = " ".join(words[:head_len] + words[-tail_len:])
                elif self.dense_strategy == "full":
                    dense_text = " ".join(words[: self.max_dense_tokens])
                else:
                    dense_text = normalized

        dense_overflow = word_count > self.max_dense_tokens
        if dense_overflow:
            logger.warning(
                "dense_query_overflow: query %s tokens %d > %d (strategy=%s)",
                qid,
                word_count,
                self.max_dense_tokens,
                self.dense_strategy,
            )

        return ProcessedQuery(
            query_id=qid,
            original_text=raw_text,
            normalized_text=normalized,
            bm25_text=bm25_text,
            dense_text=dense_text,
            is_clinical_case=is_clinical_case,
            word_count=word_count,
            bm25_original_text=normalized,
            dense_query_overflow=dense_overflow,
        )

