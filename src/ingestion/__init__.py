"""Ingestion module for ViBioMIR crawl, extraction, and validation."""

from src.ingestion.extractors import extract_document
from src.ingestion.quality import (
    FastTextLanguageDetector,
    LanguageDetector,
    QualityResult,
    build_extraction_report,
    classify_quality,
)

__all__ = [
    "extract_document",
    "classify_quality",
    "QualityResult",
    "LanguageDetector",
    "FastTextLanguageDetector",
    "build_extraction_report",
]
