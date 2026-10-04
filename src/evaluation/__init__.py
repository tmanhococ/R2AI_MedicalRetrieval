"""Evaluation package for Multilingual Medical Retrieval MVP."""

from src.evaluation.calibrator import CalibrationState, CutoffCalibrator
from src.evaluation.error_analysis import generate_error_analysis_report
from src.evaluation.metrics import (
    calculate_f2,
    calculate_precision_recall_fbeta,
    evaluate_predictions,
    evaluate_query,
)
from src.evaluation.vibiomir_text import (
    PROVISIONAL_SPEC_V1,
    TextMatchProfile,
    TextOverlapEvaluator,
    clear_eval_text_caches,
    lcs_length,
    normalize_eval_text,
    overlap_matches,
    tokenize_eval_text,
)

__all__ = [
    "calculate_f2",
    "calculate_precision_recall_fbeta",
    "evaluate_query",
    "evaluate_predictions",
    "CutoffCalibrator",
    "CalibrationState",
    "generate_error_analysis_report",
    "normalize_eval_text",
    "tokenize_eval_text",
    "lcs_length",
    "overlap_matches",
    "TextMatchProfile",
    "TextOverlapEvaluator",
    "PROVISIONAL_SPEC_V1",
    "clear_eval_text_caches",
]
