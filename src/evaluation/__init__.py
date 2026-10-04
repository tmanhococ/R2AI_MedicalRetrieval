"""Evaluation package for Multilingual Medical Retrieval MVP."""

from src.evaluation.calibrator import CalibrationState, CutoffCalibrator
from src.evaluation.dev_records import load_dev_records
from src.evaluation.error_analysis import (
    build_error_breakdown,
    generate_error_analysis_report,
    generate_error_breakdown_summary,
)
from src.evaluation.metrics import (
    calculate_f2,
    calculate_precision_recall_fbeta,
    evaluate_predictions,
    evaluate_query,
)
from src.evaluation.vibiomir_metrics import (
    EMPTY_BOTH_ZERO_PROFILE,
    PROVISIONAL_SPEC_V1_PROFILE,
    evaluate_dataset,
    evaluate_query as evaluate_vibiomir_query,
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
    "evaluate_vibiomir_query",
    "evaluate_dataset",
    "CutoffCalibrator",
    "CalibrationState",
    "generate_error_analysis_report",
    "build_error_breakdown",
    "generate_error_breakdown_summary",
    "load_dev_records",
    "normalize_eval_text",
    "tokenize_eval_text",
    "lcs_length",
    "overlap_matches",
    "TextMatchProfile",
    "TextOverlapEvaluator",
    "PROVISIONAL_SPEC_V1",
    "PROVISIONAL_SPEC_V1_PROFILE",
    "EMPTY_BOTH_ZERO_PROFILE",
    "clear_eval_text_caches",
]
