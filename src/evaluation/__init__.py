"""Evaluation package for Multilingual Medical Retrieval MVP."""

from src.evaluation.calibrator import CalibrationState, CutoffCalibrator
from src.evaluation.error_analysis import generate_error_analysis_report
from src.evaluation.metrics import (
    calculate_f2,
    calculate_precision_recall_fbeta,
    evaluate_predictions,
    evaluate_query,
)

__all__ = [
    "calculate_f2",
    "calculate_precision_recall_fbeta",
    "evaluate_query",
    "evaluate_predictions",
    "CutoffCalibrator",
    "CalibrationState",
    "generate_error_analysis_report",
]
