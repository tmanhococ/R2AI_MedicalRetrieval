"""Command-line interface and entrypoints for Multilingual Medical Retrieval MVP."""

from src.cli.calibrate import run_calibrate
from src.cli.evaluate import run_evaluate
from src.cli.index import run_index
from src.cli.predict import run_predict

__all__ = [
    "run_index",
    "run_evaluate",
    "run_calibrate",
    "run_predict",
]
