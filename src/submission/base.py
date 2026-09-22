from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence, runtime_checkable

from src.data.schemas import FinalPrediction

if TYPE_CHECKING:
    from src.submission.competition_csv import CompetitionCsvExporter
    from src.submission.internal_jsonl import InternalJsonlExporter


@runtime_checkable
class SubmissionExporter(Protocol):
    """Protocol for submission exporters."""

    def export(
        self,
        predictions: Sequence[FinalPrediction],
        output_path: Path | str,
    ) -> Path:
        """Export predictions to the target file path.

        Args:
            predictions: Sequence of FinalPrediction objects to export.
            output_path: Target destination path.

        Returns:
            Path object pointing to the written output file.
        """
        ...


def create_exporter(exporter_type: str, **kwargs: Any) -> SubmissionExporter:
    """Factory function to instantiate a submission exporter.

    Args:
        exporter_type: Exporter type ('internal_jsonl' or 'competition_csv').
        **kwargs: Additional parameters passed to the exporter constructor.

    Returns:
        Instance conforming to SubmissionExporter protocol.

    Raises:
        ValueError: If exporter_type is not recognized.
    """
    if exporter_type == "internal_jsonl":
        from src.submission.internal_jsonl import InternalJsonlExporter

        return InternalJsonlExporter(**kwargs)
    elif exporter_type == "competition_csv":
        from src.submission.competition_csv import CompetitionCsvExporter

        return CompetitionCsvExporter(**kwargs)
    else:
        raise ValueError(
            f"Unknown exporter type: '{exporter_type}'. Supported types: 'internal_jsonl', 'competition_csv'"
        )
