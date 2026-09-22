from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from src.data.schemas import FinalPrediction


class CompetitionCsvExporter:
    """Exports predictions to competition CSV format.

    Columns:
        - query_id: Unique query identifier
        - document_ids: Separator-delimited list of document IDs
        - chunk_ids: Separator-delimited list of chunk IDs
    """

    def __init__(
        self,
        separator: str = " ",
        delimiter: str = ",",
        id_separator: str | None = None,
    ) -> None:
        """Initialize CompetitionCsvExporter.

        Args:
            separator: Delimiter between IDs within a field (default: space " ").
            delimiter: CSV field delimiter (default: comma ",").
            id_separator: Alias for separator if provided.
        """
        self.separator = id_separator if id_separator is not None else separator
        self.delimiter = delimiter

    def export(
        self,
        predictions: Sequence[FinalPrediction],
        output_path: Path | str,
    ) -> Path:
        """Export predictions to a CSV file.

        Args:
            predictions: Sequence of FinalPrediction records.
            output_path: Target path to output file.

        Returns:
            Resolved Path object.
        """
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        sorted_preds = sorted(predictions, key=lambda p: p.query_id)
        fieldnames = ["query_id", "document_ids", "chunk_ids"]

        with open(target_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=self.delimiter)
            writer.writeheader()
            for pred in sorted_preds:
                writer.writerow(
                    {
                        "query_id": pred.query_id,
                        "document_ids": self.separator.join(pred.document_ids),
                        "chunk_ids": self.separator.join(pred.chunk_ids),
                    }
                )

        return target_path
