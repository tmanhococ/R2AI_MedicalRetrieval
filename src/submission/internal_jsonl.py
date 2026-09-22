from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from src.data.schemas import FinalPrediction


class InternalJsonlExporter:
    """Exports predictions to internal JSONL format.

    Each line is a JSON object with:
        - query_id: str
        - document_ids: list[str]
        - chunk_ids: list[str]
        - document_scores: dict[str, float]
        - chunk_scores: dict[str, float]
    """

    def export(
        self,
        predictions: Sequence[FinalPrediction],
        output_path: Path | str,
    ) -> Path:
        """Export predictions to a JSONL file.

        Args:
            predictions: Sequence of FinalPrediction records.
            output_path: Target path to output file.

        Returns:
            Resolved Path object.
        """
        target_path = Path(output_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        sorted_preds = sorted(predictions, key=lambda p: p.query_id)

        with open(target_path, "w", encoding="utf-8") as f:
            for pred in sorted_preds:
                record = {
                    "query_id": pred.query_id,
                    "document_ids": list(pred.document_ids),
                    "chunk_ids": list(pred.chunk_ids),
                    "document_scores": dict(pred.document_scores),
                    "chunk_scores": dict(pred.chunk_scores),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return target_path
