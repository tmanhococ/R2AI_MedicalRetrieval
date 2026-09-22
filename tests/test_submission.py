from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from src.data.schemas import FinalPrediction
from src.submission.base import SubmissionExporter, create_exporter
from src.submission.competition_csv import CompetitionCsvExporter
from src.submission.internal_jsonl import InternalJsonlExporter


def _sample_predictions() -> list[FinalPrediction]:
    return [
        FinalPrediction(
            query_id="q2",
            document_ids=["doc_b", "doc_c"],
            chunk_ids=["chunk_b1"],
            document_scores={"doc_b": 0.85, "doc_c": 0.72},
            chunk_scores={"chunk_b1": 0.85},
        ),
        FinalPrediction(
            query_id="q1",
            document_ids=["doc_a"],
            chunk_ids=["chunk_a1", "chunk_a2"],
            document_scores={"doc_a": 0.95},
            chunk_scores={"chunk_a1": 0.95, "chunk_a2": 0.88},
        ),
    ]


def test_internal_jsonl_exporter_protocol():
    exporter = InternalJsonlExporter()
    assert isinstance(exporter, SubmissionExporter)


def test_competition_csv_exporter_protocol():
    exporter = CompetitionCsvExporter()
    assert isinstance(exporter, SubmissionExporter)


def test_internal_jsonl_export(tmp_path: Path):
    exporter = InternalJsonlExporter()
    out_file = tmp_path / "nested" / "output.jsonl"

    preds = _sample_predictions()
    res_path = exporter.export(preds, out_file)

    assert res_path == out_file
    assert out_file.exists()

    lines = out_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    # Verify deterministic ordering by query_id (q1 before q2)
    record1 = json.loads(lines[0])
    record2 = json.loads(lines[1])

    assert record1["query_id"] == "q1"
    assert record1["document_ids"] == ["doc_a"]
    assert record1["chunk_ids"] == ["chunk_a1", "chunk_a2"]
    assert record1["document_scores"] == {"doc_a": 0.95}
    assert record1["chunk_scores"] == {"chunk_a1": 0.95, "chunk_a2": 0.88}

    assert record2["query_id"] == "q2"
    assert record2["document_ids"] == ["doc_b", "doc_c"]
    assert record2["chunk_ids"] == ["chunk_b1"]


def test_internal_jsonl_export_empty(tmp_path: Path):
    exporter = InternalJsonlExporter()
    out_file = tmp_path / "empty.jsonl"

    res_path = exporter.export([], out_file)
    assert res_path == out_file
    assert out_file.exists()
    assert out_file.read_text(encoding="utf-8").strip() == ""


def test_competition_csv_export_default(tmp_path: Path):
    exporter = CompetitionCsvExporter()
    out_file = tmp_path / "sub" / "submission.csv"

    preds = _sample_predictions()
    res_path = exporter.export(preds, out_file)

    assert res_path == out_file
    assert out_file.exists()

    with open(out_file, mode="r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))

    assert len(reader) == 2
    # Verify deterministic ordering by query_id
    assert reader[0]["query_id"] == "q1"
    assert reader[0]["document_ids"] == "doc_a"
    assert reader[0]["chunk_ids"] == "chunk_a1 chunk_a2"

    assert reader[1]["query_id"] == "q2"
    assert reader[1]["document_ids"] == "doc_b doc_c"
    assert reader[1]["chunk_ids"] == "chunk_b1"


def test_competition_csv_export_custom_separator(tmp_path: Path):
    exporter = CompetitionCsvExporter(separator=";")
    out_file = tmp_path / "submission_semicolon.csv"

    preds = _sample_predictions()
    exporter.export(preds, out_file)

    with open(out_file, mode="r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))

    assert reader[0]["chunk_ids"] == "chunk_a1;chunk_a2"
    assert reader[1]["document_ids"] == "doc_b;doc_c"


def test_competition_csv_export_empty(tmp_path: Path):
    exporter = CompetitionCsvExporter()
    out_file = tmp_path / "empty.csv"

    exporter.export([], out_file)
    assert out_file.exists()

    with open(out_file, mode="r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0] == ["query_id", "document_ids", "chunk_ids"]


def test_create_exporter_factory():
    jsonl_exp = create_exporter("internal_jsonl")
    assert isinstance(jsonl_exp, InternalJsonlExporter)

    csv_exp = create_exporter("competition_csv", separator=";")
    assert isinstance(csv_exp, CompetitionCsvExporter)
    assert csv_exp.separator == ";"

    with pytest.raises(ValueError, match="Unknown exporter type"):
        create_exporter("invalid_format")


def test_package_exports():
    import src.submission as sub
    assert hasattr(sub, "SubmissionExporter")
    assert hasattr(sub, "InternalJsonlExporter")
    assert hasattr(sub, "CompetitionCsvExporter")
    assert hasattr(sub, "create_exporter")
