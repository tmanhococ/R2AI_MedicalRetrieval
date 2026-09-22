from __future__ import annotations

import csv
import json
from pathlib import Path
import pytest

from src.data.schemas import FinalPrediction, GroundTruthRecord, QueryRecord
from src.evaluation.error_analysis import generate_error_analysis_report


def _sample_data():
    queries = [
        QueryRecord(query_id="q1", text_vi="Bệnh tiểu đường điều trị thế nào?"),
        QueryRecord(query_id="q2", text_vi="Triệu chứng viêm phổi ở trẻ em?"),
        QueryRecord(query_id="q3", text_vi="Tác dụng phụ của thuốc hạ sốt?"),
    ]

    # q1: Perfect match (doc_f2 = 1.0, chunk_f2 = 1.0)
    # q2: Missing doc & chunk (FN), Spurious doc & chunk (FP)
    # q3: Complete failure (doc_f2 = 0.0, chunk_f2 = 0.0)
    ground_truths = [
        GroundTruthRecord(
            query_id="q1",
            relevant_doc_ids={"doc_1"},
            relevant_chunk_ids={"chunk_1_1"},
        ),
        GroundTruthRecord(
            query_id="q2",
            relevant_doc_ids={"doc_2", "doc_3"},
            relevant_chunk_ids={"chunk_2_1", "chunk_3_1"},
        ),
        GroundTruthRecord(
            query_id="q3",
            relevant_doc_ids={"doc_4"},
            relevant_chunk_ids={"chunk_4_1"},
        ),
    ]

    predictions = [
        FinalPrediction(
            query_id="q1",
            document_ids=["doc_1"],
            chunk_ids=["chunk_1_1"],
            document_scores={"doc_1": 0.95},
            chunk_scores={"chunk_1_1": 0.95},
        ),
        FinalPrediction(
            query_id="q2",
            # doc_2 retrieved, doc_3 missing (FN), doc_99 spurious (FP)
            document_ids=["doc_2", "doc_99"],
            # chunk_2_1 retrieved, chunk_3_1 missing (FN), chunk_99_1 spurious (FP)
            chunk_ids=["chunk_2_1", "chunk_99_1"],
            document_scores={"doc_2": 0.8, "doc_99": 0.7},
            chunk_scores={"chunk_2_1": 0.8, "chunk_99_1": 0.7},
        ),
        FinalPrediction(
            query_id="q3",
            document_ids=["doc_wrong"],
            chunk_ids=["chunk_wrong"],
            document_scores={"doc_wrong": 0.6},
            chunk_scores={"chunk_wrong": 0.6},
        ),
    ]

    return queries, ground_truths, predictions


def test_generate_error_analysis_report_structure(tmp_path: Path):
    queries, ground_truths, predictions = _sample_data()
    out_dir = tmp_path / "reports" / "run_01"

    artifact_paths = generate_error_analysis_report(
        predictions=predictions,
        ground_truths=ground_truths,
        output_dir=out_dir,
        queries=queries,
    )

    assert set(artifact_paths.keys()) == {"per_query_metrics", "error_cases", "summary"}
    for p in artifact_paths.values():
        assert isinstance(p, Path)
        assert p.exists()


def test_per_query_metrics_content(tmp_path: Path):
    queries, ground_truths, predictions = _sample_data()
    out_dir = tmp_path / "reports"

    paths = generate_error_analysis_report(
        predictions=predictions,
        ground_truths=ground_truths,
        output_dir=out_dir,
        queries=queries,
    )

    with open(paths["per_query_metrics"], mode="r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    # Check expected columns
    expected_cols = [
        "query_id",
        "doc_precision",
        "doc_recall",
        "doc_f2",
        "chunk_precision",
        "chunk_recall",
        "chunk_f2",
        "num_pred_docs",
        "num_gt_docs",
        "num_pred_chunks",
        "num_gt_chunks",
    ]
    assert list(rows[0].keys()) == expected_cols

    # Verify deterministic sorting by query_id
    assert [r["query_id"] for r in rows] == ["q1", "q2", "q3"]

    # q1 is perfect
    assert float(rows[0]["doc_f2"]) == pytest.approx(1.0)
    assert float(rows[0]["chunk_f2"]) == pytest.approx(1.0)
    assert int(rows[0]["num_pred_docs"]) == 1
    assert int(rows[0]["num_gt_docs"]) == 1

    # q2 has partial retrieval
    assert 0.0 < float(rows[1]["doc_f2"]) < 1.0
    assert 0.0 < float(rows[1]["chunk_f2"]) < 1.0

    # q3 has zero retrieval
    assert float(rows[2]["doc_f2"]) == 0.0
    assert float(rows[2]["chunk_f2"]) == 0.0


def test_error_cases_content_with_queries(tmp_path: Path):
    queries, ground_truths, predictions = _sample_data()
    out_dir = tmp_path / "reports"

    paths = generate_error_analysis_report(
        predictions=predictions,
        ground_truths=ground_truths,
        output_dir=out_dir,
        queries=queries,
    )

    with open(paths["error_cases"], mode="r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    # Expected columns
    expected_cols = ["query_id", "query_text", "error_type", "missing_ids", "spurious_ids"]
    assert list(rows[0].keys()) == expected_cols

    # q1 is perfect, so q1 should NOT appear in error_cases
    query_ids_in_errors = [r["query_id"] for r in rows]
    assert "q1" not in query_ids_in_errors
    assert "q2" in query_ids_in_errors
    assert "q3" in query_ids_in_errors

    # Inspect q2 errors:
    # doc_3 was missing (doc_fn), doc_99 was spurious (doc_fp)
    # chunk_3_1 was missing (chunk_fn), chunk_99_1 was spurious (chunk_fp)
    q2_rows = [r for r in rows if r["query_id"] == "q2"]
    q2_error_types = {r["error_type"]: r for r in q2_rows}

    assert "doc_fn" in q2_error_types
    assert q2_error_types["doc_fn"]["missing_ids"] == "doc_3"
    assert q2_error_types["doc_fn"]["query_text"] == "Triệu chứng viêm phổi ở trẻ em?"

    assert "doc_fp" in q2_error_types
    assert q2_error_types["doc_fp"]["spurious_ids"] == "doc_99"

    assert "chunk_fn" in q2_error_types
    assert q2_error_types["chunk_fn"]["missing_ids"] == "chunk_3_1"

    assert "chunk_fp" in q2_error_types
    assert q2_error_types["chunk_fp"]["spurious_ids"] == "chunk_99_1"


def test_error_cases_without_queries(tmp_path: Path):
    _, ground_truths, predictions = _sample_data()
    out_dir = tmp_path / "reports"

    paths = generate_error_analysis_report(
        predictions=predictions,
        ground_truths=ground_truths,
        output_dir=out_dir,
        queries=None,
    )

    with open(paths["error_cases"], mode="r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) > 0
    # query_text should be empty string when queries is None
    for r in rows:
        assert r["query_text"] == ""


def test_summary_json_metrics(tmp_path: Path):
    queries, ground_truths, predictions = _sample_data()
    out_dir = tmp_path / "reports"

    paths = generate_error_analysis_report(
        predictions=predictions,
        ground_truths=ground_truths,
        output_dir=out_dir,
        queries=queries,
    )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

    assert "macro_doc_f2" in summary
    assert "macro_chunk_f2" in summary
    assert "combined_f2" in summary
    assert "zero_f2_query_count" in summary
    assert "perfect_query_count" in summary
    assert "total_queries" in summary

    assert summary["total_queries"] == 3
    assert summary["perfect_query_count"] == 1  # q1
    assert summary["zero_f2_query_count"] == 1  # q3

    expected_combined = (summary["macro_doc_f2"] + summary["macro_chunk_f2"]) / 2.0
    assert summary["combined_f2"] == pytest.approx(expected_combined)


def test_error_analysis_empty_inputs(tmp_path: Path):
    out_dir = tmp_path / "empty_report"
    paths = generate_error_analysis_report(
        predictions=[],
        ground_truths=[],
        output_dir=out_dir,
    )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["total_queries"] == 0
    assert summary["macro_doc_f2"] == 0.0
    assert summary["macro_chunk_f2"] == 0.0
    assert summary["combined_f2"] == 0.0
    assert summary["zero_f2_query_count"] == 0
    assert summary["perfect_query_count"] == 0

    with open(paths["per_query_metrics"], mode="r", encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 1  # header only

    with open(paths["error_cases"], mode="r", encoding="utf-8", newline="") as f:
        reader = list(csv.reader(f))
    assert len(reader) == 1  # header only


def test_error_analysis_missing_predictions(tmp_path: Path):
    # If a query is in ground_truths but not in predictions, it should be treated as empty prediction
    gt = [
        GroundTruthRecord(
            query_id="q1",
            relevant_doc_ids={"doc_1"},
            relevant_chunk_ids={"chunk_1"},
        )
    ]
    out_dir = tmp_path / "missing_preds"
    paths = generate_error_analysis_report(
        predictions=[],
        ground_truths=gt,
        output_dir=out_dir,
    )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    assert summary["total_queries"] == 1
    assert summary["zero_f2_query_count"] == 1
    assert summary["macro_doc_f2"] == 0.0

    with open(paths["error_cases"], mode="r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # doc_fn and chunk_fn
    error_types = {r["error_type"] for r in rows}
    assert error_types == {"doc_fn", "chunk_fn"}


def test_error_analysis_package_reexport():
    from src.evaluation import generate_error_analysis_report as report_fn
    assert callable(report_fn)


def test_error_analysis_custom_beta(tmp_path: Path):
    # Pred: 1 doc, GT: 2 docs -> P=1.0, R=0.5
    # beta=1.0 -> F1 = 2*1*0.5/(1+0.5) = 1/1.5 = 2/3 = 0.6667
    # beta=2.0 -> F2 = 5*1*0.5/(4*1+0.5) = 2.5/4.5 = 5/9 = 0.5556
    preds = [
        FinalPrediction(
            query_id="q1",
            document_ids=["d1"],
            chunk_ids=[],
            document_scores={"d1": 1.0},
            chunk_scores={},
        )
    ]
    gt = [
        GroundTruthRecord(
            query_id="q1",
            relevant_doc_ids={"d1", "d2"},
            relevant_chunk_ids=set(),
        )
    ]
    out_dir = tmp_path / "custom_beta"
    paths = generate_error_analysis_report(
        predictions=preds,
        ground_truths=gt,
        output_dir=out_dir,
        beta=1.0,
    )
    with open(paths["per_query_metrics"], mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert float(rows[0]["doc_f2"]) == pytest.approx(2.0 / 3.0)
