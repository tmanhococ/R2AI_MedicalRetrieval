from __future__ import annotations

import pytest

from src.data.schemas import FinalPrediction, GroundTruthRecord
from src.evaluation.metrics import (
    calculate_f2,
    calculate_precision_recall_fbeta,
    evaluate_predictions,
    evaluate_query,
)


def test_calculate_f2_standard():
    # P = 0.5, R = 1.0 -> F2 = 5 * 0.5 * 1 / (4 * 0.5 + 1) = 2.5 / 3.0 = 5/6
    f2 = calculate_f2(precision=0.5, recall=1.0)
    assert f2 == pytest.approx(5.0 / 6.0)


def test_calculate_f2_perfect():
    assert calculate_f2(precision=1.0, recall=1.0) == pytest.approx(1.0)


def test_calculate_f2_zero_cases():
    assert calculate_f2(precision=0.0, recall=1.0) == 0.0
    assert calculate_f2(precision=1.0, recall=0.0) == 0.0
    assert calculate_f2(precision=0.0, recall=0.0) == 0.0


def test_calculate_f2_custom_beta():
    # When beta=1.0, F1 = 2 * P * R / (P + R)
    # P = 0.5, R = 1.0 -> 2 * 0.5 * 1 / 1.5 = 1 / 1.5 = 2/3
    f1 = calculate_f2(precision=0.5, recall=1.0, beta=1.0)
    assert f1 == pytest.approx(2.0 / 3.0)


def test_calculate_precision_recall_fbeta_standard():
    predicted = ["d1", "d2"]
    ground_truth = ["d2", "d3"]
    metrics = calculate_precision_recall_fbeta(predicted, ground_truth, beta=2.0)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["f2"] == pytest.approx(0.5)


def test_calculate_precision_recall_fbeta_edge_cases():
    # Rule 1: Empty GT & Empty Pred -> 1.0
    res1 = calculate_precision_recall_fbeta([], [])
    assert res1["precision"] == 1.0
    assert res1["recall"] == 1.0
    assert res1["f2"] == 1.0

    # Rule 2: Empty GT & Non-empty Pred -> 0.0
    res2 = calculate_precision_recall_fbeta(["d1"], [])
    assert res2["precision"] == 0.0
    assert res2["recall"] == 0.0
    assert res2["f2"] == 0.0

    # Rule 3: Non-empty GT & Empty Pred -> 0.0
    res3 = calculate_precision_recall_fbeta([], ["d1"])
    assert res3["precision"] == 0.0
    assert res3["recall"] == 0.0
    assert res3["f2"] == 0.0


def test_calculate_precision_recall_fbeta_duplicates_handled():
    # Duplicates in predictions should be deduplicated (set semantics)
    predicted = ["d1", "d1", "d2"]
    ground_truth = ["d1", "d2"]
    res = calculate_precision_recall_fbeta(predicted, ground_truth)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f2"] == 1.0


def test_evaluate_query_with_schemas():
    pred = FinalPrediction(
        query_id="q1",
        document_ids=["d1", "d2"],
        chunk_ids=["d1_c1"],
        document_scores={"d1": 0.9, "d2": 0.8},
        chunk_scores={"d1_c1": 0.9},
    )
    gt = GroundTruthRecord(
        query_id="q1",
        relevant_doc_ids={"d1", "d3"},
        relevant_chunk_ids={"d1_c1"},
    )
    result = evaluate_query(pred, gt)
    assert "documents" in result
    assert "chunks" in result
    # Docs: TP=1, FP=1, FN=1 -> P=0.5, R=0.5, F2=0.5
    assert result["documents"]["precision"] == pytest.approx(0.5)
    assert result["documents"]["recall"] == pytest.approx(0.5)
    assert result["documents"]["f2"] == pytest.approx(0.5)
    # Chunks: TP=1, FP=0, FN=0 -> P=1.0, R=1.0, F2=1.0
    assert result["chunks"]["precision"] == pytest.approx(1.0)
    assert result["chunks"]["recall"] == pytest.approx(1.0)
    assert result["chunks"]["f2"] == pytest.approx(1.0)


def test_evaluate_query_with_raw_collections():
    result = evaluate_query(
        prediction=["d1"],
        ground_truth=["d1"],
        predicted_chunks=["c1"],
        ground_truth_chunks=["c1"],
    )
    assert result["documents"]["f2"] == 1.0
    assert result["chunks"]["f2"] == 1.0


def test_evaluate_predictions_empty_ground_truth_guard():
    preds = [
        FinalPrediction(
            query_id="q1",
            document_ids=[],
            chunk_ids=[],
            document_scores={},
            chunk_scores={},
        )
    ]
    gt = [
        GroundTruthRecord(
            query_id="q1",
            relevant_doc_ids=set(),
            relevant_chunk_ids=set(),
        )
    ]
    results = evaluate_predictions(preds, gt)
    assert results["documents"]["macro_f2"] == 1.0
    assert results["chunks"]["macro_f2"] == 1.0


def test_evaluate_predictions_empty_query_set():
    results = evaluate_predictions([], [])
    assert results["documents"]["macro_f2"] == 0.0
    assert results["chunks"]["macro_f2"] == 0.0
    assert results["documents"]["num_queries"] == 0


def test_evaluate_predictions_macro_averaging():
    preds = [
        FinalPrediction(
            query_id="q1",
            document_ids=["d1"],
            chunk_ids=["c1"],
            document_scores={"d1": 0.9},
            chunk_scores={"c1": 0.9},
        ),
        FinalPrediction(
            query_id="q2",
            document_ids=["d3"],
            chunk_ids=["c3"],
            document_scores={"d3": 0.8},
            chunk_scores={"c3": 0.8},
        ),
    ]
    gt = [
        # q1: perfect match (f2=1.0)
        GroundTruthRecord(
            query_id="q1",
            relevant_doc_ids={"d1"},
            relevant_chunk_ids={"c1"},
        ),
        # q2: zero match (f2=0.0)
        GroundTruthRecord(
            query_id="q2",
            relevant_doc_ids={"d2"},
            relevant_chunk_ids={"c2"},
        ),
    ]
    results = evaluate_predictions(preds, gt)
    assert results["documents"]["macro_f2"] == pytest.approx(0.5)
    assert results["chunks"]["macro_f2"] == pytest.approx(0.5)
    assert results["documents"]["num_queries"] == 2
    assert "per_query" in results
    assert results["per_query"]["q1"]["documents"]["f2"] == 1.0
    assert results["per_query"]["q2"]["documents"]["f2"] == 0.0


def test_evaluate_predictions_missing_prediction_handled_as_empty():
    preds = []  # No predictions produced for q1
    gt = [
        GroundTruthRecord(
            query_id="q1",
            relevant_doc_ids={"d1"},
            relevant_chunk_ids={"c1"},
        )
    ]
    results = evaluate_predictions(preds, gt)
    # Non-empty GT & empty pred -> 0.0
    assert results["documents"]["macro_f2"] == 0.0
    assert results["chunks"]["macro_f2"] == 0.0


def test_evaluate_predictions_dict_inputs():
    preds = {
        "q1": FinalPrediction(
            query_id="q1",
            document_ids=["d1"],
            chunk_ids=["c1"],
            document_scores={"d1": 1.0},
            chunk_scores={"c1": 1.0},
        )
    }
    gt = {
        "q1": GroundTruthRecord(
            query_id="q1",
            relevant_doc_ids={"d1"},
            relevant_chunk_ids={"c1"},
        )
    }
    results = evaluate_predictions(preds, gt)
    assert results["documents"]["macro_f2"] == 1.0
