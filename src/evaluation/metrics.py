from __future__ import annotations

from typing import Any, Sequence

from src.data.schemas import FinalPrediction, GroundTruthRecord


def calculate_f2(precision: float, recall: float, beta: float = 2.0) -> float:
    """Calculate F_beta score (default beta=2.0 favoring recall).

    Formula:
        F_beta = (1 + beta^2) * P * R / (beta^2 * P + R)
    For beta = 2.0:
        F_2 = 5 * P * R / (4 * P + R)

    Args:
        precision: Precision in [0.0, 1.0].
        recall: Recall in [0.0, 1.0].
        beta: Weight of recall relative to precision (default 2.0).

    Returns:
        F_beta score in [0.0, 1.0]. Returns 0.0 if denominator is <= 0.
    """
    if precision <= 0.0 or recall <= 0.0:
        return 0.0

    beta_sq = beta * beta
    denominator = (beta_sq * precision) + recall
    if denominator <= 0.0:
        return 0.0

    numerator = (1.0 + beta_sq) * precision * recall
    return numerator / denominator


def calculate_precision_recall_fbeta(
    predicted: Sequence[str] | set[str],
    ground_truth: Sequence[str] | set[str],
    beta: float = 2.0,
) -> dict[str, float]:
    """Calculate Precision, Recall, and F_beta for a set of predicted IDs vs ground truth IDs.

    Enforces strict edge case rules:
    - |GT| == 0 and |P| == 0: Precision = 1.0, Recall = 1.0, F2 = 1.0
    - |GT| == 0 and |P| > 0:  Precision = 0.0, Recall = 0.0, F2 = 0.0
    - |GT| > 0 and |P| == 0:  Precision = 0.0, Recall = 0.0, F2 = 0.0

    Args:
        predicted: Collection of predicted item IDs (documents or chunks).
        ground_truth: Collection of ground truth item IDs.
        beta: Beta factor for F-score (default 2.0).

    Returns:
        Dictionary containing 'precision', 'recall', 'f2', and 'f_beta'.
    """
    pred_set = set(predicted)
    gt_set = set(ground_truth)

    len_gt = len(gt_set)
    len_pred = len(pred_set)

    # Edge cases
    if len_gt == 0 and len_pred == 0:
        return {"precision": 1.0, "recall": 1.0, "f2": 1.0, "f_beta": 1.0}
    if len_gt == 0 and len_pred > 0:
        return {"precision": 0.0, "recall": 0.0, "f2": 0.0, "f_beta": 0.0}
    if len_gt > 0 and len_pred == 0:
        return {"precision": 0.0, "recall": 0.0, "f2": 0.0, "f_beta": 0.0}

    # Standard TP / FP / FN calculation
    tp = len(pred_set & gt_set)
    precision = tp / len_pred if len_pred > 0 else 0.0
    recall = tp / len_gt if len_gt > 0 else 0.0
    f_val = calculate_f2(precision=precision, recall=recall, beta=beta)

    return {
        "precision": precision,
        "recall": recall,
        "f2": f_val,
        "f_beta": f_val,
    }


def evaluate_query(
    prediction: FinalPrediction | dict[str, Any] | Sequence[str] | set[str],
    ground_truth: GroundTruthRecord | dict[str, Any] | Sequence[str] | set[str],
    predicted_chunks: Sequence[str] | set[str] | None = None,
    ground_truth_chunks: Sequence[str] | set[str] | None = None,
    beta: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Evaluate a single query prediction against ground truth for documents and chunks.

    Args:
        prediction: FinalPrediction object, dict, or collection of predicted doc IDs.
        ground_truth: GroundTruthRecord object, dict, or collection of ground truth doc IDs.
        predicted_chunks: Optional collection of predicted chunk IDs (overrides prediction).
        ground_truth_chunks: Optional collection of ground truth chunk IDs (overrides ground_truth).
        beta: Beta factor for F-score (default 2.0).

    Returns:
        Dictionary with 'documents' and 'chunks' metrics.
    """
    # Extract predicted documents and chunks
    if isinstance(prediction, FinalPrediction):
        pred_docs = prediction.document_ids
        pred_chunks = prediction.chunk_ids
    elif isinstance(prediction, dict):
        pred_docs = prediction.get("document_ids", [])
        pred_chunks = prediction.get("chunk_ids", [])
    else:
        pred_docs = prediction
        pred_chunks = []

    if predicted_chunks is not None:
        pred_chunks = predicted_chunks

    # Extract ground truth documents and chunks
    if isinstance(ground_truth, GroundTruthRecord):
        gt_docs = ground_truth.relevant_doc_ids
        gt_chunks = ground_truth.relevant_chunk_ids
    elif isinstance(ground_truth, dict):
        gt_docs = ground_truth.get("relevant_doc_ids", set())
        gt_chunks = ground_truth.get("relevant_chunk_ids", set())
    else:
        gt_docs = ground_truth
        gt_chunks = set()

    if ground_truth_chunks is not None:
        gt_chunks = ground_truth_chunks

    doc_metrics = calculate_precision_recall_fbeta(pred_docs, gt_docs, beta=beta)
    chunk_metrics = calculate_precision_recall_fbeta(pred_chunks, gt_chunks, beta=beta)

    return {
        "documents": doc_metrics,
        "chunks": chunk_metrics,
    }


def evaluate_predictions(
    predictions: Sequence[FinalPrediction] | dict[str, FinalPrediction],
    ground_truth: Sequence[GroundTruthRecord] | dict[str, GroundTruthRecord],
    beta: float = 2.0,
) -> dict[str, Any]:
    """Evaluate full prediction set against ground truth records and calculate Macro F2.

    Formula:
        Macro F_2 = 1 / |Q| * sum_{q in Q} F_2(q)
    If |Q| == 0: returns 0.0 for macro metrics.

    Args:
        predictions: List or dict (keyed by query_id) of FinalPrediction objects.
        ground_truth: List or dict (keyed by query_id) of GroundTruthRecord objects.
        beta: Beta factor for F-score (default 2.0).

    Returns:
        Dictionary with aggregated 'documents', 'chunks' macro metrics, and 'per_query' results.
    """
    # Normalize ground truth to dict
    if isinstance(ground_truth, dict):
        gt_dict = ground_truth
    else:
        gt_dict = {gt.query_id: gt for gt in ground_truth}

    # Normalize predictions to dict
    if isinstance(predictions, dict):
        pred_dict = predictions
    else:
        pred_dict = {p.query_id: p for p in predictions}

    num_queries = len(gt_dict)
    if num_queries == 0:
        return {
            "documents": {
                "macro_f2": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "num_queries": 0,
            },
            "chunks": {
                "macro_f2": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "num_queries": 0,
            },
            "per_query": {},
        }

    per_query: dict[str, dict[str, dict[str, float]]] = {}

    for qid, gt_record in gt_dict.items():
        pred_record = pred_dict.get(qid)
        if pred_record is None:
            # Missing prediction treated as empty prediction (|P| == 0)
            pred_record = FinalPrediction(
                query_id=qid,
                document_ids=[],
                chunk_ids=[],
                document_scores={},
                chunk_scores={},
            )
        per_query[qid] = evaluate_query(pred_record, gt_record, beta=beta)

    # Compute Macro averages across all queries
    macro_doc_f2 = sum(q["documents"]["f2"] for q in per_query.values()) / num_queries
    macro_doc_precision = sum(q["documents"]["precision"] for q in per_query.values()) / num_queries
    macro_doc_recall = sum(q["documents"]["recall"] for q in per_query.values()) / num_queries

    macro_chunk_f2 = sum(q["chunks"]["f2"] for q in per_query.values()) / num_queries
    macro_chunk_precision = sum(q["chunks"]["precision"] for q in per_query.values()) / num_queries
    macro_chunk_recall = sum(q["chunks"]["recall"] for q in per_query.values()) / num_queries

    return {
        "documents": {
            "macro_f2": macro_doc_f2,
            "macro_precision": macro_doc_precision,
            "macro_recall": macro_doc_recall,
            "num_queries": num_queries,
        },
        "chunks": {
            "macro_f2": macro_chunk_f2,
            "macro_precision": macro_chunk_precision,
            "macro_recall": macro_chunk_recall,
            "num_queries": num_queries,
        },
        "per_query": per_query,
    }
