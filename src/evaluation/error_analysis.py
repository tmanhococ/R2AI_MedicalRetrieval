from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from src.data.schemas import FinalPrediction, GroundTruthRecord, QueryRecord
from src.evaluation.metrics import calculate_precision_recall_fbeta


def generate_error_analysis_report(
    predictions: Sequence[FinalPrediction],
    ground_truths: Sequence[GroundTruthRecord],
    output_dir: Path | str,
    queries: Sequence[QueryRecord] | None = None,
    beta: float = 2.0,
) -> dict[str, Path]:
    """Generate detailed error analysis reports and metrics artifacts.

    Produces three artifacts in output_dir:
        1. per_query_metrics.csv: Precision, Recall, F2 for docs and chunks per query.
        2. error_cases.csv: Queries with F2 < 1.0 detailing FN/FP errors.
        3. summary.json: Macro F2 metrics, zero-F2 count, perfect count, and total queries.

    Args:
        predictions: Sequence of FinalPrediction records.
        ground_truths: Sequence of GroundTruthRecord objects.
        output_dir: Target directory where CSVs and summary JSON will be saved.
        queries: Optional sequence of QueryRecord objects to populate query text.
        beta: Beta parameter for F-score calculation (default 2.0).

    Returns:
        Dictionary mapping artifact keys ('per_query_metrics', 'error_cases', 'summary')
        to their resolved Path objects.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    pred_map = {p.query_id: p for p in predictions}
    gt_map = {gt.query_id: gt for gt in ground_truths}

    query_text_map: dict[str, str] = {}
    if queries is not None:
        for q in queries:
            text = getattr(q, "text_vi", None) or getattr(q, "text", "")
            query_text_map[q.query_id] = text

    all_qids = sorted(set(gt_map.keys()) | set(pred_map.keys()))

    per_query_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, str]] = []

    doc_f2_list: list[float] = []
    chunk_f2_list: list[float] = []
    zero_f2_count = 0
    perfect_count = 0

    for qid in all_qids:
        pred = pred_map.get(qid)
        gt = gt_map.get(qid)

        pred_docs = pred.document_ids if pred is not None else []
        pred_chunks = pred.chunk_ids if pred is not None else []

        gt_docs = gt.relevant_doc_ids if gt is not None else set()
        gt_chunks = gt.relevant_chunk_ids if gt is not None else set()

        doc_metrics = calculate_precision_recall_fbeta(pred_docs, gt_docs, beta=beta)
        chunk_metrics = calculate_precision_recall_fbeta(pred_chunks, gt_chunks, beta=beta)

        d_f2 = doc_metrics["f2"]
        c_f2 = chunk_metrics["f2"]

        doc_f2_list.append(d_f2)
        chunk_f2_list.append(c_f2)

        if d_f2 == 1.0 and c_f2 == 1.0:
            perfect_count += 1
        if d_f2 == 0.0 and c_f2 == 0.0:
            zero_f2_count += 1

        per_query_rows.append(
            {
                "query_id": qid,
                "doc_precision": doc_metrics["precision"],
                "doc_recall": doc_metrics["recall"],
                "doc_f2": d_f2,
                "chunk_precision": chunk_metrics["precision"],
                "chunk_recall": chunk_metrics["recall"],
                "chunk_f2": c_f2,
                "num_pred_docs": len(pred_docs),
                "num_gt_docs": len(gt_docs),
                "num_pred_chunks": len(pred_chunks),
                "num_gt_chunks": len(gt_chunks),
            }
        )

        # Generate error cases for queries with F2 < 1.0
        if d_f2 < 1.0 or c_f2 < 1.0:
            q_text = query_text_map.get(qid, "")
            pred_doc_set = set(pred_docs)
            gt_doc_set = set(gt_docs)
            pred_chunk_set = set(pred_chunks)
            gt_chunk_set = set(gt_chunks)

            missing_docs = sorted(gt_doc_set - pred_doc_set)
            spurious_docs = sorted(pred_doc_set - gt_doc_set)
            missing_chunks = sorted(gt_chunk_set - pred_chunk_set)
            spurious_chunks = sorted(pred_chunk_set - gt_chunk_set)

            if missing_docs:
                error_rows.append(
                    {
                        "query_id": qid,
                        "query_text": q_text,
                        "error_type": "doc_fn",
                        "missing_ids": " ".join(missing_docs),
                        "spurious_ids": "",
                    }
                )

            if spurious_docs:
                error_rows.append(
                    {
                        "query_id": qid,
                        "query_text": q_text,
                        "error_type": "doc_fp",
                        "missing_ids": "",
                        "spurious_ids": " ".join(spurious_docs),
                    }
                )

            if missing_chunks:
                error_rows.append(
                    {
                        "query_id": qid,
                        "query_text": q_text,
                        "error_type": "chunk_fn",
                        "missing_ids": " ".join(missing_chunks),
                        "spurious_ids": "",
                    }
                )

            if spurious_chunks:
                error_rows.append(
                    {
                        "query_id": qid,
                        "query_text": q_text,
                        "error_type": "chunk_fp",
                        "missing_ids": "",
                        "spurious_ids": " ".join(spurious_chunks),
                    }
                )

    # 1. Write per_query_metrics.csv
    per_query_path = target_dir / "per_query_metrics.csv"
    per_query_fields = [
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
    with open(per_query_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=per_query_fields)
        writer.writeheader()
        writer.writerows(per_query_rows)

    # 2. Write error_cases.csv
    error_cases_path = target_dir / "error_cases.csv"
    error_fields = ["query_id", "query_text", "error_type", "missing_ids", "spurious_ids"]
    with open(error_cases_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=error_fields)
        writer.writeheader()
        writer.writerows(error_rows)

    # 3. Write summary.json
    summary_path = target_dir / "summary.json"
    total_q = len(all_qids)
    macro_doc = (sum(doc_f2_list) / total_q) if total_q > 0 else 0.0
    macro_chunk = (sum(chunk_f2_list) / total_q) if total_q > 0 else 0.0
    combined = (macro_doc + macro_chunk) / 2.0 if total_q > 0 else 0.0

    summary_data = {
        "macro_doc_f2": macro_doc,
        "macro_chunk_f2": macro_chunk,
        "combined_f2": combined,
        "zero_f2_query_count": zero_f2_count,
        "perfect_query_count": perfect_count,
        "total_queries": total_q,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    return {
        "per_query_metrics": per_query_path,
        "error_cases": error_cases_path,
        "summary": summary_path,
    }
