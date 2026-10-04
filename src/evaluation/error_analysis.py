from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from src.data.schemas import FinalPrediction, GroundTruthRecord, QueryRecord
from src.data.vibiomir_contracts import (
    ErrorBreakdown,
    EvaluationProfile,
    EvaluationRecord,
    ReferenceChunk,
    RetrievalTrace,
    SourceStore,
)
from src.evaluation.metrics import calculate_precision_recall_fbeta
from src.evaluation.vibiomir_metrics import PROVISIONAL_SPEC_V1_PROFILE
from src.evaluation.vibiomir_text import TextMatchProfile, TextOverlapEvaluator


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
            text = str(getattr(q, "text_vi", None) or getattr(q, "text", "") or "")
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


def build_error_breakdown(
    trace: RetrievalTrace,
    gold: EvaluationRecord,
    store: SourceStore,
    profile: EvaluationProfile | None = None,
) -> ErrorBreakdown:
    """Build fine-grained retrieval and evidence error breakdown with provenance.

    Traces each failure location:
    - missing_source_docs: Gold doc missing from SourceStore
    - missing_candidate_docs: Gold doc in SourceStore but not in candidate_doc_ids
    - lost_after_rerank_docs: Gold doc in candidate_doc_ids but not in reranked_doc_ids
    - lost_after_cutoff_docs: Gold doc in reranked_doc_ids but not in final_doc_ids
    - missing_chunk_refs: Gold reference chunks not matched by predicted chunks
    - evidence_missing_given_doc: Gold chunks missed where parent doc was in final_doc_ids

    Args:
        trace: Candidate score trajectory and retrieved IDs for the query.
        gold: Evaluation record containing ground truth doc IDs and reference chunks.
        store: SourceStore containing frozen snapshot documents and chunks.
        profile: Evaluation profile for text matching (defaults to PROVISIONAL_SPEC_V1_PROFILE).

    Returns:
        ErrorBreakdown detailing all document drop stages and missing evidence chunks.
    """
    if gold.query_id != trace.query_id:
        raise ValueError(
            f"Gold query_id ({gold.query_id}) does not match trace query_id ({trace.query_id})"
        )

    # 1. Document Drop Funnel
    missing_source_docs: list[int] = []
    missing_candidate_docs: list[int] = []
    lost_after_rerank_docs: list[int] = []
    lost_after_cutoff_docs: list[int] = []

    for doc_id in sorted(gold.relevant_doc_ids):
        if not store.contains_official_doc(doc_id):
            missing_source_docs.append(doc_id)
        elif doc_id not in trace.candidate_doc_ids:
            missing_candidate_docs.append(doc_id)
        elif doc_id not in trace.reranked_doc_ids:
            lost_after_rerank_docs.append(doc_id)
        elif doc_id not in trace.final_doc_ids:
            lost_after_cutoff_docs.append(doc_id)

    # 2. Predicted Chunks Resolution
    pred_chunks: list[tuple[int, str]] = []
    if (
        trace.metadata
        and "predicted_chunks" in trace.metadata
        and trace.metadata["predicted_chunks"] is not None
    ):
        for item in trace.metadata["predicted_chunks"]:
            if isinstance(item, dict):
                p_doc_id = int(item["doc_id"])
                p_text = str(item.get("chunk_text") if item.get("chunk_text") is not None else item.get("text", ""))
            else:
                p_doc_id = int(getattr(item, "doc_id"))
                p_text = str(getattr(item, "chunk_text", getattr(item, "text", "")))
            pred_chunks.append((p_doc_id, p_text))
    elif trace.final_chunk_ids:
        for cid in trace.final_chunk_ids:
            try:
                chunk = store.get_chunk(cid)
                pred_chunks.append((int(chunk.doc_id), str(chunk.text)))
            except Exception:
                pass

    # 3. Text Overlap Evaluator Setup
    if profile is None:
        eval_profile = PROVISIONAL_SPEC_V1_PROFILE
    else:
        eval_profile = profile

    if hasattr(eval_profile, "tokenizer_name") and hasattr(eval_profile, "threshold"):
        text_profile = TextMatchProfile(
            tokenizer_name=eval_profile.tokenizer_name,
            normalization_version=eval_profile.normalization_version,
            overlap_threshold=eval_profile.threshold,
        )
        evaluator = TextOverlapEvaluator(profile=text_profile)
    elif isinstance(eval_profile, TextMatchProfile):
        evaluator = TextOverlapEvaluator(profile=eval_profile)
    else:
        evaluator = TextOverlapEvaluator()

    # 4. Reference Chunk Matching
    missing_chunk_refs: list[ReferenceChunk] = []
    evidence_missing_given_doc: list[ReferenceChunk] = []

    for ref in gold.relevant_chunks:
        matched = False
        for p_doc_id, p_text in pred_chunks:
            if p_doc_id == ref.doc_id and evaluator.matches(p_text, ref.chunk_text):
                matched = True
                break

        if not matched:
            missing_chunk_refs.append(ref)
            if ref.doc_id in trace.final_doc_ids:
                evidence_missing_given_doc.append(ref)

    return ErrorBreakdown(
        query_id=trace.query_id,
        missing_source_docs=tuple(missing_source_docs),
        missing_candidate_docs=tuple(missing_candidate_docs),
        lost_after_rerank_docs=tuple(lost_after_rerank_docs),
        lost_after_cutoff_docs=tuple(lost_after_cutoff_docs),
        missing_chunk_refs=tuple(missing_chunk_refs),
        evidence_missing_given_doc=tuple(evidence_missing_given_doc),
    )


def generate_error_breakdown_summary(
    breakdowns: Sequence[ErrorBreakdown],
) -> dict[str, Any]:
    """Generate aggregate dataset-level summary of retrieval and evidence failures.

    Args:
        breakdowns: Sequence of ErrorBreakdown objects.

    Returns:
        Summary dict containing totals, query-level loss counts, and breakdown metrics.
    """
    total_queries = len(breakdowns)
    total_missing_source = sum(len(b.missing_source_docs) for b in breakdowns)
    total_missing_candidate = sum(len(b.missing_candidate_docs) for b in breakdowns)
    total_lost_rerank = sum(len(b.lost_after_rerank_docs) for b in breakdowns)
    total_lost_cutoff = sum(len(b.lost_after_cutoff_docs) for b in breakdowns)
    total_missing_chunks = sum(len(b.missing_chunk_refs) for b in breakdowns)
    total_evidence_missing_given_doc = sum(len(b.evidence_missing_given_doc) for b in breakdowns)

    queries_missing_source = sum(1 for b in breakdowns if b.missing_source_docs)
    queries_missing_candidate = sum(1 for b in breakdowns if b.missing_candidate_docs)
    queries_lost_rerank = sum(1 for b in breakdowns if b.lost_after_rerank_docs)
    queries_lost_cutoff = sum(1 for b in breakdowns if b.lost_after_cutoff_docs)
    queries_missing_chunks = sum(1 for b in breakdowns if b.missing_chunk_refs)
    queries_evidence_missing_given_doc = sum(1 for b in breakdowns if b.evidence_missing_given_doc)

    return {
        "total_queries": total_queries,
        "total_missing_source_docs": total_missing_source,
        "total_missing_candidate_docs": total_missing_candidate,
        "total_lost_after_rerank_docs": total_lost_rerank,
        "total_lost_after_cutoff_docs": total_lost_cutoff,
        "total_missing_chunk_refs": total_missing_chunks,
        "total_evidence_missing_given_doc": total_evidence_missing_given_doc,
        "queries_with_missing_source": queries_missing_source,
        "queries_with_missing_candidates": queries_missing_candidate,
        "queries_with_rerank_losses": queries_lost_rerank,
        "queries_with_cutoff_losses": queries_lost_cutoff,
        "queries_with_missing_chunks": queries_missing_chunks,
        "queries_with_evidence_missing_given_doc": queries_evidence_missing_given_doc,
    }
