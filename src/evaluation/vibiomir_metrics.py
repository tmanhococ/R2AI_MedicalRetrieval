"""ViBioMIR evaluation metrics engine.

Implements exact specification from sections 7.2 and 7.3 of the implementation plan:
- Independent many-to-many boolean matching matrix (no 1-to-1 matching, no Hungarian algorithm).
- Correct predicted chunks = count of prediction rows with >= 1 match.
- Found reference chunks = count of reference columns with >= 1 match.
- Chunk P = correct_predictions / total_predictions.
- Chunk R = found_references / total_references.
- Documents level evaluated on unique document ID sets.
- Macro F2 is the arithmetic mean of per-query F2 scores (NOT F2 of macro P and macro R).
- Inferred final mean = (macro_documents.f2 + macro_chunks.f2) / 2.0.
- Missing queries in predictions are retained as empty predictions (|P| == 0).
- Empty/empty edge cases handled strictly per EvaluationProfile specification.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from fractions import Fraction
from typing import Any

from src.data.vibiomir_contracts import (
    EvaluationProfile,
    EvaluationRecord,
    EvaluationSummary,
    LevelMetrics,
    MacroMetrics,
    PerQueryMetrics,
    QueryMetrics,
    ReferenceChunk,
    SubmissionRow,
)
from src.evaluation.metrics import calculate_f2
from src.evaluation.vibiomir_text import (
    TextMatchProfile,
    TextOverlapEvaluator,
)

PROVISIONAL_SPEC_V1_PROFILE = EvaluationProfile(
    name="provisional_spec_v1",
    normalization_version="btc_order_v1",
    tokenizer_name="whitespace_v1",
    threshold=Fraction(2, 5),
    undefined_pr=0,
    empty_both_f2=1,
    scorer_parity_verified=False,
)

EMPTY_BOTH_ZERO_PROFILE = EvaluationProfile(
    name="empty_both_zero",
    normalization_version="btc_order_v1",
    tokenizer_name="whitespace_v1",
    threshold=Fraction(2, 5),
    undefined_pr=0,
    empty_both_f2=0,
    scorer_parity_verified=False,
)


def evaluate_query(
    query_id: int,
    prediction: SubmissionRow | dict[str, Any],
    gold: EvaluationRecord,
    profile: EvaluationProfile,
    evaluator: TextOverlapEvaluator | None = None,
) -> QueryMetrics:
    """Evaluate a single query against ground truth for documents and chunks.

    Args:
        query_id: Target query integer ID.
        prediction: Predicted submission row (relevant_docs and relevant_chunks).
        gold: Ground truth EvaluationRecord for the query.
        profile: EvaluationProfile defining thresholds and empty-case rules.
        evaluator: Optional TextOverlapEvaluator instance for token/match caching.

    Returns:
        QueryMetrics containing LevelMetrics for documents and chunks, plus trace indices.
    """
    if gold.query_id != query_id:
        raise ValueError(
            f"Gold query_id ({gold.query_id}) does not match evaluation query_id ({query_id})"
        )

    # Validate gold reference chunks
    for ref in gold.relevant_chunks:
        if not ref.chunk_text or not ref.chunk_text.strip():
            raise ValueError(
                f"Gold reference chunk in query {query_id} has empty text: {ref!r}"
            )

    # 1. Document Level Evaluation
    pred_docs = prediction.get("relevant_docs", [])
    pred_doc_set = set(pred_docs)
    gold_doc_set = set(gold.relevant_doc_ids)

    pred_doc_count = len(pred_doc_set)
    gold_doc_count = len(gold_doc_set)
    common_docs = pred_doc_set & gold_doc_set
    correct_pred_doc_count = len(common_docs)
    found_gold_doc_count = len(common_docs)

    if pred_doc_count == 0 and gold_doc_count == 0:
        doc_p = float(profile.undefined_pr)
        doc_r = float(profile.undefined_pr)
        doc_f2 = float(profile.empty_both_f2)
    elif pred_doc_count == 0:
        doc_p = float(profile.undefined_pr)
        doc_r = 0.0
        doc_f2 = 0.0
    elif gold_doc_count == 0:
        doc_p = 0.0
        doc_r = float(profile.undefined_pr)
        doc_f2 = 0.0
    else:
        doc_p = correct_pred_doc_count / pred_doc_count
        doc_r = found_gold_doc_count / gold_doc_count
        doc_f2 = calculate_f2(precision=doc_p, recall=doc_r, beta=2.0)

    doc_metrics = LevelMetrics(
        precision=doc_p,
        recall=doc_r,
        f2=doc_f2,
        prediction_count=pred_doc_count,
        reference_count=gold_doc_count,
        correct_prediction_count=correct_pred_doc_count,
        found_reference_count=found_gold_doc_count,
    )

    # 2. Chunk Level Evaluation (Many-to-Many Matching Matrix)
    pred_chunks = prediction.get("relevant_chunks", [])
    gold_chunks = gold.relevant_chunks

    num_pred_chunks = len(pred_chunks)
    num_gold_chunks = len(gold_chunks)

    if num_pred_chunks == 0 and num_gold_chunks == 0:
        chunk_p = float(profile.undefined_pr)
        chunk_r = float(profile.undefined_pr)
        chunk_f2 = float(profile.empty_both_f2)
        correct_pred_chunks = 0
        found_gold_chunks = 0
        matched_ref_indices: tuple[int, ...] = ()
        matched_pred_indices: tuple[int, ...] = ()
    elif num_pred_chunks == 0:
        chunk_p = float(profile.undefined_pr)
        chunk_r = 0.0
        chunk_f2 = 0.0
        correct_pred_chunks = 0
        found_gold_chunks = 0
        matched_ref_indices = ()
        matched_pred_indices = ()
    elif num_gold_chunks == 0:
        chunk_p = 0.0
        chunk_r = float(profile.undefined_pr)
        chunk_f2 = 0.0
        correct_pred_chunks = 0
        found_gold_chunks = 0
        matched_ref_indices = ()
        matched_pred_indices = ()
    else:
        if evaluator is None:
            text_profile = TextMatchProfile(
                tokenizer_name=profile.tokenizer_name,
                normalization_version=profile.normalization_version,
                overlap_threshold=profile.threshold,
            )
            evaluator = TextOverlapEvaluator(profile=text_profile)

        # M[i, j]: row i is pred chunk, col j is gold chunk
        row_has_match = [False] * num_pred_chunks
        col_has_match = [False] * num_gold_chunks

        for i, p_chunk in enumerate(pred_chunks):
            p_doc_id = (
                p_chunk["doc_id"]
                if isinstance(p_chunk, dict)
                else getattr(p_chunk, "doc_id")
            )
            p_text = (
                p_chunk["chunk_text"]
                if isinstance(p_chunk, dict)
                else getattr(p_chunk, "chunk_text")
            )

            for j, g_chunk in enumerate(gold_chunks):
                # Must belong to the exact same document ID
                if p_doc_id == g_chunk.doc_id:
                    if evaluator.matches(p_text, g_chunk.chunk_text):
                        row_has_match[i] = True
                        col_has_match[j] = True

        correct_pred_chunks = sum(row_has_match)
        found_gold_chunks = sum(col_has_match)
        chunk_p = correct_pred_chunks / num_pred_chunks
        chunk_r = found_gold_chunks / num_gold_chunks
        chunk_f2 = calculate_f2(precision=chunk_p, recall=chunk_r, beta=2.0)
        matched_ref_indices = tuple(
            j for j, matched in enumerate(col_has_match) if matched
        )
        matched_pred_indices = tuple(
            i for i, matched in enumerate(row_has_match) if matched
        )

    chunk_metrics = LevelMetrics(
        precision=chunk_p,
        recall=chunk_r,
        f2=chunk_f2,
        prediction_count=num_pred_chunks,
        reference_count=num_gold_chunks,
        correct_prediction_count=correct_pred_chunks,
        found_reference_count=found_gold_chunks,
    )

    return QueryMetrics(
        query_id=query_id,
        documents=doc_metrics,
        chunks=chunk_metrics,
        matched_reference_indices=matched_ref_indices,
        matched_prediction_indices=matched_pred_indices,
    )


def evaluate_dataset(
    predictions: Iterable[SubmissionRow | dict[str, Any]],
    gold: Iterable[EvaluationRecord],
    query_ids: Sequence[int],
    profile: EvaluationProfile,
) -> EvaluationSummary:
    """Evaluate an entire dataset across official query IDs.

    Args:
        predictions: Collection of SubmissionRow predictions.
        gold: Collection of Ground Truth EvaluationRecord instances.
        query_ids: Canonical sequence of query IDs defining the evaluation split.
        profile: EvaluationProfile governing evaluation parameters.

    Returns:
        EvaluationSummary containing macro statistics across documents and chunks.
    """
    text_profile = TextMatchProfile(
        tokenizer_name=profile.tokenizer_name,
        normalization_version=profile.normalization_version,
        overlap_threshold=profile.threshold,
    )
    evaluator = TextOverlapEvaluator(profile=text_profile)

    gold_map = {g.query_id: g for g in gold}
    pred_map = {p["id"]: p for p in predictions}

    per_query_list: list[QueryMetrics] = []

    for qid in query_ids:
        if qid not in gold_map:
            raise KeyError(f"Query {qid} missing in gold evaluation records")

        # Missing query in predictions treated as empty prediction (|P| == 0)
        pred_row = pred_map.get(
            qid, {"id": qid, "relevant_docs": [], "relevant_chunks": []}
        )
        qm = evaluate_query(
            query_id=qid,
            prediction=pred_row,
            gold=gold_map[qid],
            profile=profile,
            evaluator=evaluator,
        )
        per_query_list.append(qm)

    query_count = len(query_ids)
    if query_count == 0:
        macro_docs = MacroMetrics(precision=0.0, recall=0.0, f2=0.0)
        macro_chunks = MacroMetrics(precision=0.0, recall=0.0, f2=0.0)
        inferred_final_mean = 0.0
    else:
        macro_doc_p = sum(q.documents.precision for q in per_query_list) / query_count
        macro_doc_r = sum(q.documents.recall for q in per_query_list) / query_count
        macro_doc_f2 = sum(q.documents.f2 for q in per_query_list) / query_count

        macro_chunk_p = sum(q.chunks.precision for q in per_query_list) / query_count
        macro_chunk_r = sum(q.chunks.recall for q in per_query_list) / query_count
        macro_chunk_f2 = sum(q.chunks.f2 for q in per_query_list) / query_count

        macro_docs = MacroMetrics(
            precision=macro_doc_p,
            recall=macro_doc_r,
            f2=macro_doc_f2,
        )
        macro_chunks = MacroMetrics(
            precision=macro_chunk_p,
            recall=macro_chunk_r,
            f2=macro_chunk_f2,
        )
        inferred_final_mean = (macro_doc_f2 + macro_chunk_f2) / 2.0

    warnings: list[str] = []
    diagnostics: dict[str, Any] = {}
    if not profile.scorer_parity_verified:
        warnings.append(
            f"Scorer parity unverified for profile '{profile.name}'; metrics are provisional."
        )
        diagnostics["scorer_parity_verified"] = False
        diagnostics["profile_name"] = profile.name

    return EvaluationSummary(
        profile_fingerprint=profile.profile_fingerprint,
        scorer_parity_verified=profile.scorer_parity_verified,
        query_count=query_count,
        macro_documents=macro_docs,
        macro_chunks=macro_chunks,
        per_query_metrics=PerQueryMetrics(per_query_list),
        inferred_final_mean=inferred_final_mean,
        warnings=tuple(warnings),
        diagnostics=diagnostics,
    )
