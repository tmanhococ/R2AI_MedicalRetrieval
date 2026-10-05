from __future__ import annotations

from collections import defaultdict
from typing import Sequence
from src.data.phase2_contracts import EvidenceSpan, OutputPolicy
from src.data.schemas import FinalPrediction


class DevLeakageChecker:
    def __init__(self, allowed_dev_query_ids: set[str] | None = None):
        self.allowed_dev_query_ids = allowed_dev_query_ids or set()

    def validate_query(self, query_id: str) -> None:
        if self.allowed_dev_query_ids and query_id not in self.allowed_dev_query_ids:
            raise ValueError(f"Test query ID leakage or unverified split: {query_id}")


def select_outputs(
    query_id: str,
    scored_spans: Sequence[EvidenceSpan],
    policy: OutputPolicy,
) -> FinalPrediction:
    # 1. Fallback & positive filtering
    if policy.fallback_policy == "heuristic_ce_positive":
        valid_spans = [
            s for s in scored_spans
            if s.score is not None and s.score > policy.fallback_raw_logit_min
        ]
    else:
        valid_spans = [s for s in scored_spans if s.score is not None]

    if not valid_spans:
        return FinalPrediction(
            query_id=query_id,
            document_ids=[],
            chunk_ids=[],
            document_scores=[],
            chunk_scores=[]
        )

    # 2. Suppress auxiliary neighbor_block spans by default
    non_neighbor_spans = [s for s in valid_spans if s.variant_kind != "neighbor_block"]

    # Map base span id -> base span and expanded span
    base_map = {s.span_id: s for s in non_neighbor_spans if s.variant_kind == "base"}
    expanded_map = {s.base_span_id: s for s in non_neighbor_spans if s.variant_kind == "expanded" and s.base_span_id}
    
    # Map neighbors by span or check positive neighbor for expansion
    neighbor_scores = {s.span_id: s.score for s in valid_spans if s.variant_kind == "neighbor_block"}

    selected_spans: list[EvidenceSpan] = []
    
    # Handle expansion rule
    processed_base_ids = set()
    for base_id, base_span in base_map.items():
        if base_id in expanded_map:
            exp_span = expanded_map[base_id]
            # Check expansion condition: expanded_span.score >= base_span.score - 0.5 (tolerance)
            # AND added neighboring block has at least one positive score (> 0.0)
            has_positive_neighbor = any(
                n_id in neighbor_scores and neighbor_scores[n_id] > 0.0
                for n_id in exp_span.added_block_span_ids
            )
            # If no added_block_span_ids stored explicitly, check if any neighbor overlaps or matches
            if not exp_span.added_block_span_ids and neighbor_scores:
                has_positive_neighbor = any(sc > 0.0 for sc in neighbor_scores.values())

            if exp_span.score >= base_span.score - 0.5 and has_positive_neighbor:
                selected_spans.append(exp_span)
                processed_base_ids.add(base_id)
            else:
                selected_spans.append(base_span)
                processed_base_ids.add(base_id)
        else:
            selected_spans.append(base_span)
            processed_base_ids.add(base_id)

    # Include any other non-neighbor spans not tied to base/expanded
    other_spans = [
        s for s in non_neighbor_spans
        if s.variant_kind not in ("base", "expanded")
    ]
    selected_spans.extend(other_spans)

    # 3. Deduplication per document (overlap >= 0.8 or identical offsets)
    doc_groups = defaultdict(list)
    for s in selected_spans:
        doc_groups[s.doc_id].append(s)

    final_doc_spans: list[EvidenceSpan] = []
    for doc_id, spans in doc_groups.items():
        # Sort by score descending
        spans_sorted = sorted(spans, key=lambda x: x.score if x.score is not None else float('-inf'), reverse=True)
        kept = []
        for s in spans_sorted:
            is_dup = False
            for k in kept:
                # check overlap or identical offsets
                if s.start_char == k.start_char and s.end_char == k.end_char:
                    is_dup = True
                    break
                intersection = max(0, min(s.end_char, k.end_char) - max(s.start_char, k.start_char))
                union = max(s.end_char, k.end_char) - min(s.start_char, k.start_char)
                if union > 0 and (intersection / union) >= 0.8:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(s)
        final_doc_spans.extend(kept)

    # 4. Document Ranking & Scoring (doc score = max evidence score)
    doc_max_scores: dict[int, float] = {}
    doc_spans_map = defaultdict(list)
    for s in final_doc_spans:
        doc_spans_map[s.doc_id].append(s)
        current_max = doc_max_scores.get(s.doc_id, float('-inf'))
        if s.score is not None and s.score > current_max:
            doc_max_scores[s.doc_id] = s.score

    ranked_docs = sorted(
        doc_max_scores.keys(),
        key=lambda d: doc_max_scores[d],
        reverse=True
    )
    if policy.document_cap > 0:
        ranked_docs = ranked_docs[:policy.document_cap]

    # 5. Evidence Span Ranking and per-doc cap
    final_evidence: list[EvidenceSpan] = []
    for d_id in ranked_docs:
        d_spans = doc_spans_map[d_id]
        d_spans_sorted = sorted(d_spans, key=lambda x: x.score if x.score is not None else float('-inf'), reverse=True)
        if policy.evidence_cap_per_doc > 0:
            d_spans_sorted = d_spans_sorted[:policy.evidence_cap_per_doc]
        final_evidence.extend(d_spans_sorted)

    # Global evidence cap
    if policy.evidence_cap > 0:
        final_evidence = final_evidence[:policy.evidence_cap]

    # Re-sort final evidence by score descending globally
    final_evidence = sorted(
        final_evidence,
        key=lambda x: x.score if x.score is not None else float('-inf'),
        reverse=True
    )

    # Build document ids and scores for surviving evidence only
    surviving_doc_ids_ordered = []
    seen_docs = set()
    for s in final_evidence:
        if s.doc_id not in seen_docs:
            seen_docs.add(s.doc_id)
            surviving_doc_ids_ordered.append(s.doc_id)

    document_ids = surviving_doc_ids_ordered
    document_scores = [doc_max_scores[d] for d in document_ids]
    chunk_ids = [s.span_id for s in final_evidence]
    chunk_scores = [s.score if s.score is not None else 0.0 for s in final_evidence]

    return FinalPrediction(
        query_id=query_id,
        document_ids=document_ids,
        chunk_ids=chunk_ids,
        document_scores=document_scores,
        chunk_scores=chunk_scores
    )
