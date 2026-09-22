from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key
import logging
from typing import Literal

from src.data.schemas import FusedHit, SearchHit
from src.retrieval.orchestrator import StreamResults

logger = logging.getLogger(__name__)


@dataclass
class _CandidateAccumulator:
    item_id: str
    item_type: Literal["document", "chunk"]
    doc_id: str
    language: str
    best_rank: int
    best_weight: float
    best_raw_score: float
    stream_ranks: dict[str, int] = field(default_factory=dict)
    fusion_score: float = 0.0

    @property
    def stream_count(self) -> int:
        return len(self.stream_ranks)

    @property
    def avg_rank(self) -> float:
        if not self.stream_ranks:
            return 0.0
        return sum(self.stream_ranks.values()) / len(self.stream_ranks)


def _compare_candidates(a: _CandidateAccumulator, b: _CandidateAccumulator) -> int:
    """Deterministic comparator for Weighted RRF candidates.

    Tie-breaking hierarchy:
    0. Fusion score descending (within float tolerance 1e-9).
    1. Number of streams containing item descending.
    2. Average rank across streams ascending (lower is better).
    3. Item ID lexicographically ascending.
    """
    score_diff = a.fusion_score - b.fusion_score
    if abs(score_diff) > 1e-9:
        return -1 if score_diff > 0 else 1

    # Tier 1: Number of streams containing the item (descending)
    if a.stream_count != b.stream_count:
        return -1 if a.stream_count > b.stream_count else 1

    # Tier 2: Average rank across streams where item appeared (ascending)
    rank_diff = a.avg_rank - b.avg_rank
    if abs(rank_diff) > 1e-9:
        return -1 if rank_diff < 0 else 1

    # Tier 3: String comparison on item_id (ascending)
    if a.item_id < b.item_id:
        return -1
    elif a.item_id > b.item_id:
        return 1
    return 0


def weighted_rrf(
    ranked_lists: dict[str, list[SearchHit]],
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
    top_k: int = 200,
) -> list[FusedHit]:
    """Fuse ranked candidate lists from multiple streams using Weighted Reciprocal Rank Fusion.

    Formula:
        RRF_Score(item) = sum_{s in S} (w_s / (k + rank_s(item)))

    Args:
        ranked_lists: Mapping from stream name to a list of SearchHit objects.
        weights: Optional mapping from stream name to stream weight (defaults to 1.0 per stream).
        rrf_k: RRF smoothing constant (default: 60).
        top_k: Maximum number of fused items to return (default: 200).

    Returns:
        A list of FusedHit objects sorted by fusion score and deterministic tie-breaking.
    """
    if not ranked_lists or top_k <= 0:
        return []

    candidates: dict[str, _CandidateAccumulator] = {}

    for stream_name, hits in ranked_lists.items():
        if not hits:
            continue

        weight = 1.0
        if weights is not None and stream_name in weights:
            weight = float(weights[stream_name])

        seen_in_stream: set[str] = set()

        for idx, hit in enumerate(hits):
            if hit is None or not hasattr(hit, "item_id"):
                continue

            item_id = str(hit.item_id)
            if item_id in seen_in_stream:
                # Retain the first (highest-ranking) appearance in this stream
                continue
            seen_in_stream.add(item_id)

            # Defensive rank clamping: ranks must be >= 1
            raw_rank = getattr(hit, "rank", None)
            if raw_rank is None or not isinstance(raw_rank, (int, float)):
                rank = idx + 1
            else:
                rank = max(1, int(raw_rank))

            raw_score = getattr(hit, "raw_score", 0.0)
            if raw_score is None:
                raw_score = 0.0

            denom = max(1.0, float(rrf_k + rank))
            score_contribution = weight / denom

            if item_id not in candidates:
                candidates[item_id] = _CandidateAccumulator(
                    item_id=item_id,
                    item_type=getattr(hit, "item_type", "document"),
                    doc_id=getattr(hit, "doc_id", None) or item_id,
                    language=getattr(hit, "language", ""),
                    best_rank=rank,
                    best_weight=weight,
                    best_raw_score=float(raw_score),
                    stream_ranks={stream_name: rank},
                    fusion_score=score_contribution,
                )
            else:
                cand = candidates[item_id]
                cand.stream_ranks[stream_name] = rank
                cand.fusion_score += score_contribution

                # Update language based on the highest ranked hit for that item
                is_better = False
                if rank < cand.best_rank:
                    is_better = True
                elif rank == cand.best_rank:
                    if weight > cand.best_weight:
                        is_better = True
                    elif weight == cand.best_weight and float(raw_score) > cand.best_raw_score:
                        is_better = True

                if is_better:
                    cand.best_rank = rank
                    cand.best_weight = weight
                    cand.best_raw_score = float(raw_score)
                    cand.language = getattr(hit, "language", cand.language)

    if not candidates:
        return []

    candidate_list = list(candidates.values())
    candidate_list.sort(key=cmp_to_key(_compare_candidates))

    return [
        FusedHit(
            item_id=c.item_id,
            item_type=c.item_type,
            doc_id=c.doc_id,
            language=c.language,
            fusion_score=c.fusion_score,
            stream_ranks=c.stream_ranks,
        )
        for c in candidate_list[:top_k]
    ]


def fuse_stream_results(
    stream_results: StreamResults,
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
    top_k_docs: int = 200,
    top_k_chunks: int = 200,
) -> tuple[list[FusedHit], list[FusedHit]]:
    """Helper to fuse documents and chunks from StreamResults independently.

    Args:
        stream_results: StreamResults container with doc_streams and chunk_streams.
        weights: Optional stream weights mapping.
        rrf_k: Smoothing constant.
        top_k_docs: Top K limit for documents.
        top_k_chunks: Top K limit for chunks.

    Returns:
        Tuple of (fused_docs, fused_chunks).
    """
    if stream_results is None:
        return [], []

    doc_streams = getattr(stream_results, "doc_streams", {}) or {}
    chunk_streams = getattr(stream_results, "chunk_streams", {}) or {}

    fused_docs = weighted_rrf(
        ranked_lists=doc_streams,
        weights=weights,
        rrf_k=rrf_k,
        top_k=top_k_docs,
    )
    fused_chunks = weighted_rrf(
        ranked_lists=chunk_streams,
        weights=weights,
        rrf_k=rrf_k,
        top_k=top_k_chunks,
    )

    return fused_docs, fused_chunks


class WeightedRRF:
    """Class wrapper providing stateful configuration for Weighted RRF."""

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.weights = weights
        self.rrf_k = rrf_k

    def fuse(
        self,
        ranked_lists: dict[str, list[SearchHit]],
        top_k: int = 200,
    ) -> list[FusedHit]:
        """Fuse candidate ranked lists."""
        return weighted_rrf(
            ranked_lists=ranked_lists,
            weights=self.weights,
            rrf_k=self.rrf_k,
            top_k=top_k,
        )

    def fuse_stream_results(
        self,
        stream_results: StreamResults,
        top_k_docs: int = 200,
        top_k_chunks: int = 200,
    ) -> tuple[list[FusedHit], list[FusedHit]]:
        """Fuse document and chunk streams from StreamResults."""
        return fuse_stream_results(
            stream_results=stream_results,
            weights=self.weights,
            rrf_k=self.rrf_k,
            top_k_docs=top_k_docs,
            top_k_chunks=top_k_chunks,
        )
