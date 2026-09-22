from __future__ import annotations

import logging
from typing import Any, Sequence

from src.data.schemas import ScoredHit
from src.reranking import min_max_normalize

logger = logging.getLogger(__name__)


def _normalize_or_pass_scores(
    hits: Sequence[ScoredHit],
    mode: bool | str = "auto",
) -> list[float]:
    """Extract and normalize candidate scores if needed.

    - "auto": Preserves scores if already in [0.0, 1.0]. If unpopulated (all 0.0)
      with non-zero rerank_scores, or if any score is outside [0.0, 1.0], runs min_max_normalize.
    - True: Forces min_max_normalize.
    - False: Passes scores as-is.
    """
    if not hits:
        return []

    raw_scores = [float(h.normalized_score) for h in hits]
    if all(s == 0.0 for s in raw_scores) and any(float(h.rerank_score) != 0.0 for h in hits):
        raw_scores = [float(h.rerank_score) for h in hits]
        return min_max_normalize(raw_scores)

    if mode is True:
        return min_max_normalize(raw_scores)
    elif mode is False:
        return raw_scores
    else:  # "auto"
        if any(s < 0.0 or s > 1.0 for s in raw_scores):
            return min_max_normalize(raw_scores)
        return raw_scores


def compute_joint_scores(
    doc_hits: Sequence[ScoredHit],
    chunk_hits: Sequence[ScoredHit],
    *,
    doc_base_weight: float = 0.70,
    doc_best_chunk_weight: float = 0.20,
    doc_top_chunks_mean_weight: float = 0.10,
    top_m_chunks: int = 3,
    chunk_base_weight: float = 0.85,
    chunk_parent_doc_weight: float = 0.15,
    normalize_inputs: bool | str = "auto",
    parent_doc_fallback: float = 0.0,
) -> tuple[list[ScoredHit], list[ScoredHit]]:
    """Compute bidirectional joint scores for documents and chunks.

    Formulas:
        1. final_doc_score(d) = alpha * base_doc_score(d)
                              + beta * max_{c in d}(base_chunk_score(c))
                              + gamma * mean(top_m_chunk_scores(d))
           (defaults: alpha=0.70, beta=0.20, gamma=0.10, m=3)

        2. final_chunk_score(c) = delta * base_chunk_score(c)
                                + epsilon * final_doc_score(parent_doc(c))
           (defaults: delta=0.85, epsilon=0.15)

    Edge cases:
        - If document d has no matching chunks: chunk evidence is 0.0.
        - If document d has k < m chunks: mean is computed across all k available chunks.
        - If chunk parent doc is missing: parent doc score defaults to parent_doc_fallback (0.0).

    Returns:
        tuple[list[ScoredHit], list[ScoredHit]]: Joint-scored documents and chunks,
        sorted descending by (-normalized_score, -raw_fusion_score, item_id).
    """
    if not doc_hits and not chunk_hits:
        return [], []

    # 1. Base Score Normalization
    base_doc_scores = _normalize_or_pass_scores(doc_hits, mode=normalize_inputs)
    base_chunk_scores = _normalize_or_pass_scores(chunk_hits, mode=normalize_inputs)

    # Index chunks by parent doc_id
    chunks_by_doc: dict[str, list[float]] = {}
    for chunk, chunk_score in zip(chunk_hits, base_chunk_scores):
        chunks_by_doc.setdefault(chunk.doc_id, []).append(chunk_score)

    # 2. Compute Final Document Scores
    final_doc_scores: list[float] = []
    final_doc_score_map: dict[str, float] = {}

    for doc, base_doc_score in zip(doc_hits, base_doc_scores):
        matching_chunk_scores = (
            chunks_by_doc.get(doc.doc_id)
            or chunks_by_doc.get(doc.item_id)
            or []
        )

        if matching_chunk_scores:
            best_chunk_score = max(matching_chunk_scores)
            sorted_chunks = sorted(matching_chunk_scores, reverse=True)
            top_k_chunks = sorted_chunks[:top_m_chunks]
            top_chunks_mean = sum(top_k_chunks) / len(top_k_chunks)
        else:
            best_chunk_score = 0.0
            top_chunks_mean = 0.0

        final_doc_score = (
            doc_base_weight * base_doc_score
            + doc_best_chunk_weight * best_chunk_score
            + doc_top_chunks_mean_weight * top_chunks_mean
        )
        final_doc_scores.append(final_doc_score)
        final_doc_score_map[doc.doc_id] = final_doc_score
        final_doc_score_map[doc.item_id] = final_doc_score

    # 3. Compute Final Chunk Scores
    final_chunk_scores: list[float] = []
    for chunk, base_chunk_score in zip(chunk_hits, base_chunk_scores):
        parent_score = final_doc_score_map.get(chunk.doc_id, parent_doc_fallback)
        final_chunk_score = (
            chunk_base_weight * base_chunk_score
            + chunk_parent_doc_weight * parent_score
        )
        final_chunk_scores.append(final_chunk_score)

    # 4. Construct ScoredHit objects
    joint_docs = [
        ScoredHit(
            item_id=d.item_id,
            item_type=d.item_type,
            doc_id=d.doc_id,
            language=d.language,
            rerank_score=d.rerank_score,
            normalized_score=score,
            raw_fusion_score=d.raw_fusion_score,
        )
        for d, score in zip(doc_hits, final_doc_scores)
    ]
    joint_docs.sort(key=lambda h: (-h.normalized_score, -h.raw_fusion_score, h.item_id))

    joint_chunks = [
        ScoredHit(
            item_id=c.item_id,
            item_type=c.item_type,
            doc_id=c.doc_id,
            language=c.language,
            rerank_score=c.rerank_score,
            normalized_score=score,
            raw_fusion_score=c.raw_fusion_score,
        )
        for c, score in zip(chunk_hits, final_chunk_scores)
    ]
    joint_chunks.sort(key=lambda h: (-h.normalized_score, -h.raw_fusion_score, h.item_id))

    return joint_docs, joint_chunks


class JointScorer:
    """Configurable Joint Document-Chunk Scorer.

    Allows encapsulating scoring hyperparameters and running joint scoring
    on query candidate pools.
    """

    def __init__(
        self,
        doc_base_weight: float = 0.70,
        doc_best_chunk_weight: float = 0.20,
        doc_top_chunks_mean_weight: float = 0.10,
        top_m_chunks: int = 3,
        chunk_base_weight: float = 0.85,
        chunk_parent_doc_weight: float = 0.15,
        normalize_inputs: bool | str = "auto",
        parent_doc_fallback: float = 0.0,
    ) -> None:
        self.doc_base_weight = doc_base_weight
        self.doc_best_chunk_weight = doc_best_chunk_weight
        self.doc_top_chunks_mean_weight = doc_top_chunks_mean_weight
        self.top_m_chunks = top_m_chunks
        self.chunk_base_weight = chunk_base_weight
        self.chunk_parent_doc_weight = chunk_parent_doc_weight
        self.normalize_inputs = normalize_inputs
        self.parent_doc_fallback = parent_doc_fallback

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> JointScorer:
        """Instantiate JointScorer from application config mapping."""
        joint_cfg = config.get("joint_scoring", config)
        doc_cfg = joint_cfg.get("document", {})
        chunk_cfg = joint_cfg.get("chunk", {})

        return cls(
            doc_base_weight=float(doc_cfg.get("base_weight", 0.70)),
            doc_best_chunk_weight=float(doc_cfg.get("best_chunk_weight", 0.20)),
            doc_top_chunks_mean_weight=float(doc_cfg.get("top_chunks_mean_weight", 0.10)),
            top_m_chunks=int(doc_cfg.get("top_m_chunks", 3)),
            chunk_base_weight=float(chunk_cfg.get("base_weight", 0.85)),
            chunk_parent_doc_weight=float(chunk_cfg.get("parent_document_weight", 0.15)),
        )

    def score(
        self,
        doc_hits: Sequence[ScoredHit],
        chunk_hits: Sequence[ScoredHit],
    ) -> tuple[list[ScoredHit], list[ScoredHit]]:
        """Run joint scoring on document and chunk candidates."""
        return compute_joint_scores(
            doc_hits=doc_hits,
            chunk_hits=chunk_hits,
            doc_base_weight=self.doc_base_weight,
            doc_best_chunk_weight=self.doc_best_chunk_weight,
            doc_top_chunks_mean_weight=self.doc_top_chunks_mean_weight,
            top_m_chunks=self.top_m_chunks,
            chunk_base_weight=self.chunk_base_weight,
            chunk_parent_doc_weight=self.chunk_parent_doc_weight,
            normalize_inputs=self.normalize_inputs,
            parent_doc_fallback=self.parent_doc_fallback,
        )

    def __call__(
        self,
        doc_hits: Sequence[ScoredHit],
        chunk_hits: Sequence[ScoredHit],
    ) -> tuple[list[ScoredHit], list[ScoredHit]]:
        """Callable shorthand for self.score()."""
        return self.score(doc_hits, chunk_hits)
