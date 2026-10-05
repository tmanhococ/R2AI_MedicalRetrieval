from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.data.schemas import SearchHit, FusedHit
from src.data.phase2_contracts import DenseConfig, SparseConfig
from src.indexing.lucene import LuceneSparseBackend
from src.indexing.faiss_index import FaissDenseBackend
from src.indexing.dense_encoder import DenseEncoder
from src.query.query_processor import QueryVariants

logger = logging.getLogger(__name__)

@dataclass
class HybridCandidateRetriever:
    """Multilingual hybrid candidate retriever implementing:
    - Sparse search pools (top 500 docs per language, top 1000 chunks per language with doc aggregation).
    - Global dense search with query windowing (splitting queries > query_window_tokens into windows with overlap 64, grouping variant scores).
    - Weighted RRF fusion across sparse_doc, sparse_chunk, and dense families.
    """

    sparse_backend: Any = None
    dense_backend: Any = None
    dense_encoder: Any = None
    sparse_config: SparseConfig = field(default_factory=SparseConfig)
    dense_config: DenseConfig = field(default_factory=DenseConfig)
    rrf_k: int = 60
    family_weights: dict[str, float] = field(
        default_factory=lambda: {"sparse_doc": 1.0, "sparse_chunk": 0.75, "dense": 1.0}
    )
    candidate_docs_top_k: int = 200

    def _split_into_windows(self, query: str, window_tokens: int = 448, overlap: int = 64) -> list[str]:
        """Split a query into overlapping token windows for dense search."""
        if not query or not query.strip():
            return []
        words = query.strip().split()
        if len(words) <= window_tokens:
            return [query.strip()]
        
        windows = []
        step = window_tokens - overlap
        if step <= 0:
            step = window_tokens
        
        start = 0
        while start < len(words):
            end = start + window_tokens
            window_text = " ".join(words[start:end])
            if window_text:
                windows.append(window_text)
            if end >= len(words):
                break
            start += step
        return windows

    def retrieve(
        self,
        query_variants: QueryVariants,
        top_k: int = 200,
    ) -> tuple[list[FusedHit], dict[str, list[SearchHit]]]:
        """Retrieve candidate documents and evidence seeds for given query variants."""
        top_k_docs = self.sparse_config.top_docs_per_lang if hasattr(self.sparse_config, "top_docs_per_lang") else 500
        top_k_chunks = self.sparse_config.top_chunks_per_lang if hasattr(self.sparse_config, "top_chunks_per_lang") else 1000

        sparse_doc_hits_all: list[SearchHit] = []
        sparse_chunk_hits_all: list[SearchHit] = []
        dense_chunk_hits_all: list[SearchHit] = []

        lang_queries: dict[str, str] = {}
        if query_variants.vi and query_variants.vi.strip():
            lang_queries["vi"] = query_variants.vi.strip()
            lang_queries["und"] = query_variants.vi.strip()
        if query_variants.en and query_variants.en.strip():
            lang_queries["en"] = query_variants.en.strip()
        if query_variants.zh and query_variants.zh.strip():
            lang_queries["zh"] = query_variants.zh.strip()

        if self.sparse_backend is not None:
            for lang, q_text in lang_queries.items():
                try:
                    doc_hits = self.sparse_backend.search_documents(query=q_text, language=lang, top_k=top_k_docs)
                    if doc_hits:
                        for h in doc_hits:
                            sparse_doc_hits_all.append(h)
                except Exception as exc:
                    logger.warning("Sparse doc search failed for lang %s: %s", lang, exc)

                try:
                    chunk_hits = self.sparse_backend.search_chunks(query=q_text, language=lang, top_k=top_k_chunks)
                    if chunk_hits:
                        for h in chunk_hits:
                            sparse_chunk_hits_all.append(h)
                except Exception as exc:
                    logger.warning("Sparse chunk search failed for lang %s: %s", lang, exc)

        window_tokens = getattr(self.dense_config, "query_window_tokens", 448)
        if query_variants.vi and query_variants.vi.strip() and self.dense_backend is not None:
            vi_text = query_variants.vi.strip()
            windows = self._split_into_windows(vi_text, window_tokens=window_tokens, overlap=64)
            
            chunk_best_scores: dict[str, float] = {}
            chunk_best_hits: dict[str, SearchHit] = {}

            for w_text in windows:
                try:
                    dense_hits = self.dense_backend.search(query=w_text, top_k=1000)
                    if dense_hits:
                        for dh in dense_hits:
                            cid = dh.item_id
                            score = dh.score
                            if cid not in chunk_best_scores or score > chunk_best_scores[cid]:
                                chunk_best_scores[cid] = score
                                chunk_best_hits[cid] = dh
                except Exception as exc:
                    logger.warning("Dense search failed for window: %s", exc)

            dense_chunk_hits_all = sorted(chunk_best_hits.values(), key=lambda x: x.score, reverse=True)[:1000]

        def extract_doc_ranking(hits: list[SearchHit]) -> list[str]:
            seen = set()
            ranked_docs = []
            sorted_hits = sorted(hits, key=lambda x: x.score, reverse=True)
            for h in sorted_hits:
                did = str(h.doc_id) if hasattr(h, "doc_id") else h.item_id
                if did not in seen:
                    seen.add(did)
                    ranked_docs.append(did)
            return ranked_docs

        sparse_doc_ranked = extract_doc_ranking(sparse_doc_hits_all)
        sparse_chunk_ranked = extract_doc_ranking(sparse_chunk_hits_all)
        dense_doc_ranked = extract_doc_ranking(dense_chunk_hits_all)

        rrf_scores: dict[str, float] = {}
        stream_ranks_map: dict[str, dict[str, int]] = {}

        families = [
            ("sparse_doc", sparse_doc_ranked, self.family_weights.get("sparse_doc", 1.0)),
            ("sparse_chunk", sparse_chunk_ranked, self.family_weights.get("sparse_chunk", 0.75)),
            ("dense", dense_doc_ranked, self.family_weights.get("dense", 1.0)),
        ]

        for fam_name, ranked_list, weight in families:
            for rank_idx, did in enumerate(ranked_list, start=1):
                if did not in stream_ranks_map:
                    stream_ranks_map[did] = {}
                stream_ranks_map[did][fam_name] = rank_idx
                
                score_contrib = weight * (1.0 / (self.rrf_k + rank_idx))
                rrf_scores[did] = rrf_scores.get(did, 0.0) + score_contrib

        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        fused_hits: list[FusedHit] = []
        for rank_pos, (did, score) in enumerate(sorted_docs, start=1):
            fused_hits.append(
                FusedHit(
                    item_id=did,
                    item_type="document",
                    doc_id=did,
                    language="vi",
                    fusion_score=score,
                    stream_ranks=stream_ranks_map.get(did, {}),
                )
            )

        evidence_seeds = {
            "sparse_doc": sparse_doc_hits_all,
            "sparse_chunk": sparse_chunk_hits_all,
            "dense": dense_chunk_hits_all,
        }

        return fused_hits, evidence_seeds
