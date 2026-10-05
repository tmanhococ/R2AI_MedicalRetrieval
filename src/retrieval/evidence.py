from __future__ import annotations

import logging
from typing import Sequence, Any
import hashlib

from src.data.phase2_contracts import EvidenceConfig, EvidenceSpan
from src.data.schemas import FusedHit
from src.query.query_processor import QueryVariants
from src.data.vibiomir_contracts import SourceStore, SpanChunk, ExtractedDocument

logger = logging.getLogger(__name__)


def select_evidence(
    query: QueryVariants,
    docs: Sequence[FusedHit],
    store: SourceStore,
    vectors: Any | None = None,
    config: EvidenceConfig | None = None,
) -> list[EvidenceSpan]:
    """Recover local evidence inside candidate documents, expanding spans with neighboring blocks,
    binding source coordinates without synthetic text or translation artifacts, and resolving
    them through EvidenceSourceStore.
    """
    if config is None:
        config = EvidenceConfig()

    top_docs = list(docs)[: config.document_top_k]
    all_spans: list[EvidenceSpan] = []
    total_base_spans = 0

    # Extract query terms for lexical overlap scoring
    query_terms = set()
    for text in [getattr(query, "text_vi", None), getattr(query, "vi", None), getattr(query, "en", None), getattr(query, "zh", None)]:
        if text and isinstance(text, str):
            for word in text.lower().split():
                if len(word) > 2:
                    query_terms.add(word)

    for hit in top_docs:
        # Resolve doc_id
        try:
            doc_id = int(hit.doc_id)
        except (ValueError, TypeError):
            continue

        if not store.contains_official_doc(doc_id):
            continue

        try:
            doc = store.get_document(doc_id)
        except KeyError:
            continue

        # Load all chunks for doc
        try:
            chunks = list(store.iter_chunks_for_doc(doc_id))
        except Exception:
            # Fallback if iter_chunks_for_doc not available or fails
            chunks = []

        if not chunks:
            continue

        # Score chunks using lexical term overlap
        scored_chunks = []
        for chunk in chunks:
            chunk_text_lower = chunk.text.lower()
            score = sum(1.0 for term in query_terms if term in chunk_text_lower)
            # Add small tie breaker based on chunk index or length
            scored_chunks.append((score, chunk))

        # Sort descending by score, then ascending by chunk_index
        scored_chunks.sort(key=lambda x: (-x[0], x[1].chunk_index))

        # Select up to base_windows_per_doc distinct chunks
        selected_chunks = scored_chunks[: config.base_windows_per_doc]

        doc_base_spans = []
        for score, chunk in selected_chunks:
            if total_base_spans >= config.base_spans_per_query_max:
                break

            span_id = f"span:{doc_id}:{chunk.start_char}:{chunk.end_char}"
            base_span = EvidenceSpan(
                span_id=span_id,
                doc_id=doc_id,
                source_version=doc.source_version,
                source_sha256=doc.source_sha256,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
                seed_chunk_ids=(chunk.chunk_id,),
                variant_kind="base",
                score=score,
            )
            doc_base_spans.append(base_span)
            total_base_spans += 1

        # Build neighbor and expanded variants
        doc_all_variants = []
        for base_span in doc_base_spans:
            doc_all_variants.append(base_span)

            # Find neighboring block in chunks
            # Find index of base chunk in chunks
            base_chunk_idx = None
            for idx, c in enumerate(chunks):
                if c.start_char == base_span.start_char and c.end_char == base_span.end_char:
                    base_chunk_idx = idx
                    break

            if base_chunk_idx is not None:
                # Check neighbor after (or before)
                neighbor_indices = []
                if base_chunk_idx + 1 < len(chunks):
                    neighbor_indices.append(base_chunk_idx + 1)
                if base_chunk_idx - 1 >= 0 and not neighbor_indices:
                    neighbor_indices.append(base_chunk_idx - 1)

                for n_idx in neighbor_indices:
                    n_chunk = chunks[n_idx]
                    n_span_id = f"span:{doc_id}:{n_chunk.start_char}:{n_chunk.end_char}"
                    neighbor_span = EvidenceSpan(
                        span_id=n_span_id,
                        doc_id=doc_id,
                        source_version=doc.source_version,
                        source_sha256=doc.source_sha256,
                        start_char=n_chunk.start_char,
                        end_char=n_chunk.end_char,
                        seed_chunk_ids=(n_chunk.chunk_id,),
                        variant_kind="neighbor_block",
                        score=base_span.score,
                    )
                    doc_all_variants.append(neighbor_span)

                    # Expanded span
                    min_start = min(base_span.start_char, neighbor_span.start_char)
                    max_end = max(base_span.end_char, neighbor_span.end_char)
                    exp_span_id = f"span:{doc_id}:{min_start}:{max_end}"
                    expanded_span = EvidenceSpan(
                        span_id=exp_span_id,
                        doc_id=doc_id,
                        source_version=doc.source_version,
                        source_sha256=doc.source_sha256,
                        start_char=min_start,
                        end_char=max_end,
                        seed_chunk_ids=base_span.seed_chunk_ids + neighbor_span.seed_chunk_ids,
                        variant_kind="expanded",
                        base_span_id=base_span.span_id,
                        added_block_span_ids=(neighbor_span.span_id,),
                        score=base_span.score,
                    )
                    doc_all_variants.append(expanded_span)

        # Deduplicate by overlap >= config.dedup_span_overlap
        deduped = []
        for span in doc_all_variants:
            is_dup = False
            for existing in deduped:
                if existing.doc_id == span.doc_id:
                    overlap_start = max(existing.start_char, span.start_char)
                    overlap_end = min(existing.end_char, span.end_char)
                    if overlap_start < overlap_end:
                        intersection = overlap_end - overlap_start
                        union = max(existing.end_char, span.end_char) - min(existing.start_char, span.start_char)
                        if union > 0 and (intersection / union) >= config.dedup_span_overlap:
                            is_dup = True
                            break
            if not is_dup:
                deduped.append(span)

        all_spans.extend(deduped)

    return all_spans
