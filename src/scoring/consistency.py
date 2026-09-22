from __future__ import annotations

from enum import Enum
import logging
from typing import Sequence

from src.data.schemas import ScoredHit

logger = logging.getLogger(__name__)


class ConsistencyMode(str, Enum):
    """Supported structural consistency post-processing modes."""

    INDEPENDENT = "independent"
    ENFORCE_PARENT_DOC = "enforce_parent_doc"
    FILTER_ORPHAN_CHUNKS = "filter_orphan_chunks"


def _get_parent_doc_id(chunk: str | ScoredHit, chunk_to_doc: dict[str, str] | None) -> str:
    """Extract parent document id for a chunk."""
    if isinstance(chunk, ScoredHit):
        if chunk_to_doc and chunk.item_id in chunk_to_doc:
            return chunk_to_doc[chunk.item_id]
        return chunk.doc_id
    else:  # str ID
        if chunk_to_doc and chunk in chunk_to_doc:
            return chunk_to_doc[chunk]
        # Standard format heuristic fallback: docId_c<index>
        if "_c" in chunk:
            return chunk.rsplit("_c", 1)[0]
        return chunk


def apply_structural_consistency(
    selected_docs: Sequence[str] | Sequence[ScoredHit],
    selected_chunks: Sequence[str] | Sequence[ScoredHit],
    chunk_to_doc: dict[str, str] | None = None,
    mode: str | ConsistencyMode = ConsistencyMode.INDEPENDENT,
    available_docs: Sequence[ScoredHit] | None = None,
) -> tuple[list[str] | list[ScoredHit], list[str] | list[ScoredHit]]:
    """Apply structural consistency post-processing between documents and chunks.

    Modes:
        - "independent": No structural modification. Returns unique docs and chunks.
        - "enforce_parent_doc": For each selected chunk, if its parent doc_id is missing
          from selected_docs, appends parent doc_id to selected_docs (preserving existing
          order and deterministically appending missing docs at the end).
        - "filter_orphan_chunks": For each selected chunk, if its parent doc_id is not
          present in selected_docs, filters out that chunk.

    Robustness:
        - Supports both list[str] IDs and list[ScoredHit] objects.
        - Preserves uniqueness (no duplicate document or chunk IDs).
        - If chunk_to_doc is None and chunks are ScoredHit, extracts hit.doc_id.
        - In enforce_parent_doc with ScoredHit, restores hit from available_docs if provided,
          otherwise synthesizes a valid ScoredHit with 0.0 scores.
    """
    # Normalize mode string
    if isinstance(mode, ConsistencyMode):
        mode_val = mode.value
    else:
        mode_val = str(mode).strip().lower()

    valid_modes = {m.value for m in ConsistencyMode}
    if mode_val not in valid_modes:
        raise ValueError(
            f"Unknown consistency mode: '{mode}'. Supported modes: {sorted(valid_modes)}"
        )

    # Fast path for empty inputs
    if not selected_docs and not selected_chunks:
        return [], []

    # Detect input types
    docs_are_hits = any(isinstance(d, ScoredHit) for d in selected_docs)
    chunks_are_hits = any(isinstance(c, ScoredHit) for c in selected_chunks)
    treat_as_hits = docs_are_hits or (not selected_docs and chunks_are_hits)

    # 1. Deduplicate selected_docs preserving order
    seen_doc_ids: set[str] = set()
    unique_docs: list[Any] = []
    for d in selected_docs:
        doc_id = d.doc_id if isinstance(d, ScoredHit) else str(d)
        if doc_id not in seen_doc_ids:
            seen_doc_ids.add(doc_id)
            unique_docs.append(d)

    # 2. Deduplicate selected_chunks preserving order
    seen_chunk_ids: set[str] = set()
    unique_chunks: list[Any] = []
    for c in selected_chunks:
        chunk_id = c.item_id if isinstance(c, ScoredHit) else str(c)
        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            unique_chunks.append(c)

    # 3. Apply selected mode
    if mode_val == ConsistencyMode.INDEPENDENT.value:
        return unique_docs, unique_chunks

    elif mode_val == ConsistencyMode.ENFORCE_PARENT_DOC.value:
        # Build available_docs map if available
        avail_map: dict[str, ScoredHit] = {}
        if available_docs:
            for ad in available_docs:
                if ad.doc_id not in avail_map:
                    avail_map[ad.doc_id] = ad
                if ad.item_id not in avail_map:
                    avail_map[ad.item_id] = ad

        for chunk in unique_chunks:
            parent_doc_id = _get_parent_doc_id(chunk, chunk_to_doc)
            if parent_doc_id not in seen_doc_ids:
                seen_doc_ids.add(parent_doc_id)
                if treat_as_hits:
                    if parent_doc_id in avail_map:
                        unique_docs.append(avail_map[parent_doc_id])
                    else:
                        lang = chunk.language if isinstance(chunk, ScoredHit) else "vi"
                        synth_hit = ScoredHit(
                            item_id=parent_doc_id,
                            item_type="document",
                            doc_id=parent_doc_id,
                            language=lang,
                            rerank_score=0.0,
                            normalized_score=0.0,
                            raw_fusion_score=0.0,
                        )
                        unique_docs.append(synth_hit)
                else:
                    unique_docs.append(parent_doc_id)

        return unique_docs, unique_chunks

    elif mode_val == ConsistencyMode.FILTER_ORPHAN_CHUNKS.value:
        final_chunks: list[Any] = []
        for chunk in unique_chunks:
            parent_doc_id = _get_parent_doc_id(chunk, chunk_to_doc)
            if parent_doc_id in seen_doc_ids:
                final_chunks.append(chunk)

        return unique_docs, final_chunks

    # Fallback safety (should never reach here due to validation above)
    return unique_docs, unique_chunks
