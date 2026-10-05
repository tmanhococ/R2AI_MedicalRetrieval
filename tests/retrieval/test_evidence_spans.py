from __future__ import annotations

import pytest
from pathlib import Path

from src.data.phase2_contracts import EvidenceConfig, EvidenceSpan
from src.data.vibiomir_contracts import (
    ExtractedDocument,
    ExtractedBlock,
    SpanChunk,
)
from src.data.schemas import FusedHit
from src.query.query_processor import QueryVariants
from src.data.source_store import EvidenceSourceStore
from src.retrieval.evidence import select_evidence


class MockSourceStore:
    def __init__(self, doc: ExtractedDocument, chunks: list[SpanChunk]) -> None:
        self.doc = doc
        self.chunks = chunks

    def contains_official_doc(self, doc_id: int) -> bool:
        return doc_id == self.doc.doc_id

    def get_document(self, doc_id: int) -> ExtractedDocument:
        if doc_id == self.doc.doc_id:
            return self.doc
        raise KeyError(f"Doc {doc_id} not found")

    def get_documents(self, doc_ids: list[int]) -> list[ExtractedDocument]:
        return [self.get_document(d) for d in doc_ids]

    def iter_chunks_for_doc(self, doc_id: int):
        if doc_id == self.doc.doc_id:
            for c in self.chunks:
                yield c

    def get_chunk(self, chunk_id: str) -> SpanChunk:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        raise KeyError(f"Chunk {chunk_id} not found")

    def get_chunks(self, chunk_ids: list[str]) -> list[SpanChunk]:
        return [self.get_chunk(cid) for cid in chunk_ids]


def test_evidence_span_selection_and_expansion():
    text = "Introduction section text about diabetes management. Methods section discussing insulin therapy. Results section showing blood glucose reduction."
    blocks = (
        ExtractedBlock(block_index=0, kind="text", start_char=0, end_char=52, section="Introduction"),
        ExtractedBlock(block_index=1, kind="text", start_char=53, end_char=103, section="Methods"),
        ExtractedBlock(block_index=2, kind="text", start_char=104, end_char=143, section="Results"),
    )
    doc = ExtractedDocument(
        doc_id=101,
        source_version="v1",
        source_text=text,
        source_sha256="abc123sha",
        raw_sha256="raw123",
        title="Medical Study",
        language="vi",
        blocks=blocks,
        quality_flags=(),
    )
    chunks = [
        SpanChunk(
            chunk_id="chunk_1",
            doc_id=101,
            source_version="v1",
            source_sha256="abc123sha",
            start_char=0,
            end_char=52,
            chunk_index=0,
            language="vi",
            section="Introduction",
            text=text[0:52],
            index_text=text[0:52],
            dense_token_count=10,
            reranker_token_count=10,
        ),
        SpanChunk(
            chunk_id="chunk_2",
            doc_id=101,
            source_version="v1",
            source_sha256="abc123sha",
            start_char=53,
            end_char=103,
            chunk_index=1,
            language="vi",
            section="Methods",
            text=text[53:103],
            index_text=text[53:103],
            dense_token_count=10,
            reranker_token_count=10,
        ),
        SpanChunk(
            chunk_id="chunk_3",
            doc_id=101,
            source_version="v1",
            source_sha256="abc123sha",
            start_char=104,
            end_char=143,
            chunk_index=2,
            language="vi",
            section="Results",
            text=text[104:143],
            index_text=text[104:143],
            dense_token_count=10,
            reranker_token_count=10,
        ),
    ]

    store = MockSourceStore(doc, chunks)
    evidence_store = EvidenceSourceStore(store)

    query = QueryVariants(vi="insulin therapy", en="insulin therapy")
    fused_hits = [FusedHit(item_id="doc_101", item_type="document", doc_id="101", language="vi", fusion_score=0.9, stream_ranks={})]
    config = EvidenceConfig(document_top_k=1, base_windows_per_doc=2, neighbor_blocks=1)

    spans = select_evidence(query=query, docs=fused_hits, store=evidence_store, config=config)

    assert len(spans) > 0
    variants = {s.variant_kind for s in spans}
    assert "base" in variants
    assert "neighbor_block" in variants
    assert "expanded" in variants

    evidence_store.register_evidence_spans(spans)
    for s in spans:
        resolved_chunk = evidence_store.get_chunk(s.span_id)
        assert resolved_chunk.start_char == s.start_char
        assert resolved_chunk.end_char == s.end_char
        assert resolved_chunk.text == text[s.start_char : s.end_char]
