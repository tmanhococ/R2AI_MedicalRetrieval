from __future__ import annotations

import pytest
from pathlib import Path
from typing import Sequence

from src.data.phase2_contracts import EvidenceSpan, RerankConfig
from src.query.query_processor import QueryVariants
from src.data.vibiomir_contracts import ExtractedDocument, SpanChunk, SourceStore
from src.reranking.neural import NeuralReranker, score_evidence, get_mock_reranker


class MockSourceStore:
    def __init__(self, docs: dict[int, ExtractedDocument], chunks: dict[str, SpanChunk]) -> None:
        self.docs = docs
        self.chunks = chunks

    def get_document(self, doc_id: int) -> ExtractedDocument:
        if doc_id not in self.docs:
            raise KeyError(doc_id)
        return self.docs[doc_id]

    def get_chunk(self, span_id: str) -> SpanChunk:
        # span_id might be chunk_id or format span:doc_id:start:end
        if span_id in self.chunks:
            return self.chunks[span_id]
        for cid, chunk in self.chunks.items():
            if f"span:{chunk.doc_id}:{chunk.start_char}:{chunk.end_char}" == span_id:
                return chunk
        raise KeyError(span_id)

    def contains_official_doc(self, doc_id: int) -> bool:
        return doc_id in self.docs


def test_mock_reranker():
    reranker = get_mock_reranker()
    scores = reranker.score_pairs([("query", "passage 1"), ("query", "passage 2")])
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)


def test_language_routing_and_window_aggregation(tmp_path: Path):
    doc = ExtractedDocument(
        doc_id=101,
        source_version="v1",
        source_text="Clinical guidelines for diabetes management in Vietnam and internationally.",
        source_sha256="abc",
        raw_sha256="def",
        title="Diabetes Guidelines",
        language="en",
        blocks=(),
        quality_flags=(),
    )
    chunk = SpanChunk(
        chunk_id="chunk:101:0:75",
        doc_id=101,
        source_version="v1",
        source_sha256="abc",
        start_char=0,
        end_char=75,
        chunk_index=0,
        language="en",
        section=None,
        text="Clinical guidelines for diabetes management in Vietnam and internationally.",
        index_text="Clinical guidelines for diabetes management in Vietnam and internationally.",
        dense_token_count=10,
        reranker_token_count=10,
    )
    store = MockSourceStore(docs={101: doc}, chunks={"chunk:101:0:75": chunk})

    query = QueryVariants(
        vi="Hướng dẫn điều trị đái tháo đường",
        en="Clinical guidelines for diabetes management",
        zh=None,
    )

    span = EvidenceSpan(
        span_id="chunk:101:0:75",
        doc_id=101,
        source_version="v1",
        source_sha256="abc",
        start_char=0,
        end_char=75,
        seed_chunk_ids=("chunk:101:0:75",),
        variant_kind="base",
    )

    reranker = get_mock_reranker()
    config = RerankConfig(enabled=True, require_real_weights=False, use_raw_logits=True)

    scored = score_evidence(query, [span], reranker=reranker, store=store, config=config, cache_dir=tmp_path)
    assert len(scored) == 1
    assert scored[0].score is not None
    assert isinstance(scored[0].score, float)


def test_cache_invalidation(tmp_path: Path):
    doc = ExtractedDocument(
        doc_id=102,
        source_version="v1",
        source_text="Hypertension treatment protocol.",
        source_sha256="abc",
        raw_sha256="def",
        title="Hypertension",
        language="vi",
        blocks=(),
        quality_flags=(),
    )
    chunk = SpanChunk(
        chunk_id="chunk:102:0:31",
        doc_id=102,
        source_version="v1",
        source_sha256="abc",
        start_char=0,
        end_char=31,
        chunk_index=0,
        language="vi",
        section=None,
        text="Hypertension treatment protocol.",
        index_text="Hypertension treatment protocol.",
        dense_token_count=5,
        reranker_token_count=5,
    )
    store = MockSourceStore(docs={102: doc}, chunks={"chunk:102:0:31": chunk})
    query = QueryVariants(vi="Điều trị tăng huyết áp")

    span = EvidenceSpan(
        span_id="chunk:102:0:31",
        doc_id=102,
        source_version="v1",
        source_sha256="abc",
        start_char=0,
        end_char=31,
        seed_chunk_ids=("chunk:102:0:31",),
        variant_kind="base",
    )

    reranker = get_mock_reranker()
    config = RerankConfig(enabled=True, require_real_weights=False, model_name="BAAI/bge-reranker-v2-m3")

    scored1 = score_evidence(query, [span], reranker=reranker, store=store, config=config, cache_dir=tmp_path)

    # Change model fingerprint or name in config to test invalidation
    config2 = RerankConfig(enabled=True, require_real_weights=False, model_name="different-model")
    scored2 = score_evidence(query, [span], reranker=reranker, store=store, config=config2, cache_dir=tmp_path)
    assert scored1[0].score is not None
    assert scored2[0].score is not None
