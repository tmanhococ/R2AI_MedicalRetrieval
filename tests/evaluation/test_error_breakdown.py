"""Unit tests for error breakdown and failure funnel analysis with provenance.

Tests each failure drop location:
- missing_source_docs (gold doc missing from SourceStore)
- missing_candidate_docs (gold doc in SourceStore but not in candidate_doc_ids)
- lost_after_rerank_docs (gold doc in candidate_doc_ids but not in reranked_doc_ids)
- lost_after_cutoff_docs (gold doc in reranked_doc_ids but not in final_doc_ids)
- missing_chunk_refs & evidence_missing_given_doc (chunk evidence missing given/not given parent doc)
- generate_error_breakdown_summary (dataset-level aggregation of failure stages)
"""

from __future__ import annotations

import pytest

from src.data.vibiomir_contracts import (
    ErrorBreakdown,
    EvaluationRecord,
    ExtractedDocument,
    ReferenceChunk,
    RetrievalTrace,
    SpanChunk,
)
from src.evaluation.error_analysis import (
    build_error_breakdown,
    generate_error_breakdown_summary,
)
from src.evaluation.vibiomir_metrics import (
    EMPTY_BOTH_ZERO_PROFILE,
    PROVISIONAL_SPEC_V1_PROFILE,
)


class MockSourceStore:
    """Mock SourceStore for testing error breakdown."""

    def __init__(
        self,
        docs: dict[int, ExtractedDocument] | None = None,
        chunks: dict[str, SpanChunk] | None = None,
    ) -> None:
        self._docs = docs or {}
        self._chunks = chunks or {}

    def contains_official_doc(self, doc_id: int) -> bool:
        return doc_id in self._docs

    def get_document(self, doc_id: int) -> ExtractedDocument:
        if doc_id not in self._docs:
            raise KeyError(f"Doc {doc_id} not found")
        return self._docs[doc_id]

    def get_chunk(self, chunk_id: str) -> SpanChunk:
        if chunk_id not in self._chunks:
            raise KeyError(f"Chunk {chunk_id} not found")
        return self._chunks[chunk_id]


def _make_dummy_doc(doc_id: int, text: str = "Dummy text") -> ExtractedDocument:
    return ExtractedDocument(
        doc_id=doc_id,
        source_version="v1",
        source_text=text,
        source_sha256="abc",
        raw_sha256="raw_abc",
        title=f"Doc {doc_id}",
        language="vi",
        blocks=(),
        quality_flags=(),
    )


def _make_dummy_chunk(chunk_id: str, doc_id: int, text: str) -> SpanChunk:
    return SpanChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_version="v1",
        source_sha256="abc",
        start_char=0,
        end_char=len(text),
        chunk_index=0,
        language="vi",
        section=None,
        text=text,
        index_text=text,
        dense_token_count=len(text.split()),
        reranker_token_count=len(text.split()),
    )


def test_missing_source_docs_when_not_in_store():
    """Gold doc not present in SourceStore drops into missing_source_docs."""
    # Doc 100 is NOT in store
    store = MockSourceStore(docs={200: _make_dummy_doc(200)})
    gold = EvaluationRecord(
        query_id=1,
        relevant_doc_ids=frozenset({100}),
        relevant_chunks=(),
    )
    trace = RetrievalTrace(
        query_id=1,
        candidate_doc_ids=frozenset(),
        reranked_doc_ids=frozenset(),
        final_doc_ids=frozenset(),
        final_chunk_ids=(),
    )

    breakdown = build_error_breakdown(trace=trace, gold=gold, store=store)

    assert breakdown.query_id == 1
    assert breakdown.missing_source_docs == (100,)
    assert breakdown.missing_candidate_docs == ()
    assert breakdown.lost_after_rerank_docs == ()
    assert breakdown.lost_after_cutoff_docs == ()
    assert breakdown.missing_chunk_refs == ()
    assert breakdown.evidence_missing_given_doc == ()


def test_missing_candidate_docs_when_in_store_but_not_retrieved():
    """Gold doc in SourceStore but missing from candidate_doc_ids drops into missing_candidate_docs."""
    store = MockSourceStore(docs={100: _make_dummy_doc(100)})
    gold = EvaluationRecord(
        query_id=1,
        relevant_doc_ids=frozenset({100}),
        relevant_chunks=(),
    )
    trace = RetrievalTrace(
        query_id=1,
        candidate_doc_ids=frozenset({200, 300}),  # 100 not in candidate pool
        reranked_doc_ids=frozenset({200}),
        final_doc_ids=frozenset({200}),
        final_chunk_ids=(),
    )

    breakdown = build_error_breakdown(trace=trace, gold=gold, store=store)

    assert breakdown.missing_source_docs == ()
    assert breakdown.missing_candidate_docs == (100,)
    assert breakdown.lost_after_rerank_docs == ()
    assert breakdown.lost_after_cutoff_docs == ()


def test_lost_after_rerank_docs():
    """Gold doc in candidate_doc_ids but filtered out before reranked_doc_ids."""
    store = MockSourceStore(docs={100: _make_dummy_doc(100)})
    gold = EvaluationRecord(
        query_id=1,
        relevant_doc_ids=frozenset({100}),
        relevant_chunks=(),
    )
    trace = RetrievalTrace(
        query_id=1,
        candidate_doc_ids=frozenset({100, 200}),
        reranked_doc_ids=frozenset({200}),  # 100 lost during/after rerank
        final_doc_ids=frozenset({200}),
        final_chunk_ids=(),
    )

    breakdown = build_error_breakdown(trace=trace, gold=gold, store=store)

    assert breakdown.missing_source_docs == ()
    assert breakdown.missing_candidate_docs == ()
    assert breakdown.lost_after_rerank_docs == (100,)
    assert breakdown.lost_after_cutoff_docs == ()


def test_lost_after_cutoff_docs():
    """Gold doc in reranked_doc_ids but excluded from final_doc_ids by cutoff."""
    store = MockSourceStore(docs={100: _make_dummy_doc(100)})
    gold = EvaluationRecord(
        query_id=1,
        relevant_doc_ids=frozenset({100}),
        relevant_chunks=(),
    )
    trace = RetrievalTrace(
        query_id=1,
        candidate_doc_ids=frozenset({100}),
        reranked_doc_ids=frozenset({100}),
        final_doc_ids=frozenset(),  # cutoff dropped doc 100
        final_chunk_ids=(),
    )

    breakdown = build_error_breakdown(trace=trace, gold=gold, store=store)

    assert breakdown.missing_source_docs == ()
    assert breakdown.missing_candidate_docs == ()
    assert breakdown.lost_after_rerank_docs == ()
    assert breakdown.lost_after_cutoff_docs == (100,)


def test_funnel_with_multiple_gold_docs():
    """Multiple gold docs failing at distinct stages are separated cleanly."""
    # 10: missing source
    # 20: missing candidate
    # 30: lost after rerank
    # 40: lost after cutoff
    # 50: successfully retrieved in final
    store = MockSourceStore(
        docs={
            20: _make_dummy_doc(20),
            30: _make_dummy_doc(30),
            40: _make_dummy_doc(40),
            50: _make_dummy_doc(50),
        }
    )
    gold = EvaluationRecord(
        query_id=42,
        relevant_doc_ids=frozenset({10, 20, 30, 40, 50}),
        relevant_chunks=(),
    )
    trace = RetrievalTrace(
        query_id=42,
        candidate_doc_ids=frozenset({30, 40, 50}),
        reranked_doc_ids=frozenset({40, 50}),
        final_doc_ids=frozenset({50}),
        final_chunk_ids=(),
    )

    breakdown = build_error_breakdown(trace=trace, gold=gold, store=store)

    assert breakdown.missing_source_docs == (10,)
    assert breakdown.missing_candidate_docs == (20,)
    assert breakdown.lost_after_rerank_docs == (30,)
    assert breakdown.lost_after_cutoff_docs == (40,)


def test_chunk_evidence_missing_when_parent_doc_not_retrieved():
    """Gold chunk missed when parent doc was never retrieved is in missing_chunk_refs,

    but NOT in evidence_missing_given_doc.
    """
    store = MockSourceStore(docs={100: _make_dummy_doc(100)})
    ref_chunk = ReferenceChunk(doc_id=100, chunk_text="bệnh nhân suy tim độ 2")
    gold = EvaluationRecord(
        query_id=1,
        relevant_doc_ids=frozenset({100}),
        relevant_chunks=(ref_chunk,),
    )
    trace = RetrievalTrace(
        query_id=1,
        candidate_doc_ids=frozenset(),
        reranked_doc_ids=frozenset(),
        final_doc_ids=frozenset(),  # Doc 100 never retrieved
        final_chunk_ids=(),
    )

    breakdown = build_error_breakdown(trace=trace, gold=gold, store=store)

    assert breakdown.missing_chunk_refs == (ref_chunk,)
    assert breakdown.evidence_missing_given_doc == ()


def test_chunk_evidence_missing_when_parent_doc_was_retrieved():
    """Gold chunk missed when parent doc WAS retrieved in final_doc_ids

    appears in both missing_chunk_refs AND evidence_missing_given_doc.
    """
    store = MockSourceStore(docs={100: _make_dummy_doc(100)})
    ref_chunk = ReferenceChunk(doc_id=100, chunk_text="triệu chứng suy tim sung huyết")
    gold = EvaluationRecord(
        query_id=1,
        relevant_doc_ids=frozenset({100}),
        relevant_chunks=(ref_chunk,),
    )
    trace = RetrievalTrace(
        query_id=1,
        candidate_doc_ids=frozenset({100}),
        reranked_doc_ids=frozenset({100}),
        final_doc_ids=frozenset({100}),  # Doc 100 retrieved!
        final_chunk_ids=(),  # But no chunks retrieved
    )

    breakdown = build_error_breakdown(trace=trace, gold=gold, store=store)

    assert breakdown.missing_chunk_refs == (ref_chunk,)
    assert breakdown.evidence_missing_given_doc == (ref_chunk,)


def test_chunk_correctly_matched_via_store_lookup():
    """Predicted chunk retrieved via store.get_chunk matching gold chunk prevents missing error."""
    chunk = _make_dummy_chunk("c1", 100, "triệu chứng suy tim cấp và mạn tính")
    store = MockSourceStore(
        docs={100: _make_dummy_doc(100)},
        chunks={"c1": chunk},
    )
    ref_chunk = ReferenceChunk(doc_id=100, chunk_text="triệu chứng suy tim cấp")
    gold = EvaluationRecord(
        query_id=1,
        relevant_doc_ids=frozenset({100}),
        relevant_chunks=(ref_chunk,),
    )
    trace = RetrievalTrace(
        query_id=1,
        candidate_doc_ids=frozenset({100}),
        reranked_doc_ids=frozenset({100}),
        final_doc_ids=frozenset({100}),
        final_chunk_ids=("c1",),
    )

    breakdown = build_error_breakdown(trace=trace, gold=gold, store=store)

    assert breakdown.missing_chunk_refs == ()
    assert breakdown.evidence_missing_given_doc == ()


def test_chunk_correctly_matched_via_trace_metadata_predicted_chunks():
    """Predicted chunk supplied via trace.metadata['predicted_chunks'] correctly matches."""
    store = MockSourceStore(docs={100: _make_dummy_doc(100)})
    ref_chunk = ReferenceChunk(doc_id=100, chunk_text="tăng huyết áp vô căn")
    gold = EvaluationRecord(
        query_id=2,
        relevant_doc_ids=frozenset({100}),
        relevant_chunks=(ref_chunk,),
    )
    trace = RetrievalTrace(
        query_id=2,
        candidate_doc_ids=frozenset({100}),
        reranked_doc_ids=frozenset({100}),
        final_doc_ids=frozenset({100}),
        final_chunk_ids=(),
        metadata={
            "predicted_chunks": [
                {"doc_id": 100, "chunk_text": "chẩn đoán tăng huyết áp vô căn thường gặp"}
            ]
        },
    )

    breakdown = build_error_breakdown(trace=trace, gold=gold, store=store)

    assert breakdown.missing_chunk_refs == ()
    assert breakdown.evidence_missing_given_doc == ()


def test_predicted_chunk_different_doc_id_does_not_match():
    """Predicted chunk with different doc_id does not match even with identical text."""
    store = MockSourceStore(
        docs={100: _make_dummy_doc(100), 200: _make_dummy_doc(200)}
    )
    ref_chunk = ReferenceChunk(doc_id=100, chunk_text="viêm phổi thùy phải")
    gold = EvaluationRecord(
        query_id=3,
        relevant_doc_ids=frozenset({100}),
        relevant_chunks=(ref_chunk,),
    )
    trace = RetrievalTrace(
        query_id=3,
        candidate_doc_ids=frozenset({100, 200}),
        reranked_doc_ids=frozenset({100, 200}),
        final_doc_ids=frozenset({100, 200}),  # doc 100 retrieved
        final_chunk_ids=(),
        metadata={
            "predicted_chunks": [
                {"doc_id": 200, "chunk_text": "viêm phổi thùy phải"}  # Different doc_id!
            ]
        },
    )

    breakdown = build_error_breakdown(trace=trace, gold=gold, store=store)

    # Missed because doc_id differs, and since parent doc 100 was retrieved, it is also in evidence_missing_given_doc
    assert breakdown.missing_chunk_refs == (ref_chunk,)
    assert breakdown.evidence_missing_given_doc == (ref_chunk,)


def test_build_error_breakdown_mismatched_query_id_raises():
    """Mismatched query_id between trace and gold record raises ValueError."""
    store = MockSourceStore()
    gold = EvaluationRecord(query_id=1, relevant_doc_ids=frozenset(), relevant_chunks=())
    trace = RetrievalTrace(
        query_id=2,
        candidate_doc_ids=frozenset(),
        reranked_doc_ids=frozenset(),
        final_doc_ids=frozenset(),
        final_chunk_ids=(),
    )
    with pytest.raises(ValueError, match="query_id.*does not match"):
        build_error_breakdown(trace=trace, gold=gold, store=store)


def test_generate_error_breakdown_summary():
    """Aggregate summary correctly computes totals, affected queries, and rates."""
    b1 = ErrorBreakdown(
        query_id=1,
        missing_source_docs=(10,),
        missing_candidate_docs=(20, 21),
        lost_after_rerank_docs=(),
        lost_after_cutoff_docs=(30,),
        missing_chunk_refs=(ReferenceChunk(doc_id=20, chunk_text="chunk 20"),),
        evidence_missing_given_doc=(),
    )
    b2 = ErrorBreakdown(
        query_id=2,
        missing_source_docs=(),
        missing_candidate_docs=(),
        lost_after_rerank_docs=(40,),
        lost_after_cutoff_docs=(),
        missing_chunk_refs=(
            ReferenceChunk(doc_id=50, chunk_text="chunk 50a"),
            ReferenceChunk(doc_id=50, chunk_text="chunk 50b"),
        ),
        evidence_missing_given_doc=(
            ReferenceChunk(doc_id=50, chunk_text="chunk 50a"),
        ),
    )

    summary = generate_error_breakdown_summary([b1, b2])

    assert summary["total_queries"] == 2
    assert summary["total_missing_source_docs"] == 1
    assert summary["total_missing_candidate_docs"] == 2
    assert summary["total_lost_after_rerank_docs"] == 1
    assert summary["total_lost_after_cutoff_docs"] == 1
    assert summary["total_missing_chunk_refs"] == 3
    assert summary["total_evidence_missing_given_doc"] == 1

    # Affected query counts
    assert summary["queries_with_missing_source"] == 1
    assert summary["queries_with_missing_candidates"] == 1
    assert summary["queries_with_rerank_losses"] == 1
    assert summary["queries_with_cutoff_losses"] == 1
    assert summary["queries_with_missing_chunks"] == 2
    assert summary["queries_with_evidence_missing_given_doc"] == 1


def test_generate_error_breakdown_summary_empty():
    """Empty breakdowns sequence returns zeroed summary."""
    summary = generate_error_breakdown_summary([])
    assert summary["total_queries"] == 0
    assert summary["total_missing_source_docs"] == 0
    assert summary["total_missing_candidate_docs"] == 0
    assert summary["total_lost_after_rerank_docs"] == 0
    assert summary["total_lost_after_cutoff_docs"] == 0
    assert summary["total_missing_chunk_refs"] == 0
    assert summary["total_evidence_missing_given_doc"] == 0
