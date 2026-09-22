from __future__ import annotations

import pytest

from src.data.schemas import ScoredHit
from src.scoring.consistency import ConsistencyMode, apply_structural_consistency


def test_independent_mode_strings():
    selected_docs = ["d1", "d2"]
    selected_chunks = ["c1_1", "c3_1"]  # c3_1 has doc d3 not in selected_docs
    chunk_to_doc = {"c1_1": "d1", "c3_1": "d3"}

    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=selected_docs,
        selected_chunks=selected_chunks,
        chunk_to_doc=chunk_to_doc,
        mode="independent",
    )

    assert final_docs == ["d1", "d2"]
    assert final_chunks == ["c1_1", "c3_1"]


def test_independent_mode_scored_hits():
    d1 = ScoredHit("d1", "document", "d1", "vi", 0.5, 0.5, 0.1)
    c3 = ScoredHit("c3_1", "chunk", "d3", "vi", 0.9, 0.9, 0.1)

    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=[d1],
        selected_chunks=[c3],
        mode=ConsistencyMode.INDEPENDENT,
    )

    assert final_docs == [d1]
    assert final_chunks == [c3]


def test_enforce_parent_doc_mode_strings():
    selected_docs = ["d1"]
    selected_chunks = ["d1_c1", "d2_c1", "d3_c1", "d2_c2"]
    chunk_to_doc = {
        "d1_c1": "d1",
        "d2_c1": "d2",
        "d3_c1": "d3",
        "d2_c2": "d2",
    }

    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=selected_docs,
        selected_chunks=selected_chunks,
        chunk_to_doc=chunk_to_doc,
        mode="enforce_parent_doc",
    )

    # d1 preserved first, d2 and d3 appended deterministically in encounter order without duplicate d2
    assert final_docs == ["d1", "d2", "d3"]
    assert final_chunks == selected_chunks


def test_enforce_parent_doc_mode_scored_hits():
    d1 = ScoredHit("d1", "document", "d1", "vi", 0.5, 0.5, 0.1)
    d2 = ScoredHit("d2", "document", "d2", "vi", 0.4, 0.4, 0.08)
    c1 = ScoredHit("d1_c1", "chunk", "d1", "vi", 0.8, 0.8, 0.2)
    c2 = ScoredHit("d2_c1", "chunk", "d2", "vi", 0.7, 0.7, 0.15)

    # chunk_to_doc=None -> automatically extracts doc_id from chunk.doc_id
    # available_docs provides the original candidate ScoredHit for d2
    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=[d1],
        selected_chunks=[c1, c2],
        available_docs=[d1, d2],
        mode="enforce_parent_doc",
    )

    assert len(final_docs) == 2
    assert final_docs[0] == d1
    assert final_docs[1] == d2
    assert final_chunks == [c1, c2]


def test_enforce_parent_doc_synthesizes_hit_if_unavailable():
    d1 = ScoredHit("d1", "document", "d1", "vi", 0.5, 0.5, 0.1)
    c2 = ScoredHit("d2_c1", "chunk", "d2", "en", 0.7, 0.7, 0.15)

    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=[d1],
        selected_chunks=[c2],
        mode="enforce_parent_doc",
    )

    assert len(final_docs) == 2
    assert final_docs[0] == d1
    synthesized = final_docs[1]
    assert synthesized.item_id == "d2"
    assert synthesized.item_type == "document"
    assert synthesized.doc_id == "d2"
    assert synthesized.language == "en"


def test_enforce_parent_doc_already_present():
    selected_docs = ["d1", "d2"]
    selected_chunks = ["c1_1", "c2_1"]
    chunk_to_doc = {"c1_1": "d1", "c2_1": "d2"}

    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=selected_docs,
        selected_chunks=selected_chunks,
        chunk_to_doc=chunk_to_doc,
        mode=ConsistencyMode.ENFORCE_PARENT_DOC,
    )
    assert final_docs == ["d1", "d2"]
    assert final_chunks == ["c1_1", "c2_1"]


def test_filter_orphan_chunks_mode_strings():
    selected_docs = ["d1", "d3"]
    selected_chunks = ["c1_1", "c2_1", "c3_1", "c4_1"]
    chunk_to_doc = {
        "c1_1": "d1",
        "c2_1": "d2",
        "c3_1": "d3",
        "c4_1": "d4",
    }

    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=selected_docs,
        selected_chunks=selected_chunks,
        chunk_to_doc=chunk_to_doc,
        mode="filter_orphan_chunks",
    )

    # c2_1 and c4_1 filtered because d2 and d4 are not in selected_docs
    assert final_docs == ["d1", "d3"]
    assert final_chunks == ["c1_1", "c3_1"]


def test_filter_orphan_chunks_mode_scored_hits():
    d1 = ScoredHit("d1", "document", "d1", "vi", 0.5, 0.5, 0.1)
    c1 = ScoredHit("c1", "chunk", "d1", "vi", 0.8, 0.8, 0.1)
    c2 = ScoredHit("c2", "chunk", "d2", "vi", 0.7, 0.7, 0.1)

    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=[d1],
        selected_chunks=[c1, c2],
        mode=ConsistencyMode.FILTER_ORPHAN_CHUNKS,
    )

    assert final_docs == [d1]
    assert final_chunks == [c1]


def test_deduplication_preserves_order():
    selected_docs = ["d1", "d2", "d1", "d3"]
    selected_chunks = ["c1_1", "c1_2", "c1_1"]
    chunk_to_doc = {"c1_1": "d1", "c1_2": "d1"}

    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=selected_docs,
        selected_chunks=selected_chunks,
        chunk_to_doc=chunk_to_doc,
        mode="independent",
    )

    assert final_docs == ["d1", "d2", "d3"]
    assert final_chunks == ["c1_1", "c1_2"]


def test_invalid_mode_raises_error():
    with pytest.raises(ValueError, match="Unknown consistency mode"):
        apply_structural_consistency(
            selected_docs=["d1"],
            selected_chunks=["c1"],
            chunk_to_doc={"c1": "d1"},
            mode="invalid_mode",
        )


def test_empty_inputs_consistency():
    final_docs, final_chunks = apply_structural_consistency(
        selected_docs=[],
        selected_chunks=[],
        mode="enforce_parent_doc",
    )
    assert final_docs == []
    assert final_chunks == []
