from __future__ import annotations

import pytest

from src.data.schemas import ScoredHit
from src.scoring.joint_scoring import JointScorer, compute_joint_scores


def test_joint_scoring_strengthens_evidence():
    docs = [
        ScoredHit(
            item_id="d1",
            item_type="document",
            doc_id="d1",
            language="vi",
            rerank_score=0.5,
            normalized_score=0.5,
            raw_fusion_score=0.1,
        ),
        ScoredHit(
            item_id="d2",
            item_type="document",
            doc_id="d2",
            language="vi",
            rerank_score=0.5,
            normalized_score=0.5,
            raw_fusion_score=0.1,
        ),
    ]
    chunks = [
        ScoredHit(
            item_id="d1_c1",
            item_type="chunk",
            doc_id="d1",
            language="vi",
            rerank_score=1.0,
            normalized_score=1.0,
            raw_fusion_score=0.1,
        ),
        ScoredHit(
            item_id="d2_c1",
            item_type="chunk",
            doc_id="d2",
            language="vi",
            rerank_score=0.2,
            normalized_score=0.2,
            raw_fusion_score=0.1,
        ),
    ]
    joint_docs, joint_chunks = compute_joint_scores(docs, chunks)

    # d1 had strong chunk (1.0), d2 had weak chunk (0.2). d1 final score must be higher!
    d1_final = next(d.normalized_score for d in joint_docs if d.doc_id == "d1")
    d2_final = next(d.normalized_score for d in joint_docs if d.doc_id == "d2")
    assert d1_final > d2_final


def test_exact_formula_calculation():
    # alpha=0.70, beta=0.20, gamma=0.10, top_m=3
    # delta=0.85, epsilon=0.15
    docs = [
        ScoredHit(
            item_id="d1",
            item_type="document",
            doc_id="d1",
            language="vi",
            rerank_score=0.6,
            normalized_score=0.6,
            raw_fusion_score=0.2,
        ),
    ]
    chunks = [
        ScoredHit(
            item_id="d1_c1",
            item_type="chunk",
            doc_id="d1",
            language="vi",
            rerank_score=0.9,
            normalized_score=0.9,
            raw_fusion_score=0.2,
        ),
        ScoredHit(
            item_id="d1_c2",
            item_type="chunk",
            doc_id="d1",
            language="vi",
            rerank_score=0.6,
            normalized_score=0.6,
            raw_fusion_score=0.15,
        ),
        ScoredHit(
            item_id="d1_c3",
            item_type="chunk",
            doc_id="d1",
            language="vi",
            rerank_score=0.3,
            normalized_score=0.3,
            raw_fusion_score=0.1,
        ),
        ScoredHit(
            item_id="d1_c4",
            item_type="chunk",
            doc_id="d1",
            language="vi",
            rerank_score=0.1,
            normalized_score=0.1,
            raw_fusion_score=0.05,
        ),
    ]

    joint_docs, joint_chunks = compute_joint_scores(
        doc_hits=docs,
        chunk_hits=chunks,
        doc_base_weight=0.70,
        doc_best_chunk_weight=0.20,
        doc_top_chunks_mean_weight=0.10,
        top_m_chunks=3,
        chunk_base_weight=0.85,
        chunk_parent_doc_weight=0.15,
    )

    # For d1:
    # base_doc = 0.6
    # max_chunk = 0.9
    # top_3_chunks = [0.9, 0.6, 0.3], mean = 1.8 / 3 = 0.6
    # final_doc = 0.70 * 0.6 + 0.20 * 0.9 + 0.10 * 0.6 = 0.42 + 0.18 + 0.06 = 0.66
    assert len(joint_docs) == 1
    assert joint_docs[0].normalized_score == pytest.approx(0.66)

    # For chunks:
    # d1_c1: 0.85 * 0.9 + 0.15 * 0.66 = 0.765 + 0.099 = 0.864
    # d1_c2: 0.85 * 0.6 + 0.15 * 0.66 = 0.510 + 0.099 = 0.609
    # d1_c3: 0.85 * 0.3 + 0.15 * 0.66 = 0.255 + 0.099 = 0.354
    # d1_c4: 0.85 * 0.1 + 0.15 * 0.66 = 0.085 + 0.099 = 0.184
    chunk_scores = {c.item_id: c.normalized_score for c in joint_chunks}
    assert chunk_scores["d1_c1"] == pytest.approx(0.864)
    assert chunk_scores["d1_c2"] == pytest.approx(0.609)
    assert chunk_scores["d1_c3"] == pytest.approx(0.354)
    assert chunk_scores["d1_c4"] == pytest.approx(0.184)


def test_doc_without_matching_chunks():
    docs = [
        ScoredHit(
            item_id="d1",
            item_type="document",
            doc_id="d1",
            language="vi",
            rerank_score=0.8,
            normalized_score=0.8,
            raw_fusion_score=0.3,
        ),
    ]
    chunks = [
        ScoredHit(
            item_id="other_c1",
            item_type="chunk",
            doc_id="d_other",
            language="vi",
            rerank_score=0.7,
            normalized_score=0.7,
            raw_fusion_score=0.2,
        ),
    ]

    joint_docs, _ = compute_joint_scores(docs, chunks, doc_base_weight=0.70)
    # d1 has no chunks: chunk evidence is 0.0
    # final_doc = 0.70 * 0.8 + 0.20 * 0.0 + 0.10 * 0.0 = 0.56
    assert joint_docs[0].normalized_score == pytest.approx(0.56)


def test_doc_with_fewer_than_m_chunks():
    docs = [
        ScoredHit(
            item_id="d1",
            item_type="document",
            doc_id="d1",
            language="vi",
            rerank_score=0.5,
            normalized_score=0.5,
            raw_fusion_score=0.1,
        ),
    ]
    # Only 2 chunks, when top_m_chunks=3
    chunks = [
        ScoredHit(
            item_id="d1_c1",
            item_type="chunk",
            doc_id="d1",
            language="vi",
            rerank_score=0.8,
            normalized_score=0.8,
            raw_fusion_score=0.1,
        ),
        ScoredHit(
            item_id="d1_c2",
            item_type="chunk",
            doc_id="d1",
            language="vi",
            rerank_score=0.4,
            normalized_score=0.4,
            raw_fusion_score=0.1,
        ),
    ]

    joint_docs, _ = compute_joint_scores(
        docs,
        chunks,
        doc_base_weight=0.70,
        doc_best_chunk_weight=0.20,
        doc_top_chunks_mean_weight=0.10,
        top_m_chunks=3,
    )
    # max = 0.8
    # mean of available 2 chunks = (0.8 + 0.4) / 2 = 0.6
    # final = 0.70 * 0.5 + 0.20 * 0.8 + 0.10 * 0.6 = 0.35 + 0.16 + 0.06 = 0.57
    assert joint_docs[0].normalized_score == pytest.approx(0.57)


def test_chunk_with_missing_parent_doc():
    docs = [
        ScoredHit(
            item_id="d1",
            item_type="document",
            doc_id="d1",
            language="vi",
            rerank_score=0.8,
            normalized_score=0.8,
            raw_fusion_score=0.3,
        ),
    ]
    chunks = [
        ScoredHit(
            item_id="d2_c1",
            item_type="chunk",
            doc_id="d2",  # d2 not in docs
            language="vi",
            rerank_score=0.6,
            normalized_score=0.6,
            raw_fusion_score=0.2,
        ),
    ]

    _, joint_chunks = compute_joint_scores(
        docs,
        chunks,
        chunk_base_weight=0.85,
        chunk_parent_doc_weight=0.15,
        parent_doc_fallback=0.0,
    )
    # final_chunk = 0.85 * 0.6 + 0.15 * 0.0 = 0.51
    assert joint_chunks[0].normalized_score == pytest.approx(0.51)


def test_unnormalized_raw_scores_auto_normalized():
    # If rerank_scores are raw unnormalized logits (e.g. 10.0 and 2.0)
    docs = [
        ScoredHit(
            item_id="d1",
            item_type="document",
            doc_id="d1",
            language="vi",
            rerank_score=10.0,
            normalized_score=10.0,  # > 1.0 triggers auto-normalization
            raw_fusion_score=0.5,
        ),
        ScoredHit(
            item_id="d2",
            item_type="document",
            doc_id="d2",
            language="vi",
            rerank_score=2.0,
            normalized_score=2.0,
            raw_fusion_score=0.4,
        ),
    ]
    chunks = [
        ScoredHit(
            item_id="d1_c1",
            item_type="chunk",
            doc_id="d1",
            language="vi",
            rerank_score=20.0,
            normalized_score=20.0,
            raw_fusion_score=0.5,
        ),
        ScoredHit(
            item_id="d2_c1",
            item_type="chunk",
            doc_id="d2",
            language="vi",
            rerank_score=0.0,
            normalized_score=0.0,
            raw_fusion_score=0.4,
        ),
    ]

    joint_docs, joint_chunks = compute_joint_scores(docs, chunks, normalize_inputs="auto")
    # Base docs normalized: d1 -> 1.0, d2 -> 0.0
    # Chunks normalized: d1_c1 -> 1.0, d2_c1 -> 0.0
    # d1_final = 0.7*1.0 + 0.2*1.0 + 0.1*1.0 = 1.0
    # d2_final = 0.7*0.0 + 0.2*0.0 + 0.1*0.0 = 0.0
    d1_score = next(d.normalized_score for d in joint_docs if d.doc_id == "d1")
    d2_score = next(d.normalized_score for d in joint_docs if d.doc_id == "d2")
    assert d1_score == pytest.approx(1.0)
    assert d2_score == pytest.approx(0.0)


def test_deterministic_tie_breaking_and_sorting():
    docs = [
        ScoredHit(
            item_id="doc_b",
            item_type="document",
            doc_id="doc_b",
            language="vi",
            rerank_score=0.5,
            normalized_score=0.5,
            raw_fusion_score=0.2,
        ),
        ScoredHit(
            item_id="doc_a",
            item_type="document",
            doc_id="doc_a",
            language="vi",
            rerank_score=0.5,
            normalized_score=0.5,
            raw_fusion_score=0.2,
        ),
        ScoredHit(
            item_id="doc_c",
            item_type="document",
            doc_id="doc_c",
            language="vi",
            rerank_score=0.5,
            normalized_score=0.5,
            raw_fusion_score=0.3,  # Higher fusion score
        ),
    ]
    # No chunks, all final scores equal 0.70 * 0.5 = 0.35
    joint_docs, _ = compute_joint_scores(docs, [])

    # doc_c should be first (fusion_score 0.3 > 0.2)
    # doc_a should be before doc_b (tie-break on item_id asc)
    assert [d.item_id for d in joint_docs] == ["doc_c", "doc_a", "doc_b"]


def test_empty_inputs():
    docs, chunks = compute_joint_scores([], [])
    assert docs == []
    assert chunks == []


def test_joint_scorer_class_and_from_config():
    config = {
        "joint_scoring": {
            "document": {
                "base_weight": 0.60,
                "best_chunk_weight": 0.25,
                "top_chunks_mean_weight": 0.15,
                "top_m_chunks": 4,
            },
            "chunk": {
                "base_weight": 0.80,
                "parent_document_weight": 0.20,
            },
        }
    }
    scorer = JointScorer.from_config(config)
    assert scorer.doc_base_weight == 0.60
    assert scorer.doc_best_chunk_weight == 0.25
    assert scorer.doc_top_chunks_mean_weight == 0.15
    assert scorer.top_m_chunks == 4
    assert scorer.chunk_base_weight == 0.80
    assert scorer.chunk_parent_doc_weight == 0.20

    # Also test scorer.score()
    docs = [
        ScoredHit(
            item_id="d1",
            item_type="document",
            doc_id="d1",
            language="vi",
            rerank_score=0.5,
            normalized_score=0.5,
            raw_fusion_score=0.1,
        )
    ]
    j_docs, j_chunks = scorer.score(docs, [])
    assert len(j_docs) == 1
    assert j_docs[0].normalized_score == pytest.approx(0.60 * 0.5)


def test_fields_preserved_in_scored_hits():
    docs = [
        ScoredHit(
            item_id="d1",
            item_type="document",
            doc_id="d1",
            language="zh",
            rerank_score=15.2,
            normalized_score=0.7,
            raw_fusion_score=0.035,
        )
    ]
    chunks = [
        ScoredHit(
            item_id="c1",
            item_type="chunk",
            doc_id="d1",
            language="zh",
            rerank_score=12.4,
            normalized_score=0.8,
            raw_fusion_score=0.025,
        )
    ]
    j_docs, j_chunks = compute_joint_scores(docs, chunks)
    d = j_docs[0]
    c = j_chunks[0]

    assert d.item_id == "d1"
    assert d.item_type == "document"
    assert d.doc_id == "d1"
    assert d.language == "zh"
    assert d.rerank_score == 15.2
    assert d.raw_fusion_score == 0.035

    assert c.item_id == "c1"
    assert c.item_type == "chunk"
    assert c.doc_id == "d1"
    assert c.language == "zh"
    assert c.rerank_score == 12.4
    assert c.raw_fusion_score == 0.025
