from __future__ import annotations

import pytest

from src.data.schemas import FusedHit, SearchHit
from src.retrieval.orchestrator import StreamResults
from src.fusion import WeightedRRF, fuse_stream_results, weighted_rrf


def test_package_exports():
    """Verify package level exports."""
    assert callable(weighted_rrf)
    assert callable(fuse_stream_results)
    assert WeightedRRF is not None


def test_weighted_rrf_basic_scoring():
    """Verify standard weighted RRF formula: sum(w_s / (k + rank_s))."""
    hit_d1_s1 = SearchHit(
        item_id="d1",
        item_type="document",
        doc_id="d1",
        language="vi",
        source="s1",
        raw_score=10.0,
        rank=1,
    )
    hit_d2_s1 = SearchHit(
        item_id="d2",
        item_type="document",
        doc_id="d2",
        language="vi",
        source="s1",
        raw_score=5.0,
        rank=2,
    )
    hit_d1_s2 = SearchHit(
        item_id="d1",
        item_type="document",
        doc_id="d1",
        language="vi",
        source="s2",
        raw_score=0.9,
        rank=2,
    )

    ranked_lists = {
        "s1": [hit_d1_s1, hit_d2_s1],
        "s2": [hit_d1_s2],
    }
    weights = {"s1": 1.0, "s2": 1.0}

    fused = weighted_rrf(ranked_lists, weights=weights, rrf_k=60, top_k=10)
    assert len(fused) == 2

    # Expected:
    # d1: 1/(60+1) + 1/(60+2) = 1/61 + 1/62 = 0.0163934426 + 0.0161290323 = 0.0325224749
    # d2: 1/(60+2) = 1/62 = 0.0161290323
    expected_d1 = (1.0 / 61.0) + (1.0 / 62.0)
    expected_d2 = 1.0 / 62.0

    assert fused[0].item_id == "d1"
    assert fused[0].fusion_score == pytest.approx(expected_d1, rel=1e-6)
    assert fused[0].stream_ranks == {"s1": 1, "s2": 2}

    assert fused[1].item_id == "d2"
    assert fused[1].fusion_score == pytest.approx(expected_d2, rel=1e-6)
    assert fused[1].stream_ranks == {"s1": 2}


def test_weighted_rrf_custom_weights_and_defaults():
    """Verify custom weights scale the stream contributions, and missing weights default to 1.0."""
    hit_c1_s1 = SearchHit(
        item_id="c1",
        item_type="chunk",
        doc_id="d1",
        language="vi",
        source="s1",
        raw_score=10.0,
        rank=1,
    )
    hit_c2_s2 = SearchHit(
        item_id="c2",
        item_type="chunk",
        doc_id="d2",
        language="en",
        source="s2",
        raw_score=0.9,
        rank=1,
    )

    ranked_lists = {
        "s1": [hit_c1_s1],
        "s2": [hit_c2_s2],
    }
    # s1 has weight 0.5, s2 is not in weights (should default to 1.0)
    weights = {"s1": 0.5}

    fused = weighted_rrf(ranked_lists, weights=weights, rrf_k=60)
    assert len(fused) == 2

    # c2 score: 1.0 / (60 + 1) = 1/61 ≈ 0.016393
    # c1 score: 0.5 / (60 + 1) = 0.5/61 ≈ 0.008197
    assert fused[0].item_id == "c2"
    assert fused[0].fusion_score == pytest.approx(1.0 / 61.0)
    assert fused[1].item_id == "c1"
    assert fused[1].fusion_score == pytest.approx(0.5 / 61.0)


def test_weighted_rrf_custom_k():
    """Verify custom rrf_k smoothing constant."""
    hit = SearchHit(
        item_id="c1",
        item_type="chunk",
        doc_id="d1",
        language="vi",
        source="s1",
        raw_score=1.0,
        rank=1,
    )
    fused = weighted_rrf({"s1": [hit]}, rrf_k=20)
    assert fused[0].fusion_score == pytest.approx(1.0 / (20 + 1))


def test_tie_breaking_tier1_stream_count():
    """When scores are tied within 1e-9, more streams ranks higher."""
    # c1 appears in 2 streams, each with weight 0.5 and rank 1 -> 0.5/61 + 0.5/61 = 1/61
    # c2 appears in 1 stream with weight 1.0 and rank 1 -> 1.0/61 = 1/61
    c1_s1 = SearchHit(item_id="c1", item_type="chunk", doc_id="d1", language="vi", source="s1", raw_score=1.0, rank=1)
    c1_s2 = SearchHit(item_id="c1", item_type="chunk", doc_id="d1", language="vi", source="s2", raw_score=1.0, rank=1)
    c2_s3 = SearchHit(item_id="c2", item_type="chunk", doc_id="d2", language="en", source="s3", raw_score=1.0, rank=1)

    ranked_lists = {
        "s1": [c1_s1],
        "s2": [c1_s2],
        "s3": [c2_s3],
    }
    weights = {"s1": 0.5, "s2": 0.5, "s3": 1.0}

    fused = weighted_rrf(ranked_lists, weights=weights, rrf_k=60)
    assert len(fused) == 2
    assert fused[0].fusion_score == pytest.approx(fused[1].fusion_score, abs=1e-9)
    # c1 has 2 streams, c2 has 1 stream -> c1 must come first
    assert fused[0].item_id == "c1"
    assert fused[1].item_id == "c2"


def test_tie_breaking_tier2_avg_rank():
    """When scores and stream count are tied, lower average rank ranks higher."""
    # Suppose both c1 and c2 appear in 2 streams with weights s1=0.6, s2=0.4
    # Let's craft identical scores:
    # Actually, easiest: equal weights s1=1, s2=1, s3=1, s4=1
    # If c1 appears in s1 (rank 1) and s2 (rank 4):
    # avg_rank = (1 + 4)/2 = 2.5
    # score = 1/(60+1) + 1/(60+4) = 1/61 + 1/64
    # If c2 has weights tuned to yield the identical score (within 1e-9) with ranks 2 and 3:
    # avg_rank = (2 + 3)/2 = 2.5.
    # To test avg_rank:
    # Let weight be adjusted so score is identical:
    # Score for c_a: in stream A (rank 1), stream B (rank 3). Stream count = 2. avg_rank = 2.0.
    # Score for c_b: in stream C (rank 2), stream D (rank 2). Stream count = 2. avg_rank = 2.0.
    # Wait, what if avg rank is DIFFERENT?
    # Say c_low_avg: score X, 1 stream at rank 1 with weight w1.
    # Say c_high_avg: score X, 1 stream at rank 2 with weight w2 = w1 * (60+2)/(60+1).
    # Then both have score X, both have stream_count = 1.
    # c_low_avg has avg_rank = 1.0. c_high_avg has avg_rank = 2.0.
    # c_low_avg must win!
    w1 = 1.0
    w2 = 1.0 * (60.0 + 2.0) / (60.0 + 1.0)  # 62/61

    c_low = SearchHit(item_id="z_low", item_type="chunk", doc_id="d1", language="vi", source="s1", raw_score=1.0, rank=1)
    c_high = SearchHit(item_id="a_high", item_type="chunk", doc_id="d2", language="vi", source="s2", raw_score=1.0, rank=2)

    ranked_lists = {"s1": [c_low], "s2": [c_high]}
    weights = {"s1": w1, "s2": w2}

    fused = weighted_rrf(ranked_lists, weights=weights, rrf_k=60)
    assert len(fused) == 2
    assert abs(fused[0].fusion_score - fused[1].fusion_score) <= 1e-9
    # z_low has lower avg rank (1.0 vs 2.0), despite lexicographically 'z' > 'a'
    assert fused[0].item_id == "z_low"
    assert fused[1].item_id == "a_high"


def test_tie_breaking_tier3_item_id():
    """When scores, stream count, and avg rank are identical, sort by item_id ascending."""
    stream1 = [
        SearchHit(item_id="c2", item_type="chunk", doc_id="d2", language="vi", source="s1", raw_score=10.0, rank=1),
        SearchHit(item_id="c1", item_type="chunk", doc_id="d1", language="vi", source="s1", raw_score=5.0, rank=2),
    ]
    stream2 = [
        SearchHit(item_id="c1", item_type="chunk", doc_id="d1", language="vi", source="s2", raw_score=0.9, rank=1),
        SearchHit(item_id="c2", item_type="chunk", doc_id="d2", language="vi", source="s2", raw_score=0.8, rank=2),
    ]
    ranked_lists = {"s1": stream1, "s2": stream2}
    weights = {"s1": 1.0, "s2": 1.0}

    fused = weighted_rrf(ranked_lists, weights, rrf_k=60, top_k=2)
    assert len(fused) == 2
    # Both have score 1/61 + 1/62, both have 2 streams, both have avg_rank 1.5
    assert fused[0].fusion_score == pytest.approx(fused[1].fusion_score)
    assert fused[0].item_id == "c1"
    assert fused[1].item_id == "c2"


def test_metadata_preservation_and_language_selection():
    """Language must be taken from the highest ranked hit for that item."""
    # c1 appears in s1 (rank 3, language vi) and s2 (rank 1, language en)
    hit_vi = SearchHit(
        item_id="c1",
        item_type="chunk",
        doc_id="d1",
        language="vi",
        source="s1",
        raw_score=5.0,
        rank=3,
    )
    hit_en = SearchHit(
        item_id="c1",
        item_type="chunk",
        doc_id="d1",
        language="en",
        source="s2",
        raw_score=9.0,
        rank=1,
    )

    ranked_lists = {"s1": [hit_vi], "s2": [hit_en]}
    fused = weighted_rrf(ranked_lists)

    assert len(fused) == 1
    item = fused[0]
    assert item.item_id == "c1"
    assert item.item_type == "chunk"
    assert item.doc_id == "d1"
    assert item.language == "en"  # Rank 1 hit has language 'en'
    assert item.stream_ranks == {"s1": 3, "s2": 1}


def test_top_k_limiting():
    """Verify top_k parameter truncates output."""
    hits = [
        SearchHit(
            item_id=f"d{i}",
            item_type="document",
            doc_id=f"d{i}",
            language="vi",
            source="s1",
            raw_score=100.0 - i,
            rank=i,
        )
        for i in range(1, 20)
    ]
    ranked_lists = {"s1": hits}

    fused_5 = weighted_rrf(ranked_lists, top_k=5)
    assert len(fused_5) == 5
    assert [h.item_id for h in fused_5] == ["d1", "d2", "d3", "d4", "d5"]

    # top_k <= 0 should return empty list
    assert weighted_rrf(ranked_lists, top_k=0) == []
    assert weighted_rrf(ranked_lists, top_k=-5) == []


def test_edge_cases_empty_and_zero_weights():
    """Verify handling of empty inputs, empty streams, zero weights, and corrupt ranks."""
    # Empty ranked_lists
    assert weighted_rrf({}) == []

    # Streams with empty lists
    assert weighted_rrf({"s1": [], "s2": []}) == []

    # All zero weights
    hit1 = SearchHit(item_id="d1", item_type="document", doc_id="d1", language="vi", source="s1", raw_score=1.0, rank=1)
    hit2 = SearchHit(item_id="d2", item_type="document", doc_id="d2", language="vi", source="s2", raw_score=1.0, rank=1)
    fused_zero = weighted_rrf({"s1": [hit1], "s2": [hit2]}, weights={"s1": 0.0, "s2": 0.0})
    assert len(fused_zero) == 2
    assert fused_zero[0].fusion_score == 0.0
    assert fused_zero[1].fusion_score == 0.0
    # Tie broken deterministically by item_id
    assert fused_zero[0].item_id == "d1"
    assert fused_zero[1].item_id == "d2"

    # Negative / zero ranks handled safely (clamped to 1)
    corrupt_hit = SearchHit(
        item_id="d3",
        item_type="document",
        doc_id="d3",
        language="vi",
        source="s1",
        raw_score=1.0,
        rank=-10,
    )
    fused_corrupt = weighted_rrf({"s1": [corrupt_hit]}, rrf_k=60)
    assert len(fused_corrupt) == 1
    assert fused_corrupt[0].stream_ranks["s1"] >= 1
    assert fused_corrupt[0].fusion_score > 0


def test_weighted_rrf_class_interface():
    """Verify WeightedRRF class wrapper and its methods."""
    weights = {"s1": 1.5, "s2": 0.5}
    rrf = WeightedRRF(weights=weights, rrf_k=50)

    h1 = SearchHit(item_id="d1", item_type="document", doc_id="d1", language="vi", source="s1", raw_score=10.0, rank=1)
    h2 = SearchHit(item_id="d2", item_type="document", doc_id="d2", language="vi", source="s2", raw_score=5.0, rank=1)

    res = rrf.fuse({"s1": [h1], "s2": [h2]}, top_k=1)
    assert len(res) == 1
    assert res[0].item_id == "d1"

    # Test fuse_stream_results via class
    stream_results = StreamResults(
        doc_streams={"s1": [h1], "s2": [h2]},
        chunk_streams={
            "s1": [SearchHit(item_id="c1", item_type="chunk", doc_id="d1", language="vi", source="s1", raw_score=1.0, rank=1)],
            "s2": [SearchHit(item_id="c2", item_type="chunk", doc_id="d2", language="vi", source="s2", raw_score=1.0, rank=1)],
        },
    )
    fused_docs, fused_chunks = rrf.fuse_stream_results(stream_results, top_k_docs=1, top_k_chunks=1)
    assert len(fused_docs) == 1
    assert len(fused_chunks) == 1
    assert fused_docs[0].item_id == "d1"
    assert fused_chunks[0].item_id == "c1"


def test_fuse_stream_results_helper():
    """Verify standalone fuse_stream_results helper function."""
    doc_hit = SearchHit(item_id="d1", item_type="document", doc_id="d1", language="vi", source="s1", raw_score=1.0, rank=1)
    chunk_hit = SearchHit(item_id="c1", item_type="chunk", doc_id="d1", language="vi", source="s1", raw_score=1.0, rank=1)

    sr = StreamResults(
        doc_streams={"s1": [doc_hit]},
        chunk_streams={"s1": [chunk_hit]},
    )

    docs, chunks = fuse_stream_results(sr, top_k_docs=5, top_k_chunks=5)
    assert len(docs) == 1
    assert len(chunks) == 1
    assert docs[0].item_id == "d1"
    assert chunks[0].item_id == "c1"

    # Edge cases: empty StreamResults or None
    empty_sr = StreamResults()
    e_docs, e_chunks = fuse_stream_results(empty_sr)
    assert e_docs == []
    assert e_chunks == []

    none_docs, none_chunks = fuse_stream_results(None)  # type: ignore[arg-type]
    assert none_docs == []
    assert none_chunks == []


def test_corrupt_hits_and_duplicate_hits_in_stream():
    """Verify corrupt hits (None, missing rank/score) and duplicates in the same stream."""
    valid_hit = SearchHit(
        item_id="d1",
        item_type="document",
        doc_id="d1",
        language="vi",
        source="s1",
        raw_score=10.0,
        rank=1,
    )
    # Duplicate appearance of d1 in the same stream (should be ignored)
    dup_hit = SearchHit(
        item_id="d1",
        item_type="document",
        doc_id="d1",
        language="vi",
        source="s1",
        raw_score=5.0,
        rank=2,
    )

    class CorruptHit:
        pass

    class NonIntRankHit:
        item_id = "d2"
        item_type = "document"
        doc_id = "d2"
        language = "en"
        source = "s1"
        raw_score = None
        rank = None

    ranked_lists = {
        "s1": [
            valid_hit,
            dup_hit,
            None,  # type: ignore[list-item]
            CorruptHit(),  # type: ignore[list-item]
            NonIntRankHit(),  # type: ignore[list-item]
        ]
    }

    fused = weighted_rrf(ranked_lists)
    assert len(fused) == 2
    assert fused[0].item_id == "d1"
    # Duplicate did not inflate score: score should be exactly 1 / (60 + 1)
    assert fused[0].fusion_score == pytest.approx(1.0 / 61.0)
    assert fused[1].item_id == "d2"
    assert fused[1].stream_ranks["s1"] >= 1


def test_language_selection_equal_rank_tie_breakers():
    """Verify language selection when item has equal ranks across streams."""
    # Test 1: equal rank, different weights -> higher weight stream wins
    hit_vi = SearchHit(
        item_id="c1",
        item_type="chunk",
        doc_id="d1",
        language="vi",
        source="s_low_weight",
        raw_score=10.0,
        rank=1,
    )
    hit_en = SearchHit(
        item_id="c1",
        item_type="chunk",
        doc_id="d1",
        language="en",
        source="s_high_weight",
        raw_score=5.0,
        rank=1,
    )

    ranked_lists = {
        "s_low_weight": [hit_vi],
        "s_high_weight": [hit_en],
    }
    weights = {"s_low_weight": 0.5, "s_high_weight": 1.5}
    fused = weighted_rrf(ranked_lists, weights=weights)
    assert fused[0].language == "en"

    # Test 2: equal rank, equal weights, different raw_score -> higher raw_score wins
    hit_zh = SearchHit(
        item_id="c2",
        item_type="chunk",
        doc_id="d2",
        language="zh",
        source="s1",
        raw_score=2.0,
        rank=1,
    )
    hit_vi2 = SearchHit(
        item_id="c2",
        item_type="chunk",
        doc_id="d2",
        language="vi",
        source="s2",
        raw_score=8.0,
        rank=1,
    )

    fused2 = weighted_rrf({"s1": [hit_zh], "s2": [hit_vi2]}, weights={"s1": 1.0, "s2": 1.0})
    assert fused2[0].language == "vi"

