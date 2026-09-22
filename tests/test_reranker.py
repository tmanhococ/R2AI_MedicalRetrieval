from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from src.data.schemas import FusedHit, ScoredHit
from src.query.query_processor import QueryVariants
from src.reranking import (
    CrossEncoderReranker,
    DummyReranker,
    Reranker,
    min_max_normalize,
    rerank_chunks,
    rerank_documents_direct,
    rerank_documents_via_chunk_aggregation,
    resolve_query_for_language,
)


# =====================================================================
# 1. Package Exports and Protocols
# =====================================================================

def test_package_exports():
    """Verify package level exports exist and are callable/classes."""
    assert CrossEncoderReranker is not None
    assert DummyReranker is not None
    assert Reranker is not None
    assert callable(min_max_normalize)
    assert callable(resolve_query_for_language)
    assert callable(rerank_chunks)
    assert callable(rerank_documents_via_chunk_aggregation)
    assert callable(rerank_documents_direct)


def test_protocol_conformance():
    """Verify both DummyReranker and CrossEncoderReranker conform to Reranker protocol."""
    dummy = DummyReranker()
    assert isinstance(dummy, Reranker)

    callable_reranker = CrossEncoderReranker(model=lambda pairs: [0.5] * len(pairs))
    assert isinstance(callable_reranker, Reranker)


# =====================================================================
# 2. min_max_normalize Unit Tests
# =====================================================================

def test_min_max_normalize_standard_range():
    """Standard list of float scores mapped into [0.0, 1.0]."""
    scores = [-2.0, 0.0, 3.0]
    normalized = min_max_normalize(scores)
    assert len(normalized) == 3
    assert normalized[0] == pytest.approx(0.0, abs=1e-5)
    # (-2 to 3) range is 5. 0.0 is 2/5 = 0.4
    assert normalized[1] == pytest.approx(0.4, abs=1e-5)
    assert normalized[2] == pytest.approx(1.0, abs=1e-5)


def test_min_max_normalize_empty_and_single():
    """Edge cases: empty list and single element."""
    assert min_max_normalize([]) == []

    single = min_max_normalize([5.0])
    assert len(single) == 1
    assert single[0] == pytest.approx(1.0)


def test_min_max_normalize_identical_scores():
    """When max == min, returns default_val (1.0 or custom 0.5)."""
    identical = [4.2, 4.2, 4.2]
    norm_default = min_max_normalize(identical)
    assert norm_default == [1.0, 1.0, 1.0]

    norm_custom = min_max_normalize(identical, default_val=0.5)
    assert norm_custom == [0.5, 0.5, 0.5]


# =====================================================================
# 3. Language-Aware Cross-Encoder Pairing
# =====================================================================

def test_language_aware_pairing_routing():
    """Verify routing: vi -> vi, en -> en (or vi), zh -> zh (or vi), unknown -> vi."""
    variants = QueryVariants(
        vi="triệu chứng sốt xuất huyết",
        en="dengue fever symptoms",
        zh="登革热症状",
    )

    # vi candidate
    assert resolve_query_for_language(variants, "vi") == "triệu chứng sốt xuất huyết"
    assert resolve_query_for_language(variants, "VI") == "triệu chứng sốt xuất huyết"

    # en candidate with translation present
    assert resolve_query_for_language(variants, "en") == "dengue fever symptoms"
    assert resolve_query_for_language(variants, "EN") == "dengue fever symptoms"

    # zh candidate with translation present
    assert resolve_query_for_language(variants, "zh") == "登革热症状"

    # unknown language falls back to vi
    assert resolve_query_for_language(variants, "fr") == "triệu chứng sốt xuất huyết"
    assert resolve_query_for_language(variants, "") == "triệu chứng sốt xuất huyết"
    assert resolve_query_for_language(variants, None) == "triệu chứng sốt xuất huyết"


def test_language_aware_pairing_fallback_when_translation_missing():
    """When en or zh translation is None or empty, safely fall back to query_variants.vi."""
    variants_no_translations = QueryVariants(
        vi="triệu chứng sốt xuất huyết",
        en=None,
        zh="",
    )

    assert resolve_query_for_language(variants_no_translations, "en") == "triệu chứng sốt xuất huyết"
    assert resolve_query_for_language(variants_no_translations, "zh") == "triệu chứng sốt xuất huyết"


# =====================================================================
# 4. Chunk Reranking (rerank_chunks)
# =====================================================================

def test_rerank_chunks_scoring_and_normalization():
    """rerank_chunks scores pairs, min-max normalizes, preserves raw fusion score, and sorts."""
    variants = QueryVariants(vi="bệnh lao phổi", en="pulmonary tuberculosis", zh="肺结核")

    candidates = [
        FusedHit(
            item_id="c_vi_1",
            item_type="chunk",
            doc_id="doc_1",
            language="vi",
            fusion_score=0.035,
            stream_ranks={"vi_bm25": 1},
        ),
        FusedHit(
            item_id="c_en_1",
            item_type="chunk",
            doc_id="doc_2",
            language="en",
            fusion_score=0.025,
            stream_ranks={"en_dense": 2},
        ),
        FusedHit(
            item_id="c_zh_1",
            item_type="chunk",
            doc_id="doc_3",
            language="zh",
            fusion_score=0.015,
            stream_ranks={"zh_bm25": 3},
        ),
    ]

    chunk_texts = {
        "c_vi_1": "Bệnh lao phổi là một bệnh truyền nhiễm do vi khuẩn lao.",
        "c_en_1": "Pulmonary tuberculosis is a bacterial infection of the lungs.",
        "c_zh_1": "肺结核是由结核分枝杆菌引起的肺部感染。",
    }

    # Model returns scores: c_en_1 highest (8.0), c_vi_1 middle (5.0), c_zh_1 lowest (2.0)
    score_map = {
        "Pulmonary tuberculosis is a bacterial infection of the lungs.": 8.0,
        "Bệnh lao phổi là một bệnh truyền nhiễm do vi khuẩn lao.": 5.0,
        "肺结核是由结核分枝杆菌引起的肺部感染。": 2.0,
    }

    reranker = CrossEncoderReranker(
        model=lambda pairs: [score_map.get(text, 0.0) for _, text in pairs]
    )

    scored = reranker.rerank_chunks(variants, candidates, chunk_texts, top_k=2)

    assert len(scored) == 2
    # Top 1 should be c_en_1 with score 8.0
    top1 = scored[0]
    assert top1.item_id == "c_en_1"
    assert top1.item_type == "chunk"
    assert top1.doc_id == "doc_2"
    assert top1.language == "en"
    assert top1.rerank_score == pytest.approx(8.0)
    assert top1.raw_fusion_score == pytest.approx(0.025)
    assert top1.normalized_score == pytest.approx(1.0)

    # Top 2 should be c_vi_1 with score 5.0
    top2 = scored[1]
    assert top2.item_id == "c_vi_1"
    assert top2.item_type == "chunk"
    assert top2.doc_id == "doc_1"
    assert top2.rerank_score == pytest.approx(5.0)
    assert top2.raw_fusion_score == pytest.approx(0.035)
    # Min-max range across the 3 candidates: min=2.0, max=8.0. (5-2)/(8-2) = 0.5
    assert top2.normalized_score == pytest.approx(0.5, abs=1e-4)


def test_rerank_chunks_filters_missing_chunk_texts():
    """Candidates with missing chunk text are filtered out gracefully."""
    variants = QueryVariants(vi="ho ra máu")

    candidates = [
        FusedHit(
            item_id="c_valid",
            item_type="chunk",
            doc_id="doc_1",
            language="vi",
            fusion_score=0.03,
            stream_ranks={},
        ),
        FusedHit(
            item_id="c_missing",
            item_type="chunk",
            doc_id="doc_2",
            language="vi",
            fusion_score=0.02,
            stream_ranks={},
        ),
    ]

    chunk_texts = {
        "c_valid": "Bệnh nhân có triệu chứng ho ra máu kéo dài.",
    }

    dummy = DummyReranker()
    scored = dummy.rerank_chunks(variants, candidates, chunk_texts)

    assert len(scored) == 1
    assert scored[0].item_id == "c_valid"


# =====================================================================
# 5. Document Reranking via Chunk Aggregation
# =====================================================================

def test_rerank_documents_via_chunk_aggregation_max_pooling():
    """base_doc_score(d) = max_{c in d}(rerank_chunk_score(c))."""
    doc_candidates = [
        FusedHit(
            item_id="doc_A",
            item_type="document",
            doc_id="doc_A",
            language="vi",
            fusion_score=0.04,
            stream_ranks={},
        ),
        FusedHit(
            item_id="doc_B",
            item_type="document",
            doc_id="doc_B",
            language="en",
            fusion_score=0.03,
            stream_ranks={},
        ),
    ]

    scored_chunks = [
        ScoredHit(
            item_id="chunk_A1",
            item_type="chunk",
            doc_id="doc_A",
            language="vi",
            rerank_score=3.0,
            normalized_score=0.3,
            raw_fusion_score=0.04,
        ),
        ScoredHit(
            item_id="chunk_A2",
            item_type="chunk",
            doc_id="doc_A",
            language="vi",
            rerank_score=9.0,
            normalized_score=0.9,
            raw_fusion_score=0.04,
        ),
        ScoredHit(
            item_id="chunk_B1",
            item_type="chunk",
            doc_id="doc_B",
            language="en",
            rerank_score=6.0,
            normalized_score=0.6,
            raw_fusion_score=0.03,
        ),
    ]

    # doc_A max chunk score = max(3.0, 9.0) = 9.0
    # doc_B max chunk score = 6.0
    scored_docs = rerank_documents_via_chunk_aggregation(doc_candidates, scored_chunks)

    assert len(scored_docs) == 2
    assert scored_docs[0].doc_id == "doc_A"
    assert scored_docs[0].item_type == "document"
    assert scored_docs[0].rerank_score == pytest.approx(9.0)
    assert scored_docs[0].normalized_score == pytest.approx(1.0)
    assert scored_docs[0].raw_fusion_score == pytest.approx(0.04)

    assert scored_docs[1].doc_id == "doc_B"
    assert scored_docs[1].rerank_score == pytest.approx(6.0)
    assert scored_docs[1].normalized_score == pytest.approx(0.0)
    assert scored_docs[1].raw_fusion_score == pytest.approx(0.03)


def test_rerank_documents_aggregation_fallback_unmatched_doc():
    """Documents without matching chunks fall back to min-floor score, preserved in output."""
    doc_candidates = [
        FusedHit(
            item_id="doc_with_chunks",
            item_type="document",
            doc_id="doc_with_chunks",
            language="vi",
            fusion_score=0.05,
            stream_ranks={},
        ),
        FusedHit(
            item_id="doc_orphan_1",
            item_type="document",
            doc_id="doc_orphan_1",
            language="vi",
            fusion_score=0.02,
            stream_ranks={},
        ),
        FusedHit(
            item_id="doc_orphan_2",
            item_type="document",
            doc_id="doc_orphan_2",
            language="vi",
            fusion_score=0.01,
            stream_ranks={},
        ),
    ]

    scored_chunks = [
        ScoredHit(
            item_id="c1",
            item_type="chunk",
            doc_id="doc_with_chunks",
            language="vi",
            rerank_score=10.0,
            normalized_score=1.0,
            raw_fusion_score=0.05,
        ),
        ScoredHit(
            item_id="c2",
            item_type="chunk",
            doc_id="doc_with_chunks",
            language="vi",
            rerank_score=4.0,
            normalized_score=0.4,
            raw_fusion_score=0.05,
        ),
    ]

    # min chunk score is 4.0. Orphan docs should fall back below 4.0 and not be dropped
    scored_docs = rerank_documents_via_chunk_aggregation(
        doc_candidates, scored_chunks, fallback_floor=1.0
    )

    assert len(scored_docs) == 3
    assert scored_docs[0].doc_id == "doc_with_chunks"
    assert scored_docs[0].rerank_score == pytest.approx(10.0)

    # Orphan docs are preserved
    orphan_ids = [d.doc_id for d in scored_docs[1:]]
    assert "doc_orphan_1" in orphan_ids
    assert "doc_orphan_2" in orphan_ids
    # doc_orphan_1 had higher fusion_score than doc_orphan_2
    assert scored_docs[1].doc_id == "doc_orphan_1"
    assert scored_docs[2].doc_id == "doc_orphan_2"


def test_rerank_documents_aggregation_empty_chunks():
    """When scored_chunks is empty, doc candidates fall back cleanly without error."""
    doc_candidates = [
        FusedHit(
            item_id="doc_1",
            item_type="document",
            doc_id="doc_1",
            language="vi",
            fusion_score=0.05,
            stream_ranks={},
        ),
    ]
    scored_docs = rerank_documents_via_chunk_aggregation(doc_candidates, [])
    assert len(scored_docs) == 1
    assert scored_docs[0].doc_id == "doc_1"


# =====================================================================
# 6. Direct Document Truncation Strategy (rerank_documents_direct)
# =====================================================================

def test_rerank_documents_direct():
    """Direct document strategy truncates long documents and reranks them directly."""
    variants = QueryVariants(vi="ung thư phổi")

    candidates = [
        FusedHit(
            item_id="doc_1",
            item_type="document",
            doc_id="doc_1",
            language="vi",
            fusion_score=0.03,
            stream_ranks={},
        ),
        FusedHit(
            item_id="doc_2",
            item_type="document",
            doc_id="doc_2",
            language="vi",
            fusion_score=0.02,
            stream_ranks={},
        ),
    ]

    doc_texts = {
        "doc_1": "ung thư phổi " * 200,  # very long text
        "doc_2": "bệnh tim mạch thông thường",
    }

    recorded_pairs: list[tuple[str, str]] = []

    def mock_score_pairs(pairs: list[tuple[str, str]]) -> list[float]:
        nonlocal recorded_pairs
        recorded_pairs = list(pairs)
        return [10.0, 1.0]

    reranker = CrossEncoderReranker(model=mock_score_pairs)
    scored = reranker.rerank_documents_direct(
        variants, candidates, doc_texts, max_length=50, top_k=2
    )

    assert len(scored) == 2
    assert scored[0].doc_id == "doc_1"
    assert scored[0].rerank_score == pytest.approx(10.0)
    assert scored[0].normalized_score == pytest.approx(1.0)
    assert scored[1].doc_id == "doc_2"
    assert scored[1].rerank_score == pytest.approx(1.0)
    assert scored[1].normalized_score == pytest.approx(0.0)

    # Verify text passed into pairs was truncated to max_length=50
    assert len(recorded_pairs[0][1]) <= 50


# =====================================================================
# 7. DummyReranker Implementation
# =====================================================================

def test_dummy_reranker_lexical_overlap():
    """DummyReranker deterministic overlap scoring."""
    dummy = DummyReranker()
    pairs = [
        ("viêm gan b mãn tính", "Điều trị viêm gan b mãn tính theo phác đồ"),
        ("viêm gan b mãn tính", "Sốt xuất huyết Dengue thể nặng"),
    ]
    scores = dummy.score_pairs(pairs)
    assert len(scores) == 2
    # First pair has high overlap, second pair has zero overlap
    assert scores[0] > scores[1]
    assert scores[1] == pytest.approx(0.0)


def test_dummy_reranker_custom_fn():
    """DummyReranker supports custom scoring function."""
    dummy = DummyReranker(scoring_fn=lambda q, t: float(len(q) + len(t)))
    scores = dummy.score_pairs([("abc", "def")])
    assert scores == [6.0]


# =====================================================================
# 8. CrossEncoderReranker with Mock Object and Lazy Instantiation
# =====================================================================

def test_cross_encoder_with_mock_predict():
    """Supports pre-instantiated mock model with .predict(sentences, ...)."""
    mock_model = MagicMock()
    mock_model.predict.return_value = [4.5, 2.3]

    reranker = CrossEncoderReranker(model=mock_model, batch_size=16)
    scores = reranker.score_pairs([("q", "t1"), ("q", "t2")])

    assert scores == [4.5, 2.3]
    mock_model.predict.assert_called_once_with(
        [("q", "t1"), ("q", "t2")], batch_size=16, show_progress_bar=False
    )


def test_cross_encoder_lazy_instantiation():
    """Model name string triggers lazy instantiation of CrossEncoder on first call."""
    mock_st = MagicMock()
    mock_instance = MagicMock()
    mock_instance.predict.return_value = [1.23]
    mock_st.CrossEncoder.return_value = mock_instance

    with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
        reranker = CrossEncoderReranker(
            model="BAAI/bge-reranker-v2-m3",
            device="cpu",
            batch_size=8,
            max_length=256,
        )

        # Before scoring, CrossEncoder should not have been called yet
        mock_st.CrossEncoder.assert_not_called()

        # Score pairs triggers instantiation
        scores = reranker.score_pairs([("query", "passage")])
        assert scores == [1.23]
        mock_st.CrossEncoder.assert_called_once_with(
            "BAAI/bge-reranker-v2-m3",
            max_length=256,
            device="cpu",
        )


def test_empty_inputs():
    """All methods gracefully handle empty candidate lists or empty pairs."""
    dummy = DummyReranker()
    variants = QueryVariants(vi="test")

    assert dummy.score_pairs([]) == []
    assert dummy.rerank_chunks(variants, [], {}) == []
    assert dummy.rerank_documents_via_chunk_aggregation([], []) == []
    assert dummy.rerank_documents_direct(variants, [], {}) == []


def test_rerank_documents_synthesize_from_chunks():
    """When doc_candidates is None, synthesize doc candidates from scored_chunks."""
    scored_chunks = [
        ScoredHit(
            item_id="c1",
            item_type="chunk",
            doc_id="doc_X",
            language="vi",
            rerank_score=7.0,
            normalized_score=0.7,
            raw_fusion_score=0.03,
        ),
        ScoredHit(
            item_id="c2",
            item_type="chunk",
            doc_id="doc_Y",
            language="en",
            rerank_score=5.0,
            normalized_score=0.5,
            raw_fusion_score=0.02,
        ),
    ]

    docs = rerank_documents_via_chunk_aggregation(
        doc_candidates=None,
        scored_chunks=scored_chunks,
    )
    assert len(docs) == 2
    assert docs[0].doc_id == "doc_X"
    assert docs[0].rerank_score == pytest.approx(7.0)
    assert docs[1].doc_id == "doc_Y"
    assert docs[1].rerank_score == pytest.approx(5.0)


def test_standalone_rerank_functions():
    """Test standalone helper functions rerank_chunks and rerank_documents_direct."""
    variants = QueryVariants(vi="viêm họng")
    chunk_cand = [
        FusedHit(item_id="c1", item_type="chunk", doc_id="d1", language="vi", fusion_score=0.1, stream_ranks={})
    ]
    doc_cand = [
        FusedHit(item_id="d1", item_type="document", doc_id="d1", language="vi", fusion_score=0.1, stream_ranks={})
    ]
    texts = {"c1": "viêm họng hạt cấp tính", "d1": "viêm họng hạt cấp tính"}

    dummy = DummyReranker()

    # Standalone rerank_chunks: (reranker, query_variants, candidates, chunk_texts)
    res1 = rerank_chunks(dummy, variants, chunk_cand, texts)
    assert len(res1) == 1
    assert res1[0].item_id == "c1"

    # Standalone rerank_chunks: (query_variants, candidates, chunk_texts) with default dummy
    res2 = rerank_chunks(variants, chunk_cand, texts)
    assert len(res2) == 1

    # Standalone rerank_documents_direct: (reranker, query_variants, candidates, doc_texts)
    res3 = rerank_documents_direct(dummy, variants, doc_cand, texts)
    assert len(res3) == 1
    assert res3[0].doc_id == "d1"

    # Standalone rerank_documents_direct: (query_variants, candidates, doc_texts) with default dummy
    res4 = rerank_documents_direct(variants, doc_cand, texts)
    assert len(res4) == 1


def test_cross_encoder_auto_device_and_numpy_output():
    """Verify device='auto' and predict returning numpy array and TypeError fallback."""
    import numpy as np

    mock_st = MagicMock()
    mock_instance = MagicMock()
    # Return numpy array instead of list
    mock_instance.predict.return_value = np.array([3.14, 2.71], dtype=np.float32)
    mock_st.CrossEncoder.return_value = mock_instance

    with patch.dict("sys.modules", {"sentence_transformers": mock_st}):
        reranker = CrossEncoderReranker(
            model="test-model",
            device="auto",
        )
        scores = reranker.score_pairs([("q", "t1"), ("q", "t2")])
        assert len(scores) == 2
        assert scores[0] == pytest.approx(3.14)
        assert scores[1] == pytest.approx(2.71)

        # Test predict TypeError fallback for simple mocks
        def simple_predict(pairs):
            return [9.99]

        simple_mock = MagicMock()
        simple_mock.predict = simple_predict
        simple_reranker = CrossEncoderReranker(model=simple_mock)
        simple_scores = simple_reranker.score_pairs([("q", "t")])
        assert simple_scores == [9.99]


def test_candidate_keyed_by_doc_id_and_all_missing():
    """Verify fallback when candidate is keyed by doc_id in texts and when all candidates missing."""
    variants = QueryVariants(vi="tim mạch")
    cand = [
        FusedHit(item_id="c_other", item_type="chunk", doc_id="d_match", language="vi", fusion_score=0.1, stream_ranks={})
    ]
    texts = {"d_match": "bệnh tim mạch"}

    dummy = DummyReranker()
    # c_other not in texts, but d_match is in texts
    res = dummy.rerank_chunks(variants, cand, texts)
    assert len(res) == 1
    assert res[0].item_id == "c_other"

    # All missing texts
    res_none = dummy.rerank_chunks(variants, cand, {"unrelated": "text"})
    assert res_none == []

    # Direct doc strategy all missing texts
    res_doc_none = dummy.rerank_documents_direct(variants, cand, {"unrelated": "text"})
    assert res_doc_none == []

