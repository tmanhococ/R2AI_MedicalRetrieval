import unicodedata
import pytest
from src.data.schemas import QueryRecord
from src.query.query_processor import (
    AdaptiveQueryRouter,
    ProcessedQuery,
    QueryVariants,
    build_query_variants,
    normalize_vi_query,
)


def test_normalize_vi_query_preserves_medical_terms():
    raw = "  Bệnh nhân bị   HA: 140/90 mmHg,  ĐTĐ type 2 ,  HIV(+) ?? "
    cleaned = normalize_vi_query(raw)
    assert "HA: 140/90 mmHg" in cleaned
    assert "ĐTĐ type 2" in cleaned
    assert "HIV(+)" in cleaned
    assert "  " not in cleaned


def test_normalize_vi_query_unicode_nfc_and_whitespace():
    # Test decomposed form explicitly: e + combining circumflex (\u0302) + combining dot below (\u0323) -> ệ
    decomposed_e = "e\u0302\u0323"
    raw = f"  Bệnh   vi{decomposed_e}n \t\n  đa khoa   "
    cleaned = normalize_vi_query(raw)
    assert cleaned == "Bệnh viện đa khoa"
    # Verify it is NFC
    assert unicodedata.is_normalized("NFC", cleaned)


def test_normalize_vi_query_various_medical_abbreviations():
    raw = "THA, COPD giai đoạn 3, liều 50mg hoặc 10ml, glucose 7.2 mmol/L, test (-)"
    cleaned = normalize_vi_query(raw)
    assert "THA" in cleaned
    assert "COPD" in cleaned
    assert "50mg" in cleaned
    assert "10ml" in cleaned
    assert "7.2 mmol/L" in cleaned
    assert "(-)" in cleaned


def test_normalize_vi_query_empty():
    assert normalize_vi_query("") == ""
    assert normalize_vi_query("   ") == ""


def test_adaptive_query_router_short_query():
    router = AdaptiveQueryRouter(length_threshold=50)
    query = QueryRecord(query_id="q1", text_vi="Thuốc điều trị tăng huyết áp ban đầu?")
    routed = router.route(query)
    assert routed.query_id == "q1"
    assert routed.is_clinical_case is False
    assert routed.bm25_text == routed.dense_text
    assert routed.word_count <= 50
    assert routed.condensed_vi is None


def test_adaptive_query_router_long_clinical_case():
    router = AdaptiveQueryRouter(length_threshold=20, strip_clinical_boilerplate=True)
    long_text = (
        "Bệnh nhân nam 54 tuổi vào viện vì lý do đau tức ngực trái lan ra vai sau khi gắng sức, "
        "tiền sử ghi nhận tăng huyết áp 5 năm đang dùng Amlodipine 5mg, "
        "sau đó được chỉ định xét nghiệm men tim Troponin T tăng cao."
    )
    query = QueryRecord(query_id="q2", text_vi=long_text)
    routed = router.route(query)
    assert routed.query_id == "q2"
    assert routed.is_clinical_case is True
    # Boilerplate removed in bm25_text
    assert "bệnh nhân nam" not in routed.bm25_text.lower()
    assert "vào viện vì lý do" not in routed.bm25_text.lower()
    assert "tiền sử ghi nhận" not in routed.bm25_text.lower()
    assert "sau đó được chỉ định" not in routed.bm25_text.lower()
    assert "Amlodipine" in routed.bm25_text
    assert "Troponin T" in routed.bm25_text
    # Dense retains context
    assert len(routed.dense_text) >= len(routed.bm25_text)
    assert routed.condensed_vi == routed.bm25_text


def test_adaptive_query_router_no_strip_boilerplate():
    router = AdaptiveQueryRouter(length_threshold=10, strip_clinical_boilerplate=False)
    text = "Bệnh nhân nam 40 tuổi vào viện vì lý do ho kéo dài nhiều tuần."
    query = QueryRecord(query_id="q3", text_vi=text)
    routed = router.route(query)
    assert routed.is_clinical_case is True
    assert "bệnh nhân nam" in routed.bm25_text.lower()
    assert routed.bm25_text == routed.normalized_text


def test_adaptive_query_router_dense_head_tail_truncation():
    router = AdaptiveQueryRouter(
        length_threshold=10,
        dense_strategy="head_tail",
        max_dense_tokens=60,
    )
    # Generate 100 words
    words = [f"word{i}" for i in range(100)]
    words[0] = "HEAD_START"
    words[99] = "TAIL_END"
    text = " ".join(words)
    query = QueryRecord(query_id="q4", text_vi=text)
    routed = router.route(query)

    assert routed.is_clinical_case is True
    dense_words = routed.dense_text.split()
    assert len(dense_words) == 60
    assert "HEAD_START" in routed.dense_text
    assert "TAIL_END" in routed.dense_text


def test_adaptive_query_router_dense_full_truncation():
    router = AdaptiveQueryRouter(
        length_threshold=10,
        dense_strategy="full",
        max_dense_tokens=60,
    )
    words = [f"word{i}" for i in range(100)]
    words[0] = "HEAD_START"
    words[99] = "TAIL_END"
    text = " ".join(words)
    query = QueryRecord(query_id="q5", text_vi=text)
    routed = router.route(query)

    dense_words = routed.dense_text.split()
    assert len(dense_words) == 60
    assert "HEAD_START" in routed.dense_text
    assert "TAIL_END" not in routed.dense_text


def test_adaptive_query_router_invalid_strategy():
    with pytest.raises(ValueError, match="dense_strategy"):
        AdaptiveQueryRouter(dense_strategy="invalid_strat")


def test_adaptive_query_router_string_input():
    router = AdaptiveQueryRouter()
    routed = router.route("Sốt phát ban ở trẻ em", query_id="q_str")
    assert routed.query_id == "q_str"
    assert routed.normalized_text == "Sốt phát ban ở trẻ em"


def test_query_variants_dataclass_and_builder():
    router = AdaptiveQueryRouter(length_threshold=5)
    query = QueryRecord(
        query_id="q6",
        text_vi="Bệnh nhân nam vào viện vì đau đầu dữ dội kéo dài nhiều ngày",
    )
    routed = router.route(query)
    assert routed.is_clinical_case is True

    variants = build_query_variants(routed, en="severe headache", zh="严重头痛")
    assert variants.vi == routed.normalized_text
    assert variants.vi_condensed == routed.bm25_text
    assert variants.en == "severe headache"
    assert variants.zh == "严重头痛"
