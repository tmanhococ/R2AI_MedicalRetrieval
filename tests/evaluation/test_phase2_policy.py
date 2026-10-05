import pytest
from src.data.phase2_contracts import EvidenceSpan, OutputPolicy
from src.evaluation.output_selector import select_outputs, DevLeakageChecker

def test_select_outputs_no_positive_evidence():
    policy = OutputPolicy(fallback_policy="heuristic_ce_positive", fallback_raw_logit_min=0.0)
    spans = [
        EvidenceSpan(
            span_id="s1", doc_id=1, source_version="v1", source_sha256="abc",
            start_char=0, end_char=100, seed_chunk_ids=("c1",), variant_kind="base", score=-0.5
        )
    ]
    pred = select_outputs("q1", spans, policy)
    assert pred.query_id == "q1"
    assert pred.document_ids == []
    assert pred.chunk_ids == []
    assert pred.document_scores == []
    assert pred.chunk_scores == []

def test_select_outputs_caps_and_max_score():
    policy = OutputPolicy(
        fallback_policy="heuristic_ce_positive",
        document_cap=2,
        evidence_cap=3,
        evidence_cap_per_doc=2
    )
    spans = [
        EvidenceSpan(span_id="s1", doc_id=1, source_version="v1", source_sha256="abc", start_char=0, end_char=50, seed_chunk_ids=("c1",), variant_kind="base", score=2.0),
        EvidenceSpan(span_id="s2", doc_id=1, source_version="v1", source_sha256="abc", start_char=60, end_char=100, seed_chunk_ids=("c2",), variant_kind="base", score=3.0),
        EvidenceSpan(span_id="s3", doc_id=1, source_version="v1", source_sha256="abc", start_char=110, end_char=150, seed_chunk_ids=("c3",), variant_kind="base", score=1.0),
        EvidenceSpan(span_id="s4", doc_id=2, source_version="v1", source_sha256="abc", start_char=0, end_char=50, seed_chunk_ids=("c4",), variant_kind="base", score=4.0),
        EvidenceSpan(span_id="s5", doc_id=3, source_version="v1", source_sha256="abc", start_char=0, end_char=50, seed_chunk_ids=("c5",), variant_kind="base", score=5.0), # Should be capped by document_cap=2
    ]
    pred = select_outputs("q1", spans, policy)
    assert len(pred.document_ids) <= 2
    assert 3 in pred.document_ids
    assert 2 in pred.document_ids

def test_expanded_span_selection():
    policy = OutputPolicy(fallback_policy="heuristic_ce_positive")
    base = EvidenceSpan(span_id="b1", doc_id=1, source_version="v1", source_sha256="abc", start_char=0, end_char=100, seed_chunk_ids=("c1",), variant_kind="base", score=3.0)
    expanded = EvidenceSpan(span_id="e1", doc_id=1, source_version="v1", source_sha256="abc", start_char=0, end_char=200, seed_chunk_ids=("c1", "c2"), variant_kind="expanded", base_span_id="b1", score=2.8)
    neighbor = EvidenceSpan(span_id="n1", doc_id=1, source_version="v1", source_sha256="abc", start_char=100, end_char=200, seed_chunk_ids=("c2",), variant_kind="neighbor_block", score=1.5)
    
    spans = [base, expanded, neighbor]
    pred = select_outputs("q1", spans, policy)
    assert "e1" in pred.chunk_ids
    assert "b1" not in pred.chunk_ids
    assert "n1" not in pred.chunk_ids

def test_dev_anti_leakage_checker():
    checker = DevLeakageChecker(allowed_dev_query_ids={"dev1"})
    checker.validate_query("dev1")
    with pytest.raises(ValueError, match="Test query ID leakage or unverified split"):
        checker.validate_query("test_query_99")
