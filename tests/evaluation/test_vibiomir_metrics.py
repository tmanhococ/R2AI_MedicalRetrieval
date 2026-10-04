from __future__ import annotations

from fractions import Fraction
import pytest

from src.data.vibiomir_contracts import (
    EvaluationProfile,
    EvaluationRecord,
    EvaluationSummary,
    MacroMetrics,
    QueryMetrics,
    ReferenceChunk,
    SubmissionChunk,
    SubmissionRow,
)
from src.evaluation.vibiomir_metrics import (
    EMPTY_BOTH_ZERO_PROFILE,
    PROVISIONAL_SPEC_V1_PROFILE,
    evaluate_dataset,
    evaluate_query,
)


def test_one_pred_two_refs() -> None:
    """Pred 'a b c d', refs 'a b' and 'c d' (same doc_id) -> correct_pred=1, found_ref=2 -> P=1.0, R=1.0."""
    pred: SubmissionRow = {
        "id": 1,
        "relevant_docs": [100],
        "relevant_chunks": [
            {"doc_id": 100, "chunk_text": "a b c d"},
        ],
    }
    gold = EvaluationRecord(
        query_id=1,
        relevant_doc_ids=frozenset([100]),
        relevant_chunks=(
            ReferenceChunk(doc_id=100, chunk_text="a b"),
            ReferenceChunk(doc_id=100, chunk_text="c d"),
        ),
    )

    metrics = evaluate_query(
        query_id=1,
        prediction=pred,
        gold=gold,
        profile=PROVISIONAL_SPEC_V1_PROFILE,
    )

    # Chunks level checks
    assert metrics.chunks.correct_prediction_count == 1
    assert metrics.chunks.found_reference_count == 2
    assert metrics.chunks.prediction_count == 1
    assert metrics.chunks.reference_count == 2
    assert metrics.chunks.precision == pytest.approx(1.0)
    assert metrics.chunks.recall == pytest.approx(1.0)
    assert metrics.chunks.f2 == pytest.approx(1.0)

    # Documents level checks
    assert metrics.documents.correct_prediction_count == 1
    assert metrics.documents.found_reference_count == 1
    assert metrics.documents.precision == pytest.approx(1.0)
    assert metrics.documents.recall == pytest.approx(1.0)
    assert metrics.documents.f2 == pytest.approx(1.0)


def test_two_preds_same_ref() -> None:
    """Preds 'a b', 'a b', refs 'a b', 'c d' -> correct_pred=2, found_ref=1 -> P=1.0, R=0.5."""
    pred: SubmissionRow = {
        "id": 2,
        "relevant_docs": [100],
        "relevant_chunks": [
            {"doc_id": 100, "chunk_text": "a b"},
            {"doc_id": 100, "chunk_text": "a b"},
        ],
    }
    gold = EvaluationRecord(
        query_id=2,
        relevant_doc_ids=frozenset([100]),
        relevant_chunks=(
            ReferenceChunk(doc_id=100, chunk_text="a b"),
            ReferenceChunk(doc_id=100, chunk_text="c d"),
        ),
    )

    metrics = evaluate_query(
        query_id=2,
        prediction=pred,
        gold=gold,
        profile=PROVISIONAL_SPEC_V1_PROFILE,
    )

    # Chunks level: row 0 matches ref 0, row 1 matches ref 0; ref 1 not matched
    assert metrics.chunks.correct_prediction_count == 2
    assert metrics.chunks.found_reference_count == 1
    assert metrics.chunks.prediction_count == 2
    assert metrics.chunks.reference_count == 2
    assert metrics.chunks.precision == pytest.approx(1.0)
    assert metrics.chunks.recall == pytest.approx(0.5)
    # F2: 5 * 1.0 * 0.5 / (4 * 1.0 + 0.5) = 2.5 / 4.5 = 5/9
    assert metrics.chunks.f2 == pytest.approx(5.0 / 9.0)


def test_same_text_wrong_doc_no_match() -> None:
    """Matching text but different doc_id does NOT match."""
    pred: SubmissionRow = {
        "id": 3,
        "relevant_docs": [101],
        "relevant_chunks": [
            {"doc_id": 101, "chunk_text": "a b c d"},
        ],
    }
    gold = EvaluationRecord(
        query_id=3,
        relevant_doc_ids=frozenset([102]),
        relevant_chunks=(
            ReferenceChunk(doc_id=102, chunk_text="a b c d"),
        ),
    )

    metrics = evaluate_query(
        query_id=3,
        prediction=pred,
        gold=gold,
        profile=PROVISIONAL_SPEC_V1_PROFILE,
    )

    assert metrics.chunks.correct_prediction_count == 0
    assert metrics.chunks.found_reference_count == 0
    assert metrics.chunks.precision == pytest.approx(0.0)
    assert metrics.chunks.recall == pytest.approx(0.0)
    assert metrics.chunks.f2 == pytest.approx(0.0)

    assert metrics.documents.correct_prediction_count == 0
    assert metrics.documents.found_reference_count == 0
    assert metrics.documents.precision == pytest.approx(0.0)
    assert metrics.documents.recall == pytest.approx(0.0)
    assert metrics.documents.f2 == pytest.approx(0.0)


def test_no_predictions_empty_and_nonempty() -> None:
    """No predictions gives 0 with non-empty gold; with empty gold, check profile (provisional=1, empty_both_zero=0)."""
    empty_pred: SubmissionRow = {
        "id": 4,
        "relevant_docs": [],
        "relevant_chunks": [],
    }

    # 1. Non-empty gold -> F2 = 0.0
    nonempty_gold = EvaluationRecord(
        query_id=4,
        relevant_doc_ids=frozenset([100]),
        relevant_chunks=(
            ReferenceChunk(doc_id=100, chunk_text="a b"),
        ),
    )
    m_nonempty = evaluate_query(
        query_id=4,
        prediction=empty_pred,
        gold=nonempty_gold,
        profile=PROVISIONAL_SPEC_V1_PROFILE,
    )
    assert m_nonempty.documents.f2 == pytest.approx(0.0)
    assert m_nonempty.chunks.f2 == pytest.approx(0.0)

    # 2. Empty gold with provisional profile (empty_both_f2=1)
    empty_gold = EvaluationRecord(
        query_id=4,
        relevant_doc_ids=frozenset(),
        relevant_chunks=(),
    )
    m_provisional = evaluate_query(
        query_id=4,
        prediction=empty_pred,
        gold=empty_gold,
        profile=PROVISIONAL_SPEC_V1_PROFILE,
    )
    assert m_provisional.documents.f2 == pytest.approx(1.0)
    assert m_provisional.chunks.f2 == pytest.approx(1.0)

    # 3. Empty gold with empty_both_zero profile (empty_both_f2=0)
    m_zero = evaluate_query(
        query_id=4,
        prediction=empty_pred,
        gold=empty_gold,
        profile=EMPTY_BOTH_ZERO_PROFILE,
    )
    assert m_zero.documents.f2 == pytest.approx(0.0)
    assert m_zero.chunks.f2 == pytest.approx(0.0)


def test_macro_average_independence() -> None:
    """Macro F2 is mean of query F2 values, NOT F2 of macro P and macro R."""
    # Query 1: P=1.0, R=0.5 -> F2 = 5/9 ≈ 0.55556
    pred_1: SubmissionRow = {
        "id": 1,
        "relevant_docs": [100],
        "relevant_chunks": [
            {"doc_id": 100, "chunk_text": "a b"},
            {"doc_id": 100, "chunk_text": "a b"},
        ],
    }
    gold_1 = EvaluationRecord(
        query_id=1,
        relevant_doc_ids=frozenset([100]),
        relevant_chunks=(
            ReferenceChunk(doc_id=100, chunk_text="a b"),
            ReferenceChunk(doc_id=100, chunk_text="c d"),
        ),
    )

    # Query 2: P=0.5, R=1.0 -> F2 = 5/6 ≈ 0.83333
    pred_2: SubmissionRow = {
        "id": 2,
        "relevant_docs": [200],
        "relevant_chunks": [
            {"doc_id": 200, "chunk_text": "x y"},
            {"doc_id": 200, "chunk_text": "m n"},
        ],
    }
    gold_2 = EvaluationRecord(
        query_id=2,
        relevant_doc_ids=frozenset([200]),
        relevant_chunks=(
            ReferenceChunk(doc_id=200, chunk_text="x y"),
        ),
    )

    summary = evaluate_dataset(
        predictions=[pred_1, pred_2],
        gold=[gold_1, gold_2],
        query_ids=[1, 2],
        profile=PROVISIONAL_SPEC_V1_PROFILE,
    )

    # Query 1 F2 = 5/9, Query 2 F2 = 5/6
    # Mean of F2 values = (5/9 + 5/6) / 2 = 25 / 36 ≈ 0.69444
    expected_macro_f2 = (5.0 / 9.0 + 5.0 / 6.0) / 2.0
    assert summary.macro_chunks.f2 == pytest.approx(expected_macro_f2)

    # Macro P = (1.0 + 0.5) / 2 = 0.75
    # Macro R = (0.5 + 1.0) / 2 = 0.75
    # F2 of macro P and macro R would be 0.75. Verify macro_chunks.f2 != 0.75!
    f2_of_macro_pr = 0.75
    assert summary.macro_chunks.f2 != pytest.approx(f2_of_macro_pr)

    # Check inferred_final_mean calculation
    assert summary.inferred_final_mean == pytest.approx(
        (summary.macro_documents.f2 + summary.macro_chunks.f2) / 2.0
    )


def test_missing_queries_in_predictions_evaluated_as_empty() -> None:
    """Missing queries in predictions are evaluated as empty predictions (not dropped from macro average)."""
    # Only provide prediction for query 1; query 2 is missing
    pred_1: SubmissionRow = {
        "id": 1,
        "relevant_docs": [100],
        "relevant_chunks": [
            {"doc_id": 100, "chunk_text": "a b c d"},
        ],
    }
    gold_1 = EvaluationRecord(
        query_id=1,
        relevant_doc_ids=frozenset([100]),
        relevant_chunks=(
            ReferenceChunk(doc_id=100, chunk_text="a b"),
        ),
    )
    gold_2 = EvaluationRecord(
        query_id=2,
        relevant_doc_ids=frozenset([200]),
        relevant_chunks=(
            ReferenceChunk(doc_id=200, chunk_text="x y"),
        ),
    )

    summary = evaluate_dataset(
        predictions=[pred_1],
        gold=[gold_1, gold_2],
        query_ids=[1, 2],
        profile=PROVISIONAL_SPEC_V1_PROFILE,
    )

    assert summary.query_count == 2
    # Query 1: doc F2 = 1.0, chunk F2 = 1.0
    # Query 2 (missing in predictions): doc F2 = 0.0, chunk F2 = 0.0
    # Macro doc F2 = 0.5, Macro chunk F2 = 0.5
    assert summary.macro_documents.f2 == pytest.approx(0.5)
    assert summary.macro_chunks.f2 == pytest.approx(0.5)
    assert summary.inferred_final_mean == pytest.approx(0.5)


def test_labels_with_empty_chunk_text_rejected() -> None:
    """Labels with empty chunk text are rejected."""
    with pytest.raises(ValueError, match="chunk_text"):
        ReferenceChunk(doc_id=100, chunk_text="")

    with pytest.raises(ValueError, match="chunk_text"):
        ReferenceChunk(doc_id=100, chunk_text="   ")
