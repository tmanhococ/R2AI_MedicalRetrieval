from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.data.vibiomir_contracts import (
    DevQueryMetadata,
    EvaluationRecord,
    ExtractedBlock,
    ExtractedDocument,
    ReferenceChunk,
    SpanChunk,
)
from src.evaluation.calibrator import CalibrationState, CutoffCalibrator
from src.evaluation.dev_records import load_dev_records


class DummySourceStore:
    """Mock SourceStore for testing dev record loader."""

    def __init__(self, existing_doc_ids: set[int]) -> None:
        self.existing_doc_ids = existing_doc_ids

    def contains_official_doc(self, doc_id: int) -> bool:
        return doc_id in self.existing_doc_ids

    def get_document(self, doc_id: int) -> ExtractedDocument:
        if doc_id not in self.existing_doc_ids:
            raise KeyError(f"Doc {doc_id} not found")
        return ExtractedDocument(
            doc_id=doc_id,
            source_version="v1",
            source_text="Sample text content for testing doc.",
            source_sha256="abc",
            raw_sha256="def",
            title="Sample Doc",
            language="vi",
            blocks=(),
            quality_flags=(),
        )

    def get_chunk(self, chunk_id: str) -> SpanChunk:
        raise NotImplementedError


def test_dev_query_namespace_separation(tmp_path: Path) -> None:
    """Dev query IDs colliding with official 1,200 IDs are rejected."""
    dev_file = tmp_path / "dev_records.jsonl"
    record = {
        "query_id": 50,  # Collides with official ID
        "relevant_doc_ids": [100],
        "relevant_chunks": [{"doc_id": 100, "chunk_text": "sample text"}],
        "metadata": {
            "query_id": 50,
            "query_text": "sample query?",
            "origin": "annotator_1",
            "source_doc_ids": [100],
            "topic": "cardiology",
            "question_style": "factoid",
            "split": "dev",
            "pool_version": "pool_v1",
            "judgment_coverage": 1.0,
        },
    }
    dev_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    official_ids = set(range(1, 1201))
    store = DummySourceStore(existing_doc_ids={100})

    with pytest.raises(ValueError, match="collide|namespace"):
        load_dev_records(dev_file, official_query_ids=official_ids, store=store)


def test_dev_reject_test_public_private_splits(tmp_path: Path) -> None:
    """Reject loading records with test/public/private splits."""
    dev_file = tmp_path / "dev_records.jsonl"
    record = {
        "query_id": 9001,
        "relevant_doc_ids": [100],
        "relevant_chunks": [{"doc_id": 100, "chunk_text": "sample text"}],
        "metadata": {
            "query_id": 9001,
            "query_text": "sample query?",
            "origin": "annotator_1",
            "source_doc_ids": [100],
            "topic": "cardiology",
            "question_style": "factoid",
            "split": "test",  # Forbidden test split
            "pool_version": "pool_v1",
            "judgment_coverage": 1.0,
        },
    }
    dev_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    official_ids = set(range(1, 1201))
    store = DummySourceStore(existing_doc_ids={100})

    with pytest.raises(ValueError, match="split|test|forbidden"):
        load_dev_records(dev_file, official_query_ids=official_ids, store=store)


def test_dev_records_store_missing_doc(tmp_path: Path) -> None:
    """Reject dev records when referenced doc_id is missing from SourceStore."""
    dev_file = tmp_path / "dev_records.jsonl"
    record = {
        "query_id": 9002,
        "relevant_doc_ids": [9999],  # Missing from store
        "relevant_chunks": [{"doc_id": 9999, "chunk_text": "sample text"}],
        "metadata": {
            "query_id": 9002,
            "query_text": "sample query?",
            "origin": "annotator_1",
            "source_doc_ids": [9999],
            "topic": "cardiology",
            "question_style": "factoid",
            "split": "dev",
            "pool_version": "pool_v1",
            "judgment_coverage": 1.0,
        },
    }
    dev_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    official_ids = set(range(1, 1201))
    store = DummySourceStore(existing_doc_ids={100})  # 9999 is missing

    with pytest.raises(ValueError, match="SourceStore|doc_id|not found"):
        load_dev_records(dev_file, official_query_ids=official_ids, store=store)


def test_load_valid_dev_records_with_provenance(tmp_path: Path) -> None:
    """Load valid dev records and verify all metadata and provenance are preserved."""
    dev_file = tmp_path / "dev_records.jsonl"
    record = {
        "query_id": 9003,
        "relevant_doc_ids": [100, 200],
        "relevant_chunks": [
            {"doc_id": 100, "chunk_text": "first chunk text"},
            {"doc_id": 200, "chunk_text": "second chunk text"},
        ],
        "metadata": {
            "query_id": 9003,
            "query_text": "What is the diagnosis?",
            "origin": "clinician_review",
            "source_doc_ids": [100, 200],
            "topic": "oncology",
            "question_style": "symptom_to_disease",
            "split": "dev",
            "pool_version": "pilot_v1",
            "judgment_coverage": 0.95,
        },
    }
    dev_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    official_ids = set(range(1, 1201))
    store = DummySourceStore(existing_doc_ids={100, 200})

    records = load_dev_records(dev_file, official_query_ids=official_ids, store=store)
    assert len(records) == 1
    rec = records[0]
    assert rec.query_id == 9003
    assert rec.relevant_doc_ids == frozenset([100, 200])
    assert len(rec.relevant_chunks) == 2
    assert rec.metadata is not None
    assert rec.metadata.origin == "clinician_review"
    assert rec.metadata.split == "dev"
    assert rec.metadata.judgment_coverage == 0.95


def test_calibrator_rejects_test_split() -> None:
    """CutoffCalibrator.fit rejects test/public/private qrels to prevent leakage."""
    query_scores = {"9001": [("100", 0.9)]}
    test_gt = [
        EvaluationRecord(
            query_id=9001,
            relevant_doc_ids=frozenset([100]),
            relevant_chunks=(),
            metadata=DevQueryMetadata(
                query_id=9001,
                query_text="query?",
                origin="annotator",
                source_doc_ids=(100,),
                topic="topic",
                question_style="style",
                split="test",  # Test split!
                pool_version="v1",
                judgment_coverage=1.0,
            ),
        )
    ]
    calibrator = CutoffCalibrator()
    with pytest.raises(ValueError, match="leakage|test|split"):
        calibrator.fit(query_scores, test_gt)


def test_calibrator_fingerprint_mismatch() -> None:
    """CutoffCalibrator rejects profile or dataset fingerprint mismatch."""
    calibrator = CutoffCalibrator()
    state = CalibrationState(
        item_type="document",
        best_policy="fixed_top_k",
        best_params={"k": 2},
        best_f2=1.0,
        profile_fingerprint="profile_A",
        dataset_fingerprint="dataset_X",
    )
    scores = [("d1", 0.9), ("d2", 0.8)]

    # Matching fingerprints should succeed
    assert (
        calibrator.apply(
            scores,
            state,
            expected_profile_fingerprint="profile_A",
            expected_dataset_fingerprint="dataset_X",
        )
        == 2
    )

    # Mismatched profile fingerprint should be rejected
    with pytest.raises(ValueError, match="(?i)profile.*fingerprint.*mismatch"):
        calibrator.apply(
            scores,
            state,
            expected_profile_fingerprint="profile_B",
            expected_dataset_fingerprint="dataset_X",
        )

    # Mismatched dataset fingerprint should be rejected
    with pytest.raises(ValueError, match="(?i)dataset.*fingerprint.*mismatch"):
        calibrator.apply(
            scores,
            state,
            expected_profile_fingerprint="profile_A",
            expected_dataset_fingerprint="dataset_Y",
        )


def test_calibrator_fit_disabled() -> None:
    """CutoffCalibrator with fit_enabled=False rejects fit."""
    calibrator = CutoffCalibrator(fit_enabled=False)
    query_scores = {"9001": [("100", 0.9)]}
    gt = [
        EvaluationRecord(
            query_id=9001,
            relevant_doc_ids=frozenset([100]),
            relevant_chunks=(),
            metadata=DevQueryMetadata(
                query_id=9001,
                query_text="query?",
                origin="annotator",
                source_doc_ids=(100,),
                topic="topic",
                question_style="style",
                split="dev",
                pool_version="v1",
                judgment_coverage=1.0,
            ),
        )
    ]
    with pytest.raises(RuntimeError, match="disabled"):
        calibrator.fit(query_scores, gt)
