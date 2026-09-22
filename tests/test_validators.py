from dataclasses import FrozenInstanceError
import pytest
from src.data import (
    ChunkRecord,
    DocumentRecord,
    FinalPrediction,
    FusedHit,
    GroundTruthRecord,
    QueryRecord,
    ScoredHit,
    SearchHit,
    validate_corpus_integrity,
)


def test_validate_corpus_integrity_valid():
    docs = [
        DocumentRecord(doc_id="doc1", language="vi", text="Văn bản tiếng Việt."),
        DocumentRecord(doc_id="doc2", language="en", text="English text.")
    ]
    chunks = [
        ChunkRecord(chunk_id="doc1_c0", doc_id="doc1", language="vi", text="Văn bản"),
        ChunkRecord(chunk_id="doc2_c0", doc_id="doc2", language="en", text="English")
    ]
    assert validate_corpus_integrity(docs, chunks, strict=True) is True
    assert validate_corpus_integrity(docs, chunks, strict=False) is True


def test_validate_corpus_integrity_missing_parent_doc():
    docs = [DocumentRecord(doc_id="doc1", language="vi", text="Văn bản.")]
    chunks = [ChunkRecord(chunk_id="orphan_c0", doc_id="non_existing_doc", language="vi", text="Orphan")]
    with pytest.raises(ValueError, match="does not exist in document table"):
        validate_corpus_integrity(docs, chunks, strict=True)


def test_validate_corpus_integrity_missing_parent_doc_non_strict():
    docs = [DocumentRecord(doc_id="doc1", language="vi", text="Văn bản.")]
    chunks = [ChunkRecord(chunk_id="orphan_c0", doc_id="non_existing_doc", language="vi", text="Orphan")]
    assert validate_corpus_integrity(docs, chunks, strict=False) is False


def test_validate_corpus_integrity_duplicate_doc_id():
    docs = [
        DocumentRecord(doc_id="doc1", language="vi", text="Doc 1 first"),
        DocumentRecord(doc_id="doc1", language="vi", text="Doc 1 duplicate"),
    ]
    chunks = [ChunkRecord(chunk_id="c1", doc_id="doc1", language="vi", text="chunk")]
    with pytest.raises(ValueError, match="Duplicate doc_id"):
        validate_corpus_integrity(docs, chunks, strict=True)
    assert validate_corpus_integrity(docs, chunks, strict=False) is False


def test_validate_corpus_integrity_invalid_doc_language():
    docs = [
        DocumentRecord(doc_id="doc1", language="fr", text="Texte français.")  # type: ignore[arg-type]
    ]
    chunks = []
    with pytest.raises(ValueError, match="Invalid document language"):
        validate_corpus_integrity(docs, chunks, strict=True)
    assert validate_corpus_integrity(docs, chunks, strict=False) is False


def test_validate_corpus_integrity_invalid_chunk_language():
    docs = [DocumentRecord(doc_id="doc1", language="vi", text="Văn bản.")]
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="doc1", language="de", text="Deutscher text")  # type: ignore[arg-type]
    ]
    with pytest.raises(ValueError, match="Invalid chunk language"):
        validate_corpus_integrity(docs, chunks, strict=True)
    assert validate_corpus_integrity(docs, chunks, strict=False) is False


def test_schemas_instantiation():
    qr = QueryRecord(query_id="q1", text_vi="Triệu chứng cúm")
    assert qr.query_id == "q1"
    assert qr.text_vi == "Triệu chứng cúm"
    assert qr.metadata is None

    gtr = GroundTruthRecord(
        query_id="q1",
        relevant_doc_ids={"doc1"},
        relevant_chunk_ids={"doc1_c0"}
    )
    assert "doc1" in gtr.relevant_doc_ids

    sh = SearchHit(
        item_id="doc1_c0",
        item_type="chunk",
        doc_id="doc1",
        language="vi",
        source="bm25",
        raw_score=12.5,
        rank=1
    )
    assert sh.item_type == "chunk"

    fh = FusedHit(
        item_id="doc1_c0",
        item_type="chunk",
        doc_id="doc1",
        language="vi",
        fusion_score=0.85,
        stream_ranks={"bm25": 1, "dense": 2}
    )
    assert fh.fusion_score == 0.85

    sc = ScoredHit(
        item_id="doc1_c0",
        item_type="chunk",
        doc_id="doc1",
        language="vi",
        rerank_score=4.5,
        normalized_score=0.92,
        raw_fusion_score=0.85
    )
    assert sc.rerank_score == 4.5

    fp = FinalPrediction(
        query_id="q1",
        document_ids=["doc1"],
        chunk_ids=["doc1_c0"],
        document_scores={"doc1": 0.92},
        chunk_scores={"doc1_c0": 0.92}
    )
    assert fp.query_id == "q1"
    assert len(fp.document_ids) == 1


def test_dataclasses_frozen():
    doc = DocumentRecord(doc_id="doc1", language="vi", text="Test")
    with pytest.raises(FrozenInstanceError):
        doc.text = "Modified"  # type: ignore[misc]

    chunk = ChunkRecord(chunk_id="c1", doc_id="doc1", language="vi", text="Chunk")
    with pytest.raises(FrozenInstanceError):
        chunk.text = "Modified"  # type: ignore[misc]
