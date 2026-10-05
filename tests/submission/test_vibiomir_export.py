from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile
import pytest

from src.data.schemas import FinalPrediction
from src.data.vibiomir_contracts import (
    ExtractedBlock,
    ExtractedDocument,
    OfficialQuery,
    SourceStore,
    SpanChunk,
    SubmissionChunk,
    SubmissionRow,
    ValidationReport,
)
from src.submission.validators import (
    SubmissionValidationError,
    validate_predictions,
    validate_zip,
)
from src.submission.vibiomir_json_zip import ExportResult, VibiomirExporter


class MockSourceStore:
    """In-memory SourceStore protocol implementation for testing."""

    def __init__(
        self,
        docs: dict[int, ExtractedDocument] | None = None,
        chunks: dict[str, SpanChunk] | None = None,
        official_doc_ids: set[int] | None = None,
        snapshot_id: str = "snap_test_001",
    ) -> None:
        self.docs = docs or {}
        self.chunks = chunks or {}
        self.official_doc_ids = official_doc_ids if official_doc_ids is not None else set(self.docs.keys())
        self.snapshot_id = snapshot_id

    def get_document(self, doc_id: int) -> ExtractedDocument:
        if doc_id not in self.docs:
            raise KeyError(f"Document {doc_id} not found")
        return self.docs[doc_id]

    def get_chunk(self, chunk_id: str) -> SpanChunk:
        if chunk_id not in self.chunks:
            raise KeyError(f"Chunk {chunk_id} not found")
        return self.chunks[chunk_id]

    def contains_official_doc(self, doc_id: int) -> bool:
        return doc_id in self.official_doc_ids


def _make_sample_doc_and_chunk(doc_id: int = 583, text: str = "Bệnh sốt xuất huyết do virus Dengue gây ra."):
    doc = ExtractedDocument(
        doc_id=doc_id,
        source_version="v1",
        source_text=text,
        source_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        raw_sha256="raw_hash_001",
        title="Sốt xuất huyết",
        language="vi",
        blocks=(ExtractedBlock(block_index=0, kind="p", start_char=0, end_char=len(text), section=None),),
        quality_flags=(),
    )
    chunk_id = f"{doc_id}:{doc.source_sha256}:000000"
    chunk = SpanChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        source_version=doc.source_version,
        source_sha256=doc.source_sha256,
        start_char=0,
        end_char=len(text),
        chunk_index=0,
        language="vi",
        section=None,
        text=text,
        index_text=text,
        dense_token_count=10,
        reranker_token_count=10,
    )
    return doc, chunk


@pytest.fixture
def store_and_queries():
    doc1, chunk1 = _make_sample_doc_and_chunk(doc_id=583, text="Bệnh sốt xuất huyết Dengue lây qua muỗi vằn.")
    doc2, chunk2 = _make_sample_doc_and_chunk(doc_id=4294967297, text="Triệu chứng gồm sốt cao, đau cơ và phát ban.")
    store = MockSourceStore(
        docs={583: doc1, 4294967297: doc2},
        chunks={chunk1.chunk_id: chunk1, chunk2.chunk_id: chunk2},
        official_doc_ids={583, 4294967297},
        snapshot_id="snapshot_vibiomir_pilot",
    )
    queries = (
        OfficialQuery(query_id=1, query="Sốt xuất huyết lây qua đâu?"),
        OfficialQuery(query_id=2, query="Triệu chứng của sốt xuất huyết là gì?"),
    )
    return store, queries, chunk1, chunk2


def test_source_store_protocol_compliance(store_and_queries):
    store, _, _, _ = store_and_queries
    assert isinstance(store, SourceStore)


# ---------------------------------------------------------------------------
# 1. Sparse IDs and Signed int64 validation
# ---------------------------------------------------------------------------

def test_sparse_int64_ids_pass(store_and_queries):
    store, queries, _, _ = store_and_queries
    rows: list[SubmissionRow] = [
        {
            "id": 1,
            "relevant_docs": [583, 4294967297],
            "relevant_chunks": [{"doc_id": 583, "chunk_text": "Bệnh sốt xuất huyết Dengue lây qua muỗi vằn."}],
        },
        {
            "id": 2,
            "relevant_docs": [4294967297],
            "relevant_chunks": [{"doc_id": 4294967297, "chunk_text": "Triệu chứng gồm sốt cao, đau cơ và phát ban."}],
        },
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is True
    assert report.query_count == 2
    assert report.document_count == 3
    assert report.chunk_count == 2
    assert len(report.errors) == 0


@pytest.mark.parametrize("bad_query_id", [True, False, "1", 1.0, 9223372036854775808, -9223372036854775809])
def test_reject_non_int64_or_bool_query_id(store_and_queries, bad_query_id):
    store, queries, _, _ = store_and_queries
    rows = [
        {"id": bad_query_id, "relevant_docs": [583], "relevant_chunks": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is False
    assert any("query ID" in err.lower() or "signed int64" in err.lower() or "integer" in err.lower() for err in report.errors)


@pytest.mark.parametrize("bad_doc_id", [True, False, "583", 583.0, 9223372036854775808])
def test_reject_non_int64_or_bool_doc_id(store_and_queries, bad_doc_id):
    store, queries, _, _ = store_and_queries
    # in relevant_docs
    rows1 = [
        {"id": 1, "relevant_docs": [bad_doc_id], "relevant_chunks": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report1 = validate_predictions(rows1, queries, store)
    assert report1.valid is False
    assert any("doc_id" in err.lower() or "integer" in err.lower() for err in report1.errors)

    # in relevant_chunks
    rows2 = [
        {"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": bad_doc_id, "chunk_text": "some text"}]},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report2 = validate_predictions(rows2, queries, store)
    assert report2.valid is False
    assert any("doc_id" in err.lower() or "integer" in err.lower() for err in report2.errors)


# ---------------------------------------------------------------------------
# 2. Query coverage
# ---------------------------------------------------------------------------

def test_reject_missing_query(store_and_queries):
    store, queries, _, _ = store_and_queries
    # Missing query 2
    rows = [
        {"id": 1, "relevant_docs": [583], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is False
    assert any("missing" in err.lower() for err in report.errors)


def test_reject_extra_query(store_and_queries):
    store, queries, _, _ = store_and_queries
    # Extra query 999
    rows = [
        {"id": 1, "relevant_docs": [], "relevant_chunks": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
        {"id": 999, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is False
    assert any("extra" in err.lower() or "unknown query" in err.lower() for err in report.errors)


def test_reject_duplicate_query(store_and_queries):
    store, queries, _, _ = store_and_queries
    # Duplicate query 1
    rows = [
        {"id": 1, "relevant_docs": [], "relevant_chunks": []},
        {"id": 1, "relevant_docs": [], "relevant_chunks": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is False
    assert any("duplicate" in err.lower() for err in report.errors)


def test_expected_count_check(store_and_queries):
    store, queries, _, _ = store_and_queries
    rows = [
        {"id": 1, "relevant_docs": [], "relevant_chunks": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    # expected 1200 but got 2
    report = validate_predictions(rows, queries, store, expected_count=1200)
    assert report.valid is False
    assert any("expected" in err.lower() and "count" in err.lower() for err in report.errors)


# ---------------------------------------------------------------------------
# 3. Foreign doc IDs
# ---------------------------------------------------------------------------

def test_reject_foreign_doc_id(store_and_queries):
    store, queries, _, _ = store_and_queries
    # Doc 999999 not in store
    rows = [
        {"id": 1, "relevant_docs": [999999], "relevant_chunks": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is False
    assert any("foreign" in err.lower() or "registry" in err.lower() or "not in snapshot" in err.lower() for err in report.errors)


def test_reject_foreign_chunk_doc_id(store_and_queries):
    store, queries, _, _ = store_and_queries
    rows = [
        {"id": 1, "relevant_docs": [], "relevant_chunks": [{"doc_id": 999999, "chunk_text": "text"}]},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is False
    assert any("foreign" in err.lower() or "registry" in err.lower() or "not in snapshot" in err.lower() for err in report.errors)


# ---------------------------------------------------------------------------
# 4. Duplicates in relevant_docs and relevant_chunks
# ---------------------------------------------------------------------------

def test_reject_duplicate_doc_ids_in_same_query(store_and_queries):
    store, queries, _, _ = store_and_queries
    rows = [
        {"id": 1, "relevant_docs": [583, 583], "relevant_chunks": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is False
    assert any("duplicate" in err.lower() and "doc" in err.lower() for err in report.errors)


def test_reject_duplicate_chunk_in_same_query(store_and_queries):
    store, queries, _, _ = store_and_queries
    text = "Bệnh sốt xuất huyết Dengue lây qua muỗi vằn."
    rows = [
        {
            "id": 1,
            "relevant_docs": [583],
            "relevant_chunks": [
                {"doc_id": 583, "chunk_text": text},
                {"doc_id": 583, "chunk_text": text},
            ],
        },
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is False
    assert any("duplicate" in err.lower() and "chunk" in err.lower() for err in report.errors)


# ---------------------------------------------------------------------------
# 5. Missing lists and valid empty predictions
# ---------------------------------------------------------------------------

def test_missing_lists_rejected(store_and_queries):
    store, queries, _, _ = store_and_queries
    # missing relevant_docs
    rows1 = [
        {"id": 1, "relevant_chunks": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report1 = validate_predictions(rows1, queries, store)
    assert report1.valid is False
    assert any("relevant_docs" in err.lower() for err in report1.errors)

    # missing relevant_chunks
    rows2 = [
        {"id": 1, "relevant_docs": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report2 = validate_predictions(rows2, queries, store)
    assert report2.valid is False
    assert any("relevant_chunks" in err.lower() for err in report2.errors)

    # relevant_docs is not a list (e.g. set)
    rows3 = [
        {"id": 1, "relevant_docs": {583}, "relevant_chunks": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report3 = validate_predictions(rows3, queries, store)
    assert report3.valid is False


def test_valid_empty_predictions_pass(store_and_queries):
    store, queries, _, _ = store_and_queries
    rows = [
        {"id": 1, "relevant_docs": [], "relevant_chunks": []},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is True
    assert report.document_count == 0
    assert report.chunk_count == 0


# ---------------------------------------------------------------------------
# 6. Chunk exact source span and provenance
# ---------------------------------------------------------------------------

def test_reject_modified_chunk_text(store_and_queries):
    store, queries, _, _ = store_and_queries
    # chunk text modified from original doc
    rows = [
        {
            "id": 1,
            "relevant_docs": [583],
            "relevant_chunks": [{"doc_id": 583, "chunk_text": "Đây là text bị bịa đặt không có trong tài liệu"}],
        },
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is False
    assert any("substring" in err.lower() or "provenance" in err.lower() or "not found in source" in err.lower() for err in report.errors)


def test_chunk_with_official_doc_not_in_relevant_docs_accepted(store_and_queries):
    store, queries, _, _ = store_and_queries
    # Chunk from doc 4294967297, but relevant_docs only contains [583]
    rows = [
        {
            "id": 1,
            "relevant_docs": [583],
            "relevant_chunks": [{"doc_id": 4294967297, "chunk_text": "Triệu chứng gồm sốt cao, đau cơ và phát ban."}],
        },
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is True
    assert report.chunk_count == 1


# ---------------------------------------------------------------------------
# 7. chunk_order validation
# ---------------------------------------------------------------------------

def test_valid_chunk_order(store_and_queries):
    store, queries, _, _ = store_and_queries
    rows = [
        {
            "id": 1,
            "relevant_docs": [583],
            "relevant_chunks": [{"doc_id": 583, "chunk_text": "Bệnh sốt xuất huyết Dengue lây qua muỗi vằn.", "chunk_order": 0}],
        },
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is True


@pytest.mark.parametrize("bad_order", [-1, True, False, 1.5, "0"])
def test_reject_invalid_chunk_order(store_and_queries, bad_order):
    store, queries, _, _ = store_and_queries
    rows = [
        {
            "id": 1,
            "relevant_docs": [583],
            "relevant_chunks": [{"doc_id": 583, "chunk_text": "Bệnh sốt xuất huyết Dengue lây qua muỗi vằn.", "chunk_order": bad_order}],
        },
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report = validate_predictions(rows, queries, store)
    assert report.valid is False
    assert any("chunk_order" in err.lower() for err in report.errors)


# ---------------------------------------------------------------------------
# 8. ZIP validation rules
# ---------------------------------------------------------------------------

def test_validate_zip_valid(tmp_path: Path, store_and_queries):
    store, queries, _, _ = store_and_queries
    rows = [
        {"id": 1, "relevant_docs": [583], "relevant_chunks": [{"doc_id": 583, "chunk_text": "Bệnh sốt xuất huyết Dengue lây qua muỗi vằn."}]},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.json", json.dumps(rows, ensure_ascii=False, indent=2) + "\n")

    report = validate_zip(zip_path, queries, store)
    assert report.valid is True
    assert report.query_count == 2
    assert report.snapshot_id == store.snapshot_id


def test_reject_zip_with_extra_files(tmp_path: Path, store_and_queries):
    store, queries, _, _ = store_and_queries
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("results.json", json.dumps([{"id": 1, "relevant_docs": [], "relevant_chunks": []}, {"id": 2, "relevant_docs": [], "relevant_chunks": []}]))
        zf.writestr("extra.txt", "bonus file")

    report = validate_zip(zip_path, queries, store)
    assert report.valid is False
    assert any("exactly one" in err.lower() or "extra" in err.lower() for err in report.errors)


def test_reject_zip_with_subdirectories(tmp_path: Path, store_and_queries):
    store, queries, _, _ = store_and_queries
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("sub/results.json", json.dumps([{"id": 1, "relevant_docs": [], "relevant_chunks": []}, {"id": 2, "relevant_docs": [], "relevant_chunks": []}]))

    report = validate_zip(zip_path, queries, store)
    assert report.valid is False
    assert any("directory" in err.lower() or "results.json" in err.lower() for err in report.errors)


def test_reject_zip_with_wrong_filename(tmp_path: Path, store_and_queries):
    store, queries, _, _ = store_and_queries
    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("submission.json", "[]")

    report = validate_zip(zip_path, queries, store)
    assert report.valid is False
    assert any("results.json" in err.lower() for err in report.errors)


def test_reject_corrupt_or_non_zip(tmp_path: Path, store_and_queries):
    store, queries, _, _ = store_and_queries
    bad_file = tmp_path / "corrupt.zip"
    bad_file.write_bytes(b"not a real zip file header")

    report = validate_zip(bad_file, queries, store)
    assert report.valid is False
    assert any("zip" in err.lower() for err in report.errors)


def test_reject_encrypted_zip(tmp_path: Path, store_and_queries):
    store, queries, _, _ = store_and_queries
    zip_path = tmp_path / "encrypted.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("results.json", b"[]")

    # Flip bit 0 of general purpose flag in Central Directory header (PK\x01\x02)
    data = bytearray(zip_path.read_bytes())
    cd_idx = data.find(b"PK\x01\x02")
    assert cd_idx != -1
    data[cd_idx + 8] = 1
    zip_path.write_bytes(data)

    report = validate_zip(zip_path, queries, store)
    assert report.valid is False
    assert any("encrypt" in err.lower() for err in report.errors)




# ---------------------------------------------------------------------------
# 9. VibiomirExporter and Determinism
# ---------------------------------------------------------------------------

def test_vibiomir_exporter_end_to_end_and_determinism(tmp_path: Path, store_and_queries):
    store, queries, chunk1, chunk2 = store_and_queries
    predictions = [
        FinalPrediction(
            query_id="2",
            document_ids=["4294967297"],
            chunk_ids=[chunk2.chunk_id],
            document_scores={"4294967297": 0.8},
            chunk_scores={chunk2.chunk_id: 0.8},
        ),
        FinalPrediction(
            query_id="1",
            document_ids=["583", "4294967297"],
            chunk_ids=[chunk1.chunk_id],
            document_scores={"583": 0.95, "4294967297": 0.5},
            chunk_scores={chunk1.chunk_id: 0.95},
        ),
    ]

    out_dir_1 = tmp_path / "export_1"
    out_dir_2 = tmp_path / "export_2"

    exporter = VibiomirExporter()
    res1 = exporter.export(predictions, queries, store, out_dir_1)
    res2 = exporter.export(predictions, queries, store, out_dir_2)

    assert isinstance(res1, ExportResult)
    assert res1.json_path.exists()
    assert res1.zip_path.exists()
    assert res1.validation_report_path.exists()

    # Exact deterministic checksums between runs
    assert res1.json_sha256 == res2.json_sha256
    assert res1.zip_sha256 == res2.zip_sha256

    # Verify JSON content & formatting
    json_bytes = res1.json_path.read_bytes()
    assert json_bytes.endswith(b"\n")
    data = json.loads(json_bytes.decode("utf-8"))

    # Sorted by query ID ascending: query 1 before query 2
    assert [row["id"] for row in data] == [1, 2]
    assert data[0]["relevant_docs"] == [583, 4294967297]
    assert len(data[0]["relevant_chunks"]) == 1
    assert data[0]["relevant_chunks"][0]["doc_id"] == 583
    assert data[0]["relevant_chunks"][0]["chunk_text"] == chunk1.text
    assert data[0]["relevant_chunks"][0]["chunk_order"] == 0

    # Verify ZIP structure
    with zipfile.ZipFile(res1.zip_path, "r") as zf:
        infolist = zf.infolist()
        assert len(infolist) == 1
        assert infolist[0].filename == "results.json"
        assert zf.read("results.json") == json_bytes

    # Validate output ZIP with validate_zip
    report = validate_zip(res1.zip_path, queries, store)
    assert report.valid is True


def test_exporter_rejects_orphan_chunk_id(tmp_path: Path, store_and_queries):
    store, queries, _, _ = store_and_queries
    predictions = [
        FinalPrediction(
            query_id="1",
            document_ids=["583"],
            chunk_ids=["non_existing_chunk_id_xyz"],
            document_scores={"583": 0.9},
            chunk_scores={"non_existing_chunk_id_xyz": 0.9},
        ),
        FinalPrediction(
            query_id="2",
            document_ids=[],
            chunk_ids=[],
            document_scores={},
            chunk_scores={},
        ),
    ]
    exporter = VibiomirExporter()
    with pytest.raises(SubmissionValidationError) as exc_info:
        exporter.export(predictions, queries, store, tmp_path / "export_fail")
    assert any("orphan" in err.lower() or "not found" in err.lower() for err in exc_info.value.report.errors)


def test_exporter_rejects_tampered_chunk_provenance(tmp_path: Path, store_and_queries):
    store, queries, chunk1, _ = store_and_queries
    # Corrupt chunk text in store to simulate provenance mismatch with doc
    bad_chunk = SpanChunk(
        chunk_id=chunk1.chunk_id,
        doc_id=chunk1.doc_id,
        source_version=chunk1.source_version,
        source_sha256="mismatched_sha256",
        start_char=chunk1.start_char,
        end_char=chunk1.end_char,
        chunk_index=chunk1.chunk_index,
        language=chunk1.language,
        section=chunk1.section,
        text="Văn bản đã bị sửa đổi trái phép",
        index_text=chunk1.index_text,
        dense_token_count=10,
        reranker_token_count=10,
    )
    store.chunks[chunk1.chunk_id] = bad_chunk

    predictions = [
        FinalPrediction(
            query_id="1",
            document_ids=["583"],
            chunk_ids=[chunk1.chunk_id],
            document_scores={"583": 0.9},
            chunk_scores={chunk1.chunk_id: 0.9},
        ),
        FinalPrediction(query_id="2", document_ids=[], chunk_ids=[], document_scores={}, chunk_scores={}),
    ]
    exporter = VibiomirExporter()
    with pytest.raises(SubmissionValidationError) as exc_info:
        exporter.export(predictions, queries, store, tmp_path / "export_fail2")
    assert exc_info.value.report.valid is False
    assert any("provenance" in err.lower() or "mismatch" in err.lower() or "source" in err.lower() for err in exc_info.value.report.errors)


def test_exporter_rejects_boolean_query_id_and_doc_id(tmp_path: Path, store_and_queries):
    store, queries, chunk1, _ = store_and_queries
    # Boolean query_id
    predictions_bool_qid = [
        FinalPrediction(
            query_id=True,
            document_ids=["583"],
            chunk_ids=[chunk1.chunk_id],
            document_scores={"583": 0.9},
            chunk_scores={chunk1.chunk_id: 0.9},
        ),
        FinalPrediction(query_id="2", document_ids=[], chunk_ids=[], document_scores={}, chunk_scores={}),
    ]
    exporter = VibiomirExporter()
    with pytest.raises(SubmissionValidationError) as exc:
        exporter.export(predictions_bool_qid, queries, store, tmp_path / "exp_bool_q")
    assert any("boolean" in err.lower() for err in exc.value.report.errors)

    # Boolean doc_id
    predictions_bool_did = [
        FinalPrediction(
            query_id="1",
            document_ids=[True],
            chunk_ids=[chunk1.chunk_id],
            document_scores={"True": 0.9},
            chunk_scores={chunk1.chunk_id: 0.9},
        ),
        FinalPrediction(query_id="2", document_ids=[], chunk_ids=[], document_scores={}, chunk_scores={}),
    ]
    with pytest.raises(SubmissionValidationError) as exc:
        exporter.export(predictions_bool_did, queries, store, tmp_path / "exp_bool_d")
    assert any("boolean" in err.lower() for err in exc.value.report.errors)


def test_validator_rejects_unauthorized_extra_keys_in_row_and_chunk(store_and_queries):
    store, queries, _, _ = store_and_queries
    # Extra key in row
    rows_extra_row_key = [
        {"id": 1, "relevant_docs": [583], "relevant_chunks": [], "score": 0.95},
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report1 = validate_predictions(rows_extra_row_key, queries, store)
    assert report1.valid is False
    assert any("unauthorized extra keys" in err.lower() and "score" in err.lower() for err in report1.errors)

    # Extra key in chunk
    rows_extra_chunk_key = [
        {
            "id": 1,
            "relevant_docs": [583],
            "relevant_chunks": [
                {
                    "doc_id": 583,
                    "chunk_text": "Bệnh viện Bạch Mai là bệnh viện đa khoa hoàn chỉnh",
                    "chunk_order": 0,
                    "confidence_score": 0.99,
                }
            ],
        },
        {"id": 2, "relevant_docs": [], "relevant_chunks": []},
    ]
    report2 = validate_predictions(rows_extra_chunk_key, queries, store)
    assert report2.valid is False
    assert any("unauthorized extra keys" in err.lower() and "confidence_score" in err.lower() for err in report2.errors)
