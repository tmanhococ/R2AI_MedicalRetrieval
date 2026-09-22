from pathlib import Path
import json
import pytest

from src.data.dataset_adapter import DatasetAdapter, JsonlDatasetAdapter, InMemoryDatasetAdapter
from src.data.schemas import DocumentRecord, ChunkRecord, QueryRecord, GroundTruthRecord


def test_jsonl_dataset_adapter_loads_records(tmp_path: Path):
    docs_file = tmp_path / "documents.jsonl"
    docs_file.write_text(
        json.dumps({"doc_id": "d1", "language": "vi", "text": "Nội dung"}) + "\n",
        encoding="utf-8",
    )

    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(
        json.dumps({"chunk_id": "d1_c0", "doc_id": "d1", "language": "vi", "text": "Nội dung"}) + "\n",
        encoding="utf-8",
    )

    queries_file = tmp_path / "queries_val.jsonl"
    queries_file.write_text(
        json.dumps({"query_id": "q1", "text_vi": "Triệu chứng?"}) + "\n",
        encoding="utf-8",
    )

    gt_file = tmp_path / "qrels_val.jsonl"
    gt_file.write_text(
        json.dumps({"query_id": "q1", "relevant_doc_ids": ["d1"], "relevant_chunk_ids": ["d1_c0"]}) + "\n",
        encoding="utf-8",
    )

    adapter = JsonlDatasetAdapter(
        docs_path=docs_file,
        chunks_path=chunks_file,
        queries_dir=tmp_path,
        qrels_dir=tmp_path,
    )
    docs = adapter.load_documents()
    chunks = adapter.load_chunks()
    queries = adapter.load_queries("val")
    gt = adapter.load_ground_truth("val")

    assert len(docs) == 1 and docs[0].doc_id == "d1"
    assert len(chunks) == 1 and chunks[0].chunk_id == "d1_c0"
    assert len(queries) == 1 and queries[0].query_id == "q1"
    assert len(gt) == 1 and "d1" in gt[0].relevant_doc_ids


def test_jsonl_dataset_adapter_file_not_found():
    adapter = JsonlDatasetAdapter(
        docs_path="non_existent/docs.jsonl",
        chunks_path=None,
    )
    with pytest.raises(FileNotFoundError):
        adapter.load_documents()


def test_jsonl_dataset_adapter_chunks_none(tmp_path: Path):
    docs_file = tmp_path / "documents.jsonl"
    docs_file.write_text(
        json.dumps({"doc_id": "d1", "language": "vi", "text": "Nội dung"}) + "\n",
        encoding="utf-8",
    )
    adapter = JsonlDatasetAdapter(docs_path=docs_file, chunks_path=None)
    assert adapter.load_chunks() == []


def test_jsonl_dataset_adapter_chunks_file_not_found():
    adapter = JsonlDatasetAdapter(chunks_path="non_existent/chunks.jsonl")
    with pytest.raises(FileNotFoundError):
        adapter.load_chunks()


def test_jsonl_dataset_adapter_path_is_directory(tmp_path: Path):
    adapter = JsonlDatasetAdapter(docs_path=tmp_path, chunks_path=tmp_path)
    with pytest.raises(ValueError, match="not a file"):
        adapter.load_documents()
    with pytest.raises(ValueError, match="not a file"):
        adapter.load_chunks()


def test_jsonl_dataset_adapter_ground_truth_filename_variants(tmp_path: Path):
    gt_file = tmp_path / "ground_truth_test.jsonl"
    gt_file.write_text(
        json.dumps({"query_id": "q2", "relevant_doc_ids": ["d2"], "relevant_chunk_ids": ["d2_c1"]}) + "\n",
        encoding="utf-8",
    )
    adapter = JsonlDatasetAdapter(qrels_dir=tmp_path)
    gt = adapter.load_ground_truth("test")
    assert len(gt) == 1
    assert gt[0].query_id == "q2"
    assert "d2" in gt[0].relevant_doc_ids
    assert "d2_c1" in gt[0].relevant_chunk_ids


def test_jsonl_dataset_adapter_missing_queries_and_qrels():
    adapter = JsonlDatasetAdapter()
    with pytest.raises(FileNotFoundError):
        adapter.load_queries("val")
    with pytest.raises(FileNotFoundError):
        adapter.load_ground_truth("val")


def test_jsonl_dataset_adapter_split_files_not_found(tmp_path: Path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    adapter = JsonlDatasetAdapter(queries_dir=empty_dir, qrels_dir=empty_dir)
    with pytest.raises(FileNotFoundError, match="Queries file not found"):
        adapter.load_queries("missing_split")
    with pytest.raises(FileNotFoundError, match="Ground truth file not found"):
        adapter.load_ground_truth("missing_split")


def test_jsonl_dataset_adapter_normalization(tmp_path: Path):
    docs_file = tmp_path / "docs.jsonl"
    docs_file.write_text(
        json.dumps({"doc_id": 123, "language": "VI", "text": "Đoạn văn", "title": "Tiêu đề"}) + "\n",
        encoding="utf-8",
    )
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(
        json.dumps({"chunk_id": 456, "doc_id": 123, "language": "Vietnamese", "text": "Đoạn văn nhỏ"}) + "\n",
        encoding="utf-8",
    )
    adapter = JsonlDatasetAdapter(docs_path=docs_file, chunks_path=chunks_file)
    docs = adapter.load_documents()
    chunks = adapter.load_chunks()

    assert docs[0].doc_id == "123"
    assert docs[0].language == "vi"
    assert docs[0].title == "Tiêu đề"
    assert chunks[0].chunk_id == "456"
    assert chunks[0].doc_id == "123"
    assert chunks[0].language == "vi"


def test_jsonl_dataset_adapter_invalid_language(tmp_path: Path):
    docs_file = tmp_path / "docs.jsonl"
    docs_file.write_text(
        json.dumps({"doc_id": "d1", "language": "fr", "text": "Bonjour"}) + "\n",
        encoding="utf-8",
    )
    adapter = JsonlDatasetAdapter(docs_path=docs_file)
    with pytest.raises(ValueError, match="Invalid language"):
        adapter.load_documents()


def test_jsonl_dataset_adapter_empty_lines(tmp_path: Path):
    docs_file = tmp_path / "docs.jsonl"
    docs_file.write_text(
        "\n" + json.dumps({"doc_id": "d1", "language": "vi", "text": "Nội dung"}) + "\n\n   \n",
        encoding="utf-8",
    )
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text(
        "\n" + json.dumps({"chunk_id": "c1", "doc_id": "d1", "language": "vi", "text": "Đoạn"}) + "\n\n",
        encoding="utf-8",
    )
    queries_file = tmp_path / "queries_val.jsonl"
    queries_file.write_text(
        "\n" + json.dumps({"query_id": "q1", "text_vi": "Hỏi"}) + "\n\n",
        encoding="utf-8",
    )
    gt_file = tmp_path / "qrels_val.jsonl"
    gt_file.write_text(
        "\n" + json.dumps({"query_id": "q1", "relevant_doc_ids": ["d1"]}) + "\n\n",
        encoding="utf-8",
    )
    adapter = JsonlDatasetAdapter(
        docs_path=docs_file,
        chunks_path=chunks_file,
        queries_dir=tmp_path,
        qrels_dir=tmp_path,
    )
    assert len(adapter.load_documents()) == 1
    assert len(adapter.load_chunks()) == 1
    assert len(adapter.load_queries("val")) == 1
    assert len(adapter.load_ground_truth("val")) == 1


def test_jsonl_dataset_adapter_direct_file_and_fallbacks(tmp_path: Path):
    queries_file = tmp_path / "custom_queries.jsonl"
    queries_file.write_text(
        json.dumps({"query_id": "q1", "text_vi": "Câu hỏi"}) + "\n",
        encoding="utf-8",
    )
    qrels_file = tmp_path / "custom_qrels.jsonl"
    qrels_file.write_text(
        json.dumps({"query_id": "q1", "relevant_doc_ids": ["d1"]}) + "\n",
        encoding="utf-8",
    )
    adapter = JsonlDatasetAdapter(queries_dir=queries_file, qrels_dir=qrels_file)
    assert len(adapter.load_queries("val")) == 1
    assert len(adapter.load_ground_truth("val")) == 1

    sub_dir = tmp_path / "split_dir"
    sub_dir.mkdir()
    (sub_dir / "train.jsonl").write_text(
        json.dumps({"query_id": "q_tr", "text_vi": "Hỏi train"}) + "\n",
        encoding="utf-8",
    )
    (sub_dir / "dev.jsonl").write_text(
        json.dumps({"query_id": "q_dev", "relevant_doc_ids": ["d1"]}) + "\n",
        encoding="utf-8",
    )
    adapter_split = JsonlDatasetAdapter(queries_dir=sub_dir, qrels_dir=sub_dir)
    assert len(adapter_split.load_queries("train")) == 1
    assert len(adapter_split.load_ground_truth("dev")) == 1


def test_in_memory_dataset_adapter():
    doc = DocumentRecord(doc_id="d1", language="en", text="Sample text")
    chunk = ChunkRecord(chunk_id="d1_c0", doc_id="d1", language="en", text="Sample")
    query = QueryRecord(query_id="q1", text_vi="Truy vấn")
    gt = GroundTruthRecord(query_id="q1", relevant_doc_ids={"d1"}, relevant_chunk_ids={"d1_c0"})

    adapter = InMemoryDatasetAdapter(
        documents=[doc],
        chunks=[chunk],
        queries={"val": [query]},
        ground_truth={"val": [gt]},
    )

    assert adapter.load_documents() == [doc]
    assert adapter.load_chunks() == [chunk]
    assert adapter.load_queries("val") == [query]
    assert adapter.load_queries("test") == []
    assert adapter.load_ground_truth("val") == [gt]
    assert adapter.load_ground_truth("test") == []


def test_dataset_adapter_protocol_conformance(tmp_path: Path):
    docs_file = tmp_path / "documents.jsonl"
    docs_file.write_text(
        json.dumps({"doc_id": "d1", "language": "vi", "text": "Nội dung"}) + "\n",
        encoding="utf-8",
    )
    jsonl_adapter = JsonlDatasetAdapter(docs_path=docs_file)
    mem_adapter = InMemoryDatasetAdapter()

    assert isinstance(jsonl_adapter, DatasetAdapter)
    assert isinstance(mem_adapter, DatasetAdapter)
