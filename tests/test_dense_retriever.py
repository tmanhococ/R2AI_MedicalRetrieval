from pathlib import Path
from typing import Any
import numpy as np
import pytest

from src.data.schemas import ChunkRecord, SearchHit
from src.indexing.dense_index import DenseIndexer
from src.retrieval.dense_retriever import DenseRetriever


class DummyEncoder:
    """Mock encoder that generates deterministic embeddings for testing."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self.encoded_texts: list[str] = []

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        self.encoded_texts.extend(texts)
        vecs = []
        for t in texts:
            # Deterministic vector based on length and characters
            val = float(len(t) % 10) / 10.0 + 0.1
            v = np.full(self.dim, val, dtype=np.float32)
            # Add character sum variation so different texts have distinct values
            char_factor = (sum(ord(c) for c in t) % 100) / 1000.0
            v[0] += char_factor
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                v = v / (norm if norm > 0 else 1.0)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


def test_dense_indexing_and_search_numpy(tmp_path: Path):
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="đoạn văn bản một"),
        ChunkRecord(chunk_id="c2", doc_id="d2", language="vi", text="đoạn văn bản dài hơn nhiều hai"),
    ]
    encoder = DummyEncoder()
    indexer = DenseIndexer(encoder=encoder, backend="numpy")
    index_dir = tmp_path / "dense_index_numpy"
    indexer.build_index(chunks=chunks, output_dir=index_dir)

    retriever = DenseRetriever(encoder=encoder, index_dir=index_dir, backend="numpy")
    chunk_hits = retriever.search_chunks(query="đoạn văn bản một", language="vi", top_k=2)
    assert len(chunk_hits) == 2
    assert chunk_hits[0].item_id in {"c1", "c2"}
    assert chunk_hits[0].item_type == "chunk"
    assert chunk_hits[0].source == "dense"
    assert chunk_hits[0].rank == 1
    assert chunk_hits[1].rank == 2
    assert chunk_hits[0].raw_score >= chunk_hits[1].raw_score

    # Document aggregation (max score)
    doc_hits = retriever.aggregate_documents(chunk_hits)
    assert len(doc_hits) == 2
    assert doc_hits[0].item_type == "document"
    assert doc_hits[0].doc_id in {"d1", "d2"}
    assert doc_hits[0].rank == 1
    assert doc_hits[1].rank == 2
    assert doc_hits[0].raw_score >= doc_hits[1].raw_score


def test_dense_indexing_and_search_faiss(tmp_path: Path):
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="đoạn văn bản một"),
        ChunkRecord(chunk_id="c2", doc_id="d2", language="vi", text="đoạn văn bản dài hơn nhiều hai"),
    ]
    encoder = DummyEncoder()
    indexer = DenseIndexer(encoder=encoder, backend="faiss")
    index_dir = tmp_path / "dense_index_faiss"
    manifest = indexer.build_index(chunks=chunks, output_dir=index_dir)
    assert manifest["backend"] == "faiss"
    assert (index_dir / "faiss_vi.index").exists()

    retriever = DenseRetriever(encoder=encoder, index_dir=index_dir, backend="faiss")
    chunk_hits = retriever.search_chunks(query="đoạn văn bản một", language="vi", top_k=2)
    assert len(chunk_hits) == 2
    assert chunk_hits[0].item_id in {"c1", "c2"}
    assert chunk_hits[0].item_type == "chunk"
    assert chunk_hits[0].source == "dense"
    assert chunk_hits[0].rank == 1


def test_dense_instruction_prefix_support(tmp_path: Path):
    encoder = DummyEncoder()
    prefix_config = {"query": "query: ", "passage": "passage: "}
    indexer = DenseIndexer(encoder=encoder, backend="numpy", instruction_prefix=prefix_config)
    index_dir = tmp_path / "dense_prefix_index"
    chunks = [ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="text")]
    indexer.build_index(chunks=chunks, output_dir=index_dir)

    # Verify passage prefix was prepended during encoding
    assert any("passage: text" in t for t in encoder.encoded_texts)

    encoder.encoded_texts.clear()
    retriever = DenseRetriever(
        encoder=encoder,
        index_dir=index_dir,
        backend="numpy",
        instruction_prefix=prefix_config,
    )
    hits = retriever.search_chunks(query="test", language="vi", top_k=1)
    assert len(hits) == 1
    # Verify query prefix was prepended during query encoding
    assert any("query: test" in t for t in encoder.encoded_texts)


def test_dense_empty_language_partition(tmp_path: Path):
    encoder = DummyEncoder()
    indexer = DenseIndexer(encoder=encoder, backend="numpy")
    index_dir = tmp_path / "dense_empty"
    indexer.build_index(chunks=[], output_dir=index_dir)

    retriever = DenseRetriever(encoder=encoder, index_dir=index_dir, backend="numpy")
    hits = retriever.search_chunks(query="test", language="en", top_k=5)
    assert hits == []


def test_dense_document_aggregation_max_score(tmp_path: Path):
    retriever = DenseRetriever(encoder=DummyEncoder(), index_dir=tmp_path, backend="numpy")
    # c1 and c2 belong to doc d1, c3 belongs to d2
    hits = [
        SearchHit(
            item_id="c1",
            item_type="chunk",
            doc_id="d1",
            language="vi",
            source="dense",
            raw_score=0.92,
            rank=1,
        ),
        SearchHit(
            item_id="c3",
            item_type="chunk",
            doc_id="d2",
            language="vi",
            source="dense",
            raw_score=0.88,
            rank=2,
        ),
        SearchHit(
            item_id="c2",
            item_type="chunk",
            doc_id="d1",
            language="vi",
            source="dense",
            raw_score=0.55,
            rank=3,
        ),
    ]
    doc_hits = retriever.aggregate_documents(hits)
    assert len(doc_hits) == 2
    # d1 should have score 0.92, rank 1
    assert doc_hits[0].doc_id == "d1"
    assert doc_hits[0].item_id == "d1"
    assert doc_hits[0].item_type == "document"
    assert doc_hits[0].raw_score == pytest.approx(0.92)
    assert doc_hits[0].rank == 1
    assert doc_hits[0].source == "dense"
    assert doc_hits[0].language == "vi"

    # d2 should have score 0.88, rank 2
    assert doc_hits[1].doc_id == "d2"
    assert doc_hits[1].item_id == "d2"
    assert doc_hits[1].raw_score == pytest.approx(0.88)
    assert doc_hits[1].rank == 2


def test_dense_multilingual_partitioning(tmp_path: Path):
    chunks = [
        ChunkRecord(chunk_id="vi_c1", doc_id="vi_d1", language="vi", text="tiểu đường tuýp 2"),
        ChunkRecord(chunk_id="en_c1", doc_id="en_d1", language="en", text="type 2 diabetes"),
        ChunkRecord(chunk_id="zh_c1", doc_id="zh_d1", language="zh", text="2型糖尿病"),
    ]
    encoder = DummyEncoder()
    indexer = DenseIndexer(encoder=encoder, backend="numpy")
    index_dir = tmp_path / "dense_multilingual"
    manifest = indexer.build_index(chunks=chunks, output_dir=index_dir)

    assert manifest["partitions"]["vi"]["count"] == 1
    assert manifest["partitions"]["en"]["count"] == 1
    assert manifest["partitions"]["zh"]["count"] == 1

    retriever = DenseRetriever(encoder=encoder, index_dir=index_dir)
    assert retriever.backend == "numpy"  # auto-inferred from manifest

    vi_hits = retriever.search_chunks(query="tiểu đường", language="vi", top_k=2)
    assert len(vi_hits) == 1
    assert vi_hits[0].item_id == "vi_c1"

    en_hits = retriever.search_chunks(query="diabetes", language="en", top_k=2)
    assert len(en_hits) == 1
    assert en_hits[0].item_id == "en_c1"

    zh_hits = retriever.search_chunks(query="糖尿病", language="zh", top_k=2)
    assert len(zh_hits) == 1
    assert zh_hits[0].item_id == "zh_c1"


def test_dense_edge_cases(tmp_path: Path):
    encoder = DummyEncoder()
    indexer = DenseIndexer(encoder=encoder, backend="numpy")
    index_dir = tmp_path / "dense_edges"
    indexer.build_index(
        chunks=[ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="nội dung")],
        output_dir=index_dir,
    )

    retriever = DenseRetriever(encoder=encoder, index_dir=index_dir, backend="numpy")

    # Empty / whitespace query
    assert retriever.search_chunks("", language="vi") == []
    assert retriever.search_chunks("   ", language="vi") == []

    # top_k <= 0
    assert retriever.search_chunks("query", language="vi", top_k=0) == []
    assert retriever.search_chunks("query", language="vi", top_k=-5) == []

    # Non-existent language
    assert retriever.search_chunks("query", language="fr", top_k=5) == []

    # Non-existent index dir
    missing_retriever = DenseRetriever(
        encoder=encoder, index_dir=tmp_path / "non_existent", backend="numpy"
    )
    assert missing_retriever.search_chunks("query", language="vi") == []

    # Invalid backend
    with pytest.raises(ValueError, match="Unsupported dense backend"):
        DenseIndexer(encoder=encoder, backend="invalid_backend")

    with pytest.raises(ValueError, match="Unsupported dense backend"):
        DenseRetriever(encoder=encoder, index_dir=index_dir, backend="invalid_backend")


def test_dense_search_documents(tmp_path: Path):
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="triệu chứng suy tim"),
        ChunkRecord(chunk_id="c2", doc_id="d1", language="vi", text="điều trị suy tim với chẹn beta"),
        ChunkRecord(chunk_id="c3", doc_id="d2", language="vi", text="bệnh phổi tắc nghẽn mạn tính"),
    ]
    encoder = DummyEncoder()
    indexer = DenseIndexer(encoder=encoder, backend="numpy")
    index_dir = tmp_path / "dense_doc_search"
    indexer.build_index(chunks=chunks, output_dir=index_dir)

    retriever = DenseRetriever(encoder=encoder, index_dir=index_dir, backend="numpy")
    doc_hits = retriever.search_documents(query="suy tim", language="vi", top_k=2)
    assert len(doc_hits) <= 2
    assert doc_hits[0].item_type == "document"
    assert doc_hits[0].rank == 1

    # Empty query returns empty
    assert retriever.search_documents("", language="vi") == []
    assert retriever.search_documents("suy tim", language="vi", top_k=0) == []


def test_dense_backend_fallbacks(tmp_path: Path):
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="đoạn 1"),
        ChunkRecord(chunk_id="c2", doc_id="d2", language="vi", text="đoạn 2"),
    ]
    encoder = DummyEncoder()
    # Build faiss index (which saves both faiss_vi.index and chunk_embeddings_vi.npy)
    indexer = DenseIndexer(encoder=encoder, backend="faiss")
    index_dir = tmp_path / "dense_fallbacks"
    indexer.build_index(chunks=chunks, output_dir=index_dir)

    faiss_file = index_dir / "faiss_vi.index"
    npy_file = index_dir / "chunk_embeddings_vi.npy"
    assert faiss_file.exists()
    assert npy_file.exists()

    # Case 1: FAISS backend requested, but faiss_vi.index deleted -> in-memory fallback from .npy
    faiss_file.unlink()
    retriever_faiss = DenseRetriever(encoder=encoder, index_dir=index_dir, backend="faiss")
    hits = retriever_faiss.search_chunks("đoạn 1", language="vi", top_k=2)
    assert len(hits) == 2

    # Case 2: Numpy backend requested, but .npy deleted -> fallback reconstruct from faiss_vi.index
    # Re-build index so both exist again
    indexer.build_index(chunks=chunks, output_dir=index_dir)
    npy_file.unlink()
    retriever_numpy = DenseRetriever(encoder=encoder, index_dir=index_dir, backend="numpy")
    hits_np = retriever_numpy.search_chunks("đoạn 1", language="vi", top_k=2)
    assert len(hits_np) == 2


def test_dense_callable_encoder(tmp_path: Path):
    # Test functional encoder callable
    def my_callable_encoder(texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 8), dtype=np.float32)

    indexer = DenseIndexer(encoder=my_callable_encoder, backend="numpy")
    index_dir = tmp_path / "dense_callable"
    indexer.build_index(
        chunks=[ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="nội dung")],
        output_dir=index_dir,
    )

    retriever = DenseRetriever(encoder=my_callable_encoder, index_dir=index_dir, backend="numpy")
    hits = retriever.search_chunks("test", language="vi", top_k=1)
    assert len(hits) == 1
    assert hits[0].item_id == "c1"


def test_dense_string_model_name_mocked(tmp_path: Path, monkeypatch):
    class MockST:
        def __init__(self, model_name: str, device: str = "cpu"):
            self.model_name = model_name
            self.device = device

        def encode(self, texts, batch_size=32, normalize_embeddings=True):
            return np.ones((len(texts), 16), dtype=np.float32)

    import sys
    from types import ModuleType

    st_module = ModuleType("sentence_transformers")
    st_module.SentenceTransformer = MockST
    monkeypatch.setitem(sys.modules, "sentence_transformers", st_module)

    indexer = DenseIndexer(encoder="mock-model-name", backend="numpy", device="cpu")
    index_dir = tmp_path / "dense_mock_st"
    indexer.build_index(
        chunks=[ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="nội dung")],
        output_dir=index_dir,
    )

    retriever = DenseRetriever(
        encoder="mock-model-name", index_dir=index_dir, backend="numpy", device="cpu"
    )
    hits = retriever.search_chunks("test", language="vi", top_k=1)
    assert len(hits) == 1


def test_dense_invalid_encoder():
    with pytest.raises(ValueError, match="is neither callable nor"):
        indexer = DenseIndexer(encoder=12345, backend="numpy")
        indexer.build_index(
            chunks=[ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="text")],
            output_dir="tmp",
        )

