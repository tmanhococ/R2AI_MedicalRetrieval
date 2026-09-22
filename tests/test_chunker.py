import pytest
from src.data.schemas import DocumentRecord, ChunkRecord
from src.indexing.chunker import (
    Chunker,
    AutoChunker,
    HybridChunker,
    PrechunkedChunker,
    PrechunkedAdapter,
    create_chunker,
)
from src.utils.config import load_config


def test_auto_chunker_sliding_window():
    long_text = " ".join([f"word{i}" for i in range(100)])
    doc = DocumentRecord(doc_id="doc_long", language="vi", text=long_text)
    chunker = AutoChunker(chunk_size=40, chunk_overlap=10, min_chunk_length=5)
    chunks = chunker.chunk_documents([doc])

    assert len(chunks) >= 3
    assert chunks[0].chunk_id == "doc_long_c0"
    assert chunks[0].doc_id == "doc_long"
    assert chunks[0].language == "vi"
    assert chunks[1].chunk_index == 1
    assert chunks[0].metadata is not None
    assert chunks[0].metadata.get("is_auto_generated") is True


def test_hybrid_chunker_preserves_existing_chunks():
    docs = [
        DocumentRecord(doc_id="d1", language="vi", text="Doc 1 text"),
        DocumentRecord(doc_id="d2", language="en", text="Doc 2 " * 30),
    ]
    existing_chunks = [ChunkRecord(chunk_id="d1_custom", doc_id="d1", language="vi", text="Doc 1 text")]
    chunker = HybridChunker(auto_chunker=AutoChunker(chunk_size=20, chunk_overlap=5))
    result = chunker.chunk_documents(docs, existing_chunks=existing_chunks)

    # d1 uses existing chunk, d2 gets auto chunked
    d1_chunks = [c for c in result if c.doc_id == "d1"]
    d2_chunks = [c for c in result if c.doc_id == "d2"]
    assert len(d1_chunks) == 1 and d1_chunks[0].chunk_id == "d1_custom"
    assert len(d2_chunks) > 1


def test_create_chunker_factory():
    cfg_auto = {
        "indexing": {
            "chunking": {
                "mode": "auto",
                "auto_strategy": {"chunk_size": 100, "chunk_overlap": 20, "min_chunk_length": 10},
            }
        }
    }
    c = create_chunker(cfg_auto)
    assert isinstance(c, AutoChunker)
    assert c.chunk_size == 100
    assert c.chunk_overlap == 20
    assert c.min_chunk_length == 10

    cfg_hybrid = {
        "indexing": {
            "chunking": {
                "mode": "hybrid",
                "auto_strategy": {"chunk_size": 512, "chunk_overlap": 64, "min_chunk_length": 50},
            }
        }
    }
    c_hybrid = create_chunker(cfg_hybrid)
    assert isinstance(c_hybrid, HybridChunker)
    assert isinstance(c_hybrid.auto_chunker, AutoChunker)
    assert c_hybrid.auto_chunker.chunk_size == 512

    cfg_prechunked = {
        "indexing": {
            "chunking": {
                "mode": "prechunked",
            }
        }
    }
    c_prechunked = create_chunker(cfg_prechunked)
    assert isinstance(c_prechunked, PrechunkedChunker)


def test_create_chunker_from_default_yaml():
    cfg = load_config("configs/r2ai_mvp.yaml")
    chunker = create_chunker(cfg)
    assert isinstance(chunker, HybridChunker)
    assert isinstance(chunker.auto_chunker, AutoChunker)
    assert chunker.auto_chunker.chunk_size == 512
    assert chunker.auto_chunker.chunk_overlap == 64
    assert chunker.auto_chunker.min_chunk_length == 50


def test_create_chunker_unknown_mode():
    with pytest.raises(ValueError, match="Unknown chunking mode"):
        create_chunker({"indexing": {"chunking": {"mode": "unsupported_mode"}}})


def test_protocol_conformance():
    auto = AutoChunker()
    hybrid = HybridChunker()
    prechunked = PrechunkedChunker()

    assert isinstance(auto, Chunker)
    assert isinstance(hybrid, Chunker)
    assert isinstance(prechunked, Chunker)
    assert PrechunkedAdapter is PrechunkedChunker


def test_prechunked_chunker():
    docs = [
        DocumentRecord(doc_id="d1", language="vi", text="Văn bản 1"),
        DocumentRecord(doc_id="d2", language="vi", text="Văn bản 2"),
    ]
    chunks = [
        ChunkRecord(chunk_id="d1_c0", doc_id="d1", language="vi", text="Đoạn 1"),
        ChunkRecord(chunk_id="d2_c0", doc_id="d2", language="vi", text="Đoạn 2"),
        ChunkRecord(chunk_id="d3_c0", doc_id="d3", language="vi", text="Đoạn ngoài"),
    ]
    chunker = PrechunkedChunker()
    res = chunker.chunk_documents(docs, existing_chunks=chunks)
    assert len(res) == 2
    assert {c.chunk_id for c in res} == {"d1_c0", "d2_c0"}

    # Default chunks in constructor
    chunker_with_defaults = PrechunkedChunker(default_chunks=chunks)
    res_def = chunker_with_defaults.chunk_documents(docs)
    assert len(res_def) == 2

    # Non-strict mode when no chunks provided returns empty list
    chunker_non_strict = PrechunkedChunker(strict=False)
    assert chunker_non_strict.chunk_documents(docs) == []

    # Strict mode error when no chunks
    strict_chunker = PrechunkedChunker(strict=True)
    with pytest.raises(ValueError, match="No existing chunks"):
        strict_chunker.chunk_documents(docs)


def test_create_chunker_alternate_config_structures():
    # Direct "chunking" root
    c1 = create_chunker({"chunking": {"mode": "auto"}})
    assert isinstance(c1, AutoChunker)

    # Flat dict
    c2 = create_chunker({"mode": "prechunked", "strict": True})
    assert isinstance(c2, PrechunkedChunker)
    assert c2.strict is True


def test_auto_chunker_validation():
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        AutoChunker(chunk_size=0)

    with pytest.raises(ValueError, match="chunk_overlap must be non-negative"):
        AutoChunker(chunk_overlap=-1)

    with pytest.raises(ValueError, match="chunk_overlap must be strictly less than chunk_size"):
        AutoChunker(chunk_size=50, chunk_overlap=50)

    with pytest.raises(ValueError, match="min_chunk_length must be non-negative"):
        AutoChunker(min_chunk_length=-1)


def test_auto_chunker_empty_and_short_docs():
    chunker = AutoChunker(chunk_size=50, chunk_overlap=10, min_chunk_length=5)
    doc_empty = DocumentRecord(doc_id="d_empty", language="vi", text="   ")
    assert chunker.chunk_documents([doc_empty]) == []

    # Below min_chunk_length
    doc_short = DocumentRecord(doc_id="d_short", language="vi", text="Một hai ba")
    assert chunker.chunk_documents([doc_short]) == []


def test_auto_chunker_metadata_preservation():
    doc = DocumentRecord(
        doc_id="d_meta",
        language="en",
        text="A B C D E F G H I J",
        metadata={"source": "who", "section": "cardiology"},
    )
    chunker = AutoChunker(chunk_size=5, chunk_overlap=1, min_chunk_length=2)
    chunks = chunker.chunk_documents([doc])
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.metadata.get("source") == "who"
        assert chunk.metadata.get("section") == "cardiology"
        assert chunk.metadata.get("is_auto_generated") is True


def test_auto_chunker_chinese_text():
    # Chinese text without spaces
    doc_zh = DocumentRecord(
        doc_id="d_zh",
        language="zh",
        text="患者有高血压病史三年经常头晕头痛伴随胸闷气短",
    )
    chunker = AutoChunker(chunk_size=10, chunk_overlap=2, min_chunk_length=3)
    chunks = chunker.chunk_documents([doc_zh])
    assert len(chunks) >= 2
    assert chunks[0].language == "zh"
    assert chunks[0].chunk_id == "d_zh_c0"
    assert chunks[0].metadata.get("is_auto_generated") is True
