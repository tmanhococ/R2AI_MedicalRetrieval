from pathlib import Path
import pytest
from src.data.schemas import DocumentRecord, ChunkRecord
from src.indexing.bm25_index import BM25Indexer
from src.indexing.tokenizers import get_tokenizer
from src.retrieval.bm25_retriever import BM25Retriever


def test_bm25_indexing_and_search(tmp_path: Path):
    docs = [
        DocumentRecord(doc_id="d1", language="vi", text="Bệnh nhân đái tháo đường cần xét nghiệm HbA1c."),
        DocumentRecord(doc_id="d2", language="vi", text="Cao huyết áp vô căn điều trị bằng Amlodipine."),
    ]
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="đái tháo đường xét nghiệm HbA1c"),
        ChunkRecord(chunk_id="c2", doc_id="d2", language="vi", text="huyết áp điều trị Amlodipine"),
    ]
    indexer = BM25Indexer(backend="rank_bm25")  # test with standard backend
    index_dir = tmp_path / "bm25_index"
    indexer.build_indexes(documents=docs, chunks=chunks, output_dir=index_dir)

    retriever = BM25Retriever(index_dir=index_dir, backend="rank_bm25")
    hits = retriever.search_chunks(query="HbA1c đái tháo đường", language="vi", top_k=5)

    assert len(hits) > 0
    assert hits[0].item_id == "c1"
    assert hits[0].doc_id == "d1"
    assert hits[0].raw_score > 0
    assert hits[0].rank == 1
    assert hits[0].item_type == "chunk"
    assert hits[0].source == "bm25"
    assert hits[0].language == "vi"


def test_bm25_search_documents(tmp_path: Path):
    docs = [
        DocumentRecord(doc_id="d1", language="vi", text="Bệnh nhân đái tháo đường cần xét nghiệm HbA1c."),
        DocumentRecord(doc_id="d2", language="vi", text="Cao huyết áp vô căn điều trị bằng Amlodipine."),
    ]
    indexer = BM25Indexer(backend="rank_bm25")
    index_dir = tmp_path / "bm25_index"
    indexer.build_indexes(documents=docs, chunks=[], output_dir=index_dir)

    retriever = BM25Retriever(index_dir=index_dir, backend="rank_bm25")
    hits = retriever.search_documents(query="đái tháo đường", language="vi", top_k=2)
    assert len(hits) > 0
    assert hits[0].item_id == "d1"
    assert hits[0].doc_id == "d1"
    assert hits[0].item_type == "document"
    assert hits[0].rank == 1


def test_bm25_empty_partition_graceful(tmp_path: Path):
    indexer = BM25Indexer(backend="rank_bm25")
    index_dir = tmp_path / "bm25_index"
    indexer.build_indexes(documents=[], chunks=[], output_dir=index_dir)

    retriever = BM25Retriever(index_dir=index_dir, backend="rank_bm25")
    hits = retriever.search_chunks(query="test", language="en", top_k=5)
    assert hits == []


def test_bm25s_backend_indexing_and_search(tmp_path: Path):
    docs = [
        DocumentRecord(doc_id="en1", language="en", text="Patient diagnosed with acute myocardial infarction."),
        DocumentRecord(doc_id="en2", language="en", text="Management of chronic kidney disease stage 3."),
    ]
    chunks = [
        ChunkRecord(chunk_id="enc1", doc_id="en1", language="en", text="acute myocardial infarction troponin"),
        ChunkRecord(chunk_id="enc2", doc_id="en2", language="en", text="chronic kidney disease eGFR serum creatinine"),
    ]
    indexer = BM25Indexer(backend="bm25s")
    index_dir = tmp_path / "bm25s_index"
    manifest = indexer.build_indexes(documents=docs, chunks=chunks, output_dir=index_dir)

    assert manifest["backend"] == "bm25s"
    assert manifest["partitions"]["en_documents"]["count"] == 2
    assert manifest["partitions"]["en_chunks"]["count"] == 2

    # Retriever auto-infers backend from manifest
    retriever = BM25Retriever(index_dir=index_dir)
    assert retriever.backend == "bm25s"

    chunk_hits = retriever.search_chunks(query="myocardial infarction troponin", language="en", top_k=3)
    assert len(chunk_hits) > 0
    assert chunk_hits[0].item_id == "enc1"
    assert chunk_hits[0].doc_id == "en1"
    assert chunk_hits[0].raw_score > 0
    assert chunk_hits[0].rank == 1
    assert chunk_hits[0].item_type == "chunk"

    doc_hits = retriever.search_documents(query="kidney disease", language="en", top_k=3)
    assert len(doc_hits) > 0
    assert doc_hits[0].item_id == "en2"
    assert doc_hits[0].doc_id == "en2"
    assert doc_hits[0].item_type == "document"


def test_bm25_multilingual_partitions(tmp_path: Path):
    docs = [
        DocumentRecord(doc_id="vi_d1", language="vi", text="Viêm gan B mạn tính giai đoạn hoạt động"),
        DocumentRecord(doc_id="en_d1", language="en", text="Chronic active hepatitis B infection viral load"),
        DocumentRecord(doc_id="zh_d1", language="zh", text="慢性乙型病毒性肝炎活动期抗病毒治疗"),
    ]
    chunks = [
        ChunkRecord(chunk_id="vi_c1", doc_id="vi_d1", language="vi", text="Viêm gan B Tenofovir"),
        ChunkRecord(chunk_id="en_c1", doc_id="en_d1", language="en", text="Hepatitis B Tenofovir disoproxil"),
        ChunkRecord(chunk_id="zh_c1", doc_id="zh_d1", language="zh", text="乙肝 替诺福韦 治疗"),
    ]
    indexer = BM25Indexer(backend="rank_bm25")
    index_dir = tmp_path / "multi_bm25"
    indexer.build_indexes(documents=docs, chunks=chunks, output_dir=index_dir)

    retriever = BM25Retriever(index_dir=index_dir, backend="rank_bm25")

    # VI search
    hits_vi = retriever.search_chunks(query="Viêm gan B", language="vi", top_k=2)
    assert len(hits_vi) > 0
    assert hits_vi[0].item_id == "vi_c1"

    # EN search
    hits_en = retriever.search_chunks(query="Tenofovir hepatitis", language="en", top_k=2)
    assert len(hits_en) > 0
    assert hits_en[0].item_id == "en_c1"

    # ZH search
    hits_zh = retriever.search_chunks(query="乙肝 替诺福韦", language="zh", top_k=2)
    assert len(hits_zh) > 0
    assert hits_zh[0].item_id == "zh_c1"

    # Cross-check: searching for VI term in EN partition returns empty
    hits_cross = retriever.search_chunks(query="Viêm gan", language="en", top_k=2)
    assert hits_cross == []


def test_tokenizers_functionality():
    # Vietnamese syllable
    vi_tok = get_tokenizer("vi", "syllable")
    tokens_vi = vi_tok("Bệnh nhân Đái tháo đường!")
    assert "bệnh" in tokens_vi
    assert "đái" in tokens_vi
    assert "tháo" in tokens_vi
    assert "đường" in tokens_vi

    # Vietnamese pyvi fallback / test
    vi_pyvi = get_tokenizer("vi", "pyvi")
    tokens_pyvi = vi_pyvi("Bệnh nhân tiểu đường")
    assert len(tokens_pyvi) > 0

    # English whitespace
    en_tok = get_tokenizer("en", "whitespace")
    tokens_en = en_tok("Type 2 diabetes, HbA1c test!")
    assert tokens_en == ["type", "2", "diabetes", "hba1c", "test"]

    # English stemmer
    en_stem = get_tokenizer("en", "nltk_stem")
    tokens_stem = en_stem("patients testing treatments")
    assert len(tokens_stem) == 3

    # Chinese char
    zh_char = get_tokenizer("zh", "char")
    tokens_zh = zh_char("糖尿病检验 123")
    assert "糖" in tokens_zh
    assert "尿" in tokens_zh
    assert "123" in tokens_zh

    # Chinese jieba fallback / test
    zh_jieba = get_tokenizer("zh", "jieba")
    tokens_jieba = zh_jieba("糖尿病 检验")
    assert len(tokens_jieba) > 0

    # Generic
    gen_tok = get_tokenizer("fr")
    tokens_gen = gen_tok("Bonjour le monde")
    assert "bonjour" in tokens_gen


def test_tokenizers_edge_cases():
    tok = get_tokenizer("vi")
    assert tok("") == []
    assert tok("   ") == []
    assert tok(None) == []
    assert tok("!!! ??? ,,, ...") == []


def test_bm25_edge_cases(tmp_path: Path):
    docs = [
        DocumentRecord(doc_id="d1", language="vi", text="Sốt xuất huyết Dengue có dấu hiệu cảnh báo."),
        DocumentRecord(doc_id="d2", language="vi", text="Sốt xuất huyết Dengue nặng thể tổn thương gan."),
    ]
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="Sốt Dengue cảnh báo"),
        ChunkRecord(chunk_id="c2", doc_id="d2", language="vi", text="Sốt Dengue nặng gan"),
    ]
    index_dir = tmp_path / "edge_bm25"
    indexer = BM25Indexer(backend="rank_bm25")
    indexer.build_indexes(documents=docs, chunks=chunks, output_dir=index_dir)

    retriever = BM25Retriever(index_dir=index_dir, backend="rank_bm25")

    # Empty query
    assert retriever.search_chunks("", "vi") == []
    assert retriever.search_chunks("   ", "vi") == []
    assert retriever.search_documents("", "vi") == []

    # Non-matching query
    assert retriever.search_chunks("COVID-19 vaccine Pfizer", "vi") == []

    # top_k limit
    hits_top1 = retriever.search_chunks("Sốt Dengue", "vi", top_k=1)
    assert len(hits_top1) == 1
    assert hits_top1[0].rank == 1

    # top_k invalid
    assert retriever.search_chunks("Sốt Dengue", "vi", top_k=0) == []
    assert retriever.search_chunks("Sốt Dengue", "vi", top_k=-1) == []

    # Non-existent language
    assert retriever.search_chunks("test", "nonexistent_lang") == []

    # Non-existent index dir
    missing_retriever = BM25Retriever(index_dir=tmp_path / "does_not_exist")
    assert missing_retriever.search_chunks("test", "vi") == []


def test_bm25_invalid_backend():
    with pytest.raises(ValueError, match="Unsupported BM25 backend"):
        BM25Indexer(backend="invalid_backend")
