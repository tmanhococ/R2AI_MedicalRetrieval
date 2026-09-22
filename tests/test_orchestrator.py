import time
from pathlib import Path
import pytest

from src.data.schemas import SearchHit, DocumentRecord, ChunkRecord
from src.query.query_processor import QueryVariants
from src.retrieval import RetrievalOrchestrator, StreamResults
from src.indexing.bm25_index import BM25Indexer
from src.retrieval.bm25_retriever import BM25Retriever
from src.indexing.dense_index import DenseIndexer
from src.retrieval.dense_retriever import DenseRetriever


class MockRetriever:
    """Configurable mock retriever recording calls and simulating returns/failures."""

    def __init__(self, name: str, delay: float = 0.0, fail_on: set[tuple[str, str]] | None = None):
        self.name = name
        self.delay = delay
        self.fail_on = fail_on or set()
        self.recorded_doc_calls: list[dict] = []
        self.recorded_chunk_calls: list[dict] = []

    def search_documents(self, query: str, language: str, top_k: int = 10) -> list[SearchHit]:
        if self.delay > 0:
            time.sleep(self.delay)
        if ("doc", language) in self.fail_on:
            raise RuntimeError(f"Simulated search_documents failure for {language} in {self.name}")
        self.recorded_doc_calls.append({"query": query, "language": language, "top_k": top_k})
        return [
            SearchHit(
                item_id=f"{self.name}_{language}_doc_1",
                item_type="document",
                doc_id=f"d_{language}_1",
                language=language,
                source=self.name,
                raw_score=10.0,
                rank=1,
            )
        ]

    def search_chunks(self, query: str, language: str, top_k: int = 10) -> list[SearchHit]:
        if self.delay > 0:
            time.sleep(self.delay)
        if ("chunk", language) in self.fail_on:
            raise RuntimeError(f"Simulated search_chunks failure for {language} in {self.name}")
        self.recorded_chunk_calls.append({"query": query, "language": language, "top_k": top_k})
        return [
            SearchHit(
                item_id=f"{self.name}_{language}_chunk_1",
                item_type="chunk",
                doc_id=f"d_{language}_1",
                language=language,
                source=self.name,
                raw_score=5.0,
                rank=1,
            )
        ]


def test_orchestrator_imports():
    from src.retrieval import RetrievalOrchestrator as RO1, StreamResults as SR1
    from src.retrieval.orchestrator import RetrievalOrchestrator as RO2, StreamResults as SR2

    assert RO1 is RO2
    assert SR1 is SR2


def test_orchestrator_all_six_streams_success():
    bm25 = MockRetriever("bm25")
    dense = MockRetriever("dense")
    orchestrator = RetrievalOrchestrator(
        bm25_retriever=bm25,
        dense_retriever=dense,
        default_top_k=50,
        parallel=False,
    )

    variants = QueryVariants(
        vi="bệnh nhân đau tức ngực",
        vi_condensed="đau tức ngực",
        en="chest tightness pain",
        zh="胸闷胸痛",
    )

    results = orchestrator.retrieve_all_streams(variants, top_k=25)

    assert isinstance(results, StreamResults)
    expected_streams = {"vi_bm25", "vi_dense", "en_bm25", "en_dense", "zh_bm25", "zh_dense"}
    assert set(results.doc_streams.keys()) == expected_streams
    assert set(results.chunk_streams.keys()) == expected_streams

    # Verify query routing: vi_bm25 uses vi_condensed, vi_dense uses vi
    bm25_doc_queries = {call["language"]: call["query"] for call in bm25.recorded_doc_calls}
    dense_doc_queries = {call["language"]: call["query"] for call in dense.recorded_doc_calls}

    assert bm25_doc_queries["vi"] == "đau tức ngực"
    assert dense_doc_queries["vi"] == "bệnh nhân đau tức ngực"
    assert bm25_doc_queries["en"] == "chest tightness pain"
    assert dense_doc_queries["en"] == "chest tightness pain"
    assert bm25_doc_queries["zh"] == "胸闷胸痛"
    assert dense_doc_queries["zh"] == "胸闷胸痛"

    # Verify top_k was passed properly
    for call in bm25.recorded_doc_calls + dense.recorded_doc_calls:
        assert call["top_k"] == 25


def test_orchestrator_vi_condensed_fallback():
    bm25 = MockRetriever("bm25")
    dense = MockRetriever("dense")
    orchestrator = RetrievalOrchestrator(bm25_retriever=bm25, dense_retriever=dense)

    # vi_condensed is None
    variants = QueryVariants(vi="sốt xuất huyết", vi_condensed=None)
    orchestrator.retrieve_all_streams(variants)

    assert bm25.recorded_doc_calls[0]["query"] == "sốt xuất huyết"

    # vi_condensed is empty or whitespace
    bm25.recorded_doc_calls.clear()
    variants2 = QueryVariants(vi="sốt xuất huyết", vi_condensed="   ")
    orchestrator.retrieve_all_streams(variants2)

    assert bm25.recorded_doc_calls[0]["query"] == "sốt xuất huyết"


def test_orchestrator_missing_language_handling():
    bm25 = MockRetriever("bm25")
    dense = MockRetriever("dense")
    orchestrator = RetrievalOrchestrator(bm25_retriever=bm25, dense_retriever=dense)

    # Query with EN present, ZH None
    variants = QueryVariants(vi="đau đầu", vi_condensed="đau đầu", en="headache", zh=None)
    results = orchestrator.retrieve_all_streams(variants)

    assert "vi_bm25" in results.doc_streams
    assert "vi_dense" in results.doc_streams
    assert "en_bm25" in results.doc_streams
    assert "en_dense" in results.doc_streams
    assert "zh_bm25" not in results.doc_streams
    assert "zh_dense" not in results.doc_streams

    # Query with EN empty and ZH whitespace
    variants_empty = QueryVariants(vi="đau đầu", en="", zh="   ")
    results2 = orchestrator.retrieve_all_streams(variants_empty)

    assert set(results2.doc_streams.keys()) == {"vi_bm25", "vi_dense"}
    assert set(results2.chunk_streams.keys()) == {"vi_bm25", "vi_dense"}

    # Query with VI empty
    variants_no_vi = QueryVariants(vi="", en="headache", zh="头痛")
    results3 = orchestrator.retrieve_all_streams(variants_no_vi)

    assert "vi_bm25" not in results3.doc_streams
    assert "vi_dense" not in results3.doc_streams
    assert "en_bm25" in results3.doc_streams
    assert "zh_bm25" in results3.doc_streams


def test_orchestrator_missing_retriever_handling():
    bm25 = MockRetriever("bm25")
    variants = QueryVariants(vi="vi query", en="en query", zh="zh query")

    # bm25_retriever is None
    orchestrator_no_bm25 = RetrievalOrchestrator(bm25_retriever=None, dense_retriever=MockRetriever("dense"))
    results = orchestrator_no_bm25.retrieve_all_streams(variants)
    assert set(results.doc_streams.keys()) == {"vi_dense", "en_dense", "zh_dense"}

    # dense_retriever is None
    orchestrator_no_dense = RetrievalOrchestrator(bm25_retriever=bm25, dense_retriever=None)
    results2 = orchestrator_no_dense.retrieve_all_streams(variants)
    assert set(results2.doc_streams.keys()) == {"vi_bm25", "en_bm25", "zh_bm25"}

    # Both retrievers are None
    orchestrator_none = RetrievalOrchestrator(bm25_retriever=None, dense_retriever=None)
    results3 = orchestrator_none.retrieve_all_streams(variants)
    assert results3.doc_streams == {}
    assert results3.chunk_streams == {}


def test_orchestrator_enabled_streams_filtering():
    bm25 = MockRetriever("bm25")
    dense = MockRetriever("dense")
    orchestrator = RetrievalOrchestrator(bm25_retriever=bm25, dense_retriever=dense)
    variants = QueryVariants(vi="vi", en="en", zh="zh")

    # Filter to 2 streams
    results = orchestrator.retrieve_all_streams(variants, enabled_streams=["vi_bm25", "en_dense"])
    assert set(results.doc_streams.keys()) == {"vi_bm25", "en_dense"}
    assert set(results.chunk_streams.keys()) == {"vi_bm25", "en_dense"}

    # Empty enabled streams
    results_empty = orchestrator.retrieve_all_streams(variants, enabled_streams=[])
    assert results_empty.doc_streams == {}
    assert results_empty.chunk_streams == {}

    # Unknown stream
    results_unknown = orchestrator.retrieve_all_streams(variants, enabled_streams=["non_existent"])
    assert results_unknown.doc_streams == {}


def test_orchestrator_exception_resilience():
    # BM25 fails for EN doc search
    bm25 = MockRetriever("bm25", fail_on={("doc", "en")})
    dense = MockRetriever("dense")
    orchestrator = RetrievalOrchestrator(bm25_retriever=bm25, dense_retriever=dense)
    variants = QueryVariants(vi="vi", en="en", zh="zh")

    # Does NOT crash
    results = orchestrator.retrieve_all_streams(variants)

    # en_bm25 failed for doc, so omitted from doc_streams
    assert "en_bm25" not in results.doc_streams
    # en_bm25 succeeded for chunk, so present in chunk_streams
    assert "en_bm25" in results.chunk_streams
    # other streams untouched
    assert "vi_bm25" in results.doc_streams
    assert "en_dense" in results.doc_streams


def test_orchestrator_standalone_documents_and_chunks_methods():
    bm25 = MockRetriever("bm25")
    dense = MockRetriever("dense")
    orchestrator = RetrievalOrchestrator(bm25_retriever=bm25, dense_retriever=dense)
    variants = QueryVariants(vi="vi", en="en")

    doc_results = orchestrator.retrieve_documents(variants, top_k=15)
    assert isinstance(doc_results, dict)
    assert set(doc_results.keys()) == {"vi_bm25", "vi_dense", "en_bm25", "en_dense"}
    assert doc_results["vi_bm25"][0].item_type == "document"

    chunk_results = orchestrator.retrieve_chunks(variants, top_k=12)
    assert isinstance(chunk_results, dict)
    assert set(chunk_results.keys()) == {"vi_bm25", "vi_dense", "en_bm25", "en_dense"}
    assert chunk_results["vi_bm25"][0].item_type == "chunk"


def test_orchestrator_concurrency_and_parallelism():
    bm25 = MockRetriever("bm25", delay=0.05)
    dense = MockRetriever("dense", delay=0.05)

    variants = QueryVariants(vi="vi", en="en", zh="zh")

    # Sequential run
    orchestrator_seq = RetrievalOrchestrator(
        bm25_retriever=bm25,
        dense_retriever=dense,
        parallel=False,
    )
    t0 = time.perf_counter()
    res_seq = orchestrator_seq.retrieve_documents(variants)
    seq_duration = time.perf_counter() - t0

    # Parallel run
    orchestrator_par = RetrievalOrchestrator(
        bm25_retriever=bm25,
        dense_retriever=dense,
        parallel=True,
        max_workers=6,
    )
    t0 = time.perf_counter()
    res_par = orchestrator_par.retrieve_documents(variants)
    par_duration = time.perf_counter() - t0

    # Both produced identical streams and counts
    assert set(res_seq.keys()) == set(res_par.keys())
    assert len(res_seq) == 6

    # 6 streams with 0.05s delay:
    # Sequential ~ 6 * 0.05 = 0.30s
    # Parallel with 6 workers ~ 0.05s to 0.12s
    assert par_duration < seq_duration * 0.75


def test_orchestrator_top_k_default_and_custom():
    bm25 = MockRetriever("bm25")
    dense = MockRetriever("dense")
    orchestrator = RetrievalOrchestrator(bm25_retriever=bm25, dense_retriever=dense, default_top_k=123)
    variants = QueryVariants(vi="vi")

    # Default top_k
    orchestrator.retrieve_documents(variants)
    assert bm25.recorded_doc_calls[-1]["top_k"] == 123

    # Custom top_k overrides default
    orchestrator.retrieve_documents(variants, top_k=42)
    assert bm25.recorded_doc_calls[-1]["top_k"] == 42


def test_orchestrator_with_real_retrievers(tmp_path: Path):
    """Integration test with real BM25Retriever and DenseRetriever instances."""
    docs = [
        DocumentRecord(doc_id="d1", language="vi", text="Bệnh viêm phổi cấp tính do vi khuẩn."),
        DocumentRecord(doc_id="d2", language="en", text="Acute bacterial pneumonia infection."),
    ]
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="d1", language="vi", text="viêm phổi cấp tính vi khuẩn"),
        ChunkRecord(chunk_id="c2", doc_id="d2", language="en", text="acute bacterial pneumonia"),
    ]

    # Build real BM25 Index
    bm25_dir = tmp_path / "bm25_idx"
    bm25_indexer = BM25Indexer(backend="rank_bm25")
    bm25_indexer.build_indexes(documents=docs, chunks=chunks, output_dir=bm25_dir)
    bm25_retriever = BM25Retriever(index_dir=bm25_dir, backend="rank_bm25")

    # Build real Dense Index
    class RealTestEncoder:
        def encode(self, texts, **kwargs):
            import numpy as np
            vecs = []
            for t in texts:
                v = np.full(16, float(len(t) % 5 + 1) / 10.0, dtype=np.float32)
                v = v / np.linalg.norm(v)
                vecs.append(v)
            return np.array(vecs, dtype=np.float32)

    dense_dir = tmp_path / "dense_idx"
    encoder = RealTestEncoder()
    dense_indexer = DenseIndexer(encoder=encoder, backend="numpy")
    dense_indexer.build_index(chunks=chunks, output_dir=dense_dir)
    dense_retriever = DenseRetriever(encoder=encoder, index_dir=dense_dir, backend="numpy")

    orchestrator = RetrievalOrchestrator(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        default_top_k=5,
    )

    variants = QueryVariants(vi="viêm phổi", en="pneumonia", zh=None)
    results = orchestrator.retrieve_all_streams(variants)

    assert "vi_bm25" in results.chunk_streams
    assert "vi_dense" in results.chunk_streams
    assert "en_bm25" in results.chunk_streams
    assert "en_dense" in results.chunk_streams
    assert "zh_bm25" not in results.chunk_streams

    assert len(results.chunk_streams["vi_bm25"]) > 0
    assert results.chunk_streams["vi_bm25"][0].item_id == "c1"


def test_orchestrator_query_variants_none():
    orchestrator = RetrievalOrchestrator()
    assert orchestrator.retrieve_documents(None) == {}
    assert orchestrator.retrieve_chunks(None) == {}
    res = orchestrator.retrieve_all_streams(None)
    assert res.doc_streams == {}
    assert res.chunk_streams == {}


def test_orchestrator_retriever_returns_none():
    class NoneRetriever:
        def search_documents(self, query, language, top_k):
            return None
        def search_chunks(self, query, language, top_k):
            return None

    orchestrator = RetrievalOrchestrator(bm25_retriever=NoneRetriever(), parallel=False)
    variants = QueryVariants(vi="vi")
    results = orchestrator.retrieve_all_streams(variants)
    assert results.doc_streams["vi_bm25"] == []
    assert results.chunk_streams["vi_bm25"] == []


def test_orchestrator_positional_args_and_method_missing():
    class PositionalRetriever:
        def search_documents(self, q, l, k):  # Non-standard param names
            return [SearchHit(item_id="d1", item_type="document", doc_id="d1", language=l, source="bm25", raw_score=1.0, rank=1)]
        # search_chunks is intentionally missing

    orchestrator = RetrievalOrchestrator(bm25_retriever=PositionalRetriever(), parallel=False)
    variants = QueryVariants(vi="vi")
    res = orchestrator.retrieve_all_streams(variants)
    assert "vi_bm25" in res.doc_streams
    assert "vi_bm25" not in res.chunk_streams  # search_chunks was missing


def test_orchestrator_aggregate_documents_fallback():
    class DenseWithAggOnly:
        def search_chunks(self, query, language, top_k):
            return [SearchHit(item_id="c1", item_type="chunk", doc_id="d1", language=language, source="dense", raw_score=0.9, rank=1)]
        def aggregate_documents(self, chunk_hits):
            return [SearchHit(item_id="d1", item_type="document", doc_id="d1", language="vi", source="dense", raw_score=0.9, rank=1)]

    orchestrator = RetrievalOrchestrator(dense_retriever=DenseWithAggOnly(), parallel=False)
    variants = QueryVariants(vi="vi")
    res = orchestrator.retrieve_documents(variants)
    assert "vi_dense" in res
    assert res["vi_dense"][0].item_type == "document"

