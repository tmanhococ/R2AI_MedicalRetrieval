from __future__ import annotations

import pytest
from pathlib import Path

from src.query.query_processor import QueryVariants
from src.data.schemas import SearchHit, FusedHit
from src.retrieval.hybrid import HybridCandidateRetriever
from src.query.translator import PrecomputedCacheTranslator

class MockSparseBackend:
    def search_documents(self, query: str, language: str, top_k: int = 500):
        if language == "vi":
            return [SearchHit(item_id="doc1", item_type="document", doc_id="doc1", language="vi", source="sparse", raw_score=0.9, rank=1)]
        elif language == "en":
            return [SearchHit(item_id="doc2", item_type="document", doc_id="doc2", language="en", source="sparse", raw_score=0.8, rank=1)]
        return []

    def search_chunks(self, query: str, language: str, top_k: int = 1000):
        if language == "vi":
            return [SearchHit(item_id="chunk1", item_type="chunk", doc_id="doc1", language="vi", source="sparse", raw_score=0.95, rank=1)]
        return []

class MockDenseBackend:
    def search(self, query: str, top_k: int = 1000):
        return [SearchHit(item_id="chunk10", item_type="chunk", doc_id="doc1", language="vi", source="dense", raw_score=0.85, rank=1)]

def test_hybrid_retriever_execution():
    retriever = HybridCandidateRetriever(
        sparse_backend=MockSparseBackend(),
        dense_backend=MockDenseBackend(),
    )
    variants = QueryVariants(vi="vi query", en="en query", zh=None)
    fused, seeds = retriever.retrieve(variants, top_k=10)
    assert isinstance(fused, list)
    assert isinstance(seeds, dict)
    assert "sparse_doc" in seeds
    assert "sparse_chunk" in seeds
    assert "dense" in seeds
