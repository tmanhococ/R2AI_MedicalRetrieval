from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

from src.data.dataset_adapter import JsonlDatasetAdapter
from src.data.schemas import ChunkRecord, DocumentRecord, QueryRecord, ScoredHit
from src.fusion.weighted_rrf import fuse_stream_results
from src.indexing.chunker import create_chunker
from src.query.query_processor import AdaptiveQueryRouter
from src.query.translator import MultilingualQueryBuilder, PassthroughTranslator, create_translator
from src.reranking.reranker import CrossEncoderReranker, DummyReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.scoring.joint_scoring import JointScorer

logger = logging.getLogger(__name__)


def resolve_index_dirs(
    config: dict[str, Any],
    index_dir: Path | str | None = None,
) -> tuple[Path, Path]:
    """Resolve BM25 and Dense index directories from explicit path or configuration."""
    if index_dir is not None:
        base = Path(index_dir)
    else:
        out_cfg = config.get("run", {}).get("output_dir", "artifacts")
        base = Path(out_cfg) / "indexes"

    bm25_candidate = base / "bm25"
    bm25_dir = bm25_candidate if bm25_candidate.exists() else base

    dense_candidate = base / "dense"
    dense_dir = dense_candidate if dense_candidate.exists() else base

    return bm25_dir, dense_dir


def load_corpus_and_texts(
    dataset_dir: Path | str,
    index_dir: Path | str | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[
    list[DocumentRecord],
    list[ChunkRecord],
    dict[str, str],
    dict[str, str],
    dict[str, str],
]:
    """Load corpus documents and chunks, providing fast lookup tables for text and doc relationships."""
    adapter = JsonlDatasetAdapter(dataset_dir)
    documents = adapter.load_documents()

    chunks: list[ChunkRecord] = []
    if index_dir is not None:
        idx_chunks = Path(index_dir) / "chunks.jsonl"
        if idx_chunks.exists():
            chunk_adapter = JsonlDatasetAdapter(chunks_path=idx_chunks)
            chunks = chunk_adapter.load_chunks()

    if not chunks:
        chunks = adapter.load_chunks()

    if not chunks and config is not None:
        chunker = create_chunker(config.get("indexing", {}).get("chunking", {}))
        chunks = chunker.chunk_documents(documents)

    doc_texts = {d.doc_id: f"{d.title} {d.text}".strip() if d.title else d.text for d in documents}
    chunk_texts = {c.chunk_id: c.text for c in chunks}
    chunk_to_doc = {c.chunk_id: c.doc_id for c in chunks}

    return documents, chunks, doc_texts, chunk_texts, chunk_to_doc


def build_retrievers(
    config: dict[str, Any],
    bm25_dir: Path,
    dense_dir: Path,
    encoder: Any = None,
) -> tuple[BM25Retriever | None, DenseRetriever | None]:
    """Instantiate BM25 and Dense retrievers based on config and manifest locations."""
    retrieval_cfg = config.get("retrieval", {})
    bm25_cfg = retrieval_cfg.get("bm25", {})
    dense_cfg = retrieval_cfg.get("dense", {})

    bm25_retriever: BM25Retriever | None = None
    if bm25_cfg.get("enabled", True) and bm25_dir.exists():
        try:
            bm25_retriever = BM25Retriever(
                index_dir=bm25_dir,
                backend=bm25_cfg.get("backend"),
                tokenizers_config=bm25_cfg.get("tokenizers"),
            )
        except Exception as exc:
            logger.warning("Failed to initialize BM25Retriever: %s", exc)

    dense_retriever: DenseRetriever | None = None
    if dense_cfg.get("enabled", True) and dense_dir.exists():
        try:
            dense_encoder = encoder if encoder is not None else dense_cfg.get("encoder") or dense_cfg.get("model_name", "intfloat/multilingual-e5-large")
            dense_retriever = DenseRetriever(
                encoder=dense_encoder,
                index_dir=dense_dir,
                backend=dense_cfg.get("backend"),
                instruction_prefix=dense_cfg.get("instruction_prefix"),
                device=dense_cfg.get("device", "auto"),
                normalize_embeddings=dense_cfg.get("normalize_embeddings", True),
            )
        except Exception as exc:
            logger.warning("Failed to initialize DenseRetriever: %s", exc)

    return bm25_retriever, dense_retriever


def build_query_components(
    config: dict[str, Any],
    translator: Any = None,
) -> tuple[AdaptiveQueryRouter, MultilingualQueryBuilder]:
    """Instantiate query understanding router and multilingual expansion builder."""
    qu_cfg = config.get("query_understanding", {})
    adaptive_cfg = qu_cfg.get("adaptive_routing", {})

    router = AdaptiveQueryRouter(
        length_threshold=adaptive_cfg.get("length_threshold", 50),
        strip_clinical_boilerplate=adaptive_cfg.get("strip_clinical_boilerplate", True),
        dense_strategy=adaptive_cfg.get("dense_strategy", "head_tail"),
        max_dense_tokens=adaptive_cfg.get("max_dense_tokens", 512),
    )

    trans_cfg = qu_cfg.get("translation", {})
    if translator is None:
        if trans_cfg.get("enabled", True):
            translator_inst = create_translator(config)
        else:
            translator_inst = PassthroughTranslator()
    else:
        translator_inst = translator

    target_languages = trans_cfg.get("target_languages", ["en", "zh"])
    query_builder = MultilingualQueryBuilder(
        translator=translator_inst,
        target_languages=target_languages,
    )

    return router, query_builder


def build_reranker(config: dict[str, Any], reranker: Any = None) -> Any:
    """Instantiate reranker instance based on config or provided override."""
    if reranker is not None:
        return reranker

    reranker_cfg = config.get("reranker", {})
    if not reranker_cfg.get("enabled", True):
        return None

    model_name = reranker_cfg.get("model_name", "BAAI/bge-reranker-v2-m3")
    if model_name == "dummy":
        return DummyReranker()

    return CrossEncoderReranker(
        model_name=model_name,
        batch_size=reranker_cfg.get("batch_size", 32),
        max_length=reranker_cfg.get("max_length", 512),
    )


def run_pipeline_for_queries(
    queries: Sequence[QueryRecord],
    config: dict[str, Any],
    orchestrator: RetrievalOrchestrator,
    router: AdaptiveQueryRouter,
    query_builder: MultilingualQueryBuilder,
    reranker: Any,
    joint_scorer: JointScorer,
    doc_texts: dict[str, str],
    chunk_texts: dict[str, str],
) -> dict[str, tuple[list[ScoredHit], list[ScoredHit]]]:
    """Execute the multi-stage retrieval pipeline up to joint scoring for a list of queries.

    Pipeline stages per query:
    1. Query routing & normalization
    2. Multilingual translation & variant construction
    3. 6 parallel retrieval streams
    4. Weighted RRF candidate fusion
    5. Reranking (chunk reranking + doc aggregation or direct)
    6. Joint scoring

    Returns:
        Mapping from query_id to (joint_docs, joint_chunks)
    """
    retrieval_cfg = config.get("retrieval", {})
    candidate_top_k = retrieval_cfg.get("candidate_top_k", 200)

    fusion_cfg = config.get("fusion", {})
    fusion_weights = fusion_cfg.get("weights")
    rrf_k = fusion_cfg.get("rrf_k", 60)

    reranker_cfg = config.get("reranker", {})
    reranker_enabled = reranker_cfg.get("enabled", True) and reranker is not None
    doc_strategy = reranker_cfg.get("document_strategy", "chunk_aggregation")
    doc_top_k = reranker_cfg.get("document_top_k", 100)
    chunk_top_k = reranker_cfg.get("chunk_top_k", 200)
    max_length = reranker_cfg.get("max_length", 512)

    results: dict[str, tuple[list[ScoredHit], list[ScoredHit]]] = {}

    for query in queries:
        qid = query.query_id
        processed = router.route(query)
        variants = query_builder.build(processed)

        # Multi-stream candidate retrieval
        stream_results = orchestrator.retrieve_all_streams(variants, top_k=candidate_top_k)

        # Weighted RRF fusion
        fused_docs, fused_chunks = fuse_stream_results(
            stream_results=stream_results,
            weights=fusion_weights,
            rrf_k=rrf_k,
            top_k_docs=doc_top_k,
            top_k_chunks=chunk_top_k,
        )

        # Reranking
        if reranker_enabled:
            # Rerank chunks
            scored_chunks = reranker.rerank_chunks(
                query_variants=variants,
                candidates=fused_chunks,
                chunk_texts=chunk_texts,
                top_k=chunk_top_k,
            )

            # Rerank documents
            if doc_strategy == "chunk_aggregation":
                scored_docs = reranker.rerank_documents_via_chunk_aggregation(
                    doc_candidates=fused_docs,
                    scored_chunks=scored_chunks,
                    top_k=doc_top_k,
                )
            else:
                scored_docs = reranker.rerank_documents_direct(
                    query_variants=variants,
                    candidates=fused_docs,
                    doc_texts=doc_texts,
                    max_length=max_length,
                    top_k=doc_top_k,
                )
        else:
            # Fallback without reranking: treat fused hits as scored hits
            scored_docs = [
                ScoredHit(
                    item_id=h.item_id,
                    item_type="document",
                    doc_id=h.doc_id,
                    language=h.language,
                    rerank_score=h.fusion_score,
                    normalized_score=h.fusion_score,
                    raw_fusion_score=h.fusion_score,
                )
                for h in fused_docs
            ]
            scored_chunks = [
                ScoredHit(
                    item_id=h.item_id,
                    item_type="chunk",
                    doc_id=h.doc_id,
                    language=h.language,
                    rerank_score=h.fusion_score,
                    normalized_score=h.fusion_score,
                    raw_fusion_score=h.fusion_score,
                )
                for h in fused_chunks
            ]

        # Bidirectional Joint Scoring
        joint_docs, joint_chunks = joint_scorer.score(scored_docs, scored_chunks)
        results[qid] = (joint_docs, joint_chunks)

    return results
