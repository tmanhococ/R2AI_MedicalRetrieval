from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import logging
from typing import Any

from src.data.schemas import SearchHit
from src.query.query_processor import QueryVariants

logger = logging.getLogger(__name__)

# Candidate 6 parallel streams in canonical evaluation order
ALL_STREAMS = (
    "vi_bm25",
    "vi_dense",
    "en_bm25",
    "en_dense",
    "zh_bm25",
    "zh_dense",
)


@dataclass
class StreamResults:
    """Aggregated retrieval candidates partitioned into document and chunk streams."""

    doc_streams: dict[str, list[SearchHit]] = field(default_factory=dict)
    chunk_streams: dict[str, list[SearchHit]] = field(default_factory=dict)


class RetrievalOrchestrator:
    """Orchestrates candidate retrieval across 6 multilingual parallel streams.

    Streams:
    - vi_bm25: Query is vi_condensed (or vi if condensed is absent), language="vi", BM25 retriever
    - vi_dense: Query is vi, language="vi", Dense retriever
    - en_bm25: Query is en, language="en", BM25 retriever
    - en_dense: Query is en, language="en", Dense retriever
    - zh_bm25: Query is zh, language="zh", BM25 retriever
    - zh_dense: Query is zh, language="zh", Dense retriever
    """

    def __init__(
        self,
        bm25_retriever: Any = None,
        dense_retriever: Any = None,
        default_top_k: int = 200,
        parallel: bool = True,
        max_workers: int = 6,
    ) -> None:
        self.bm25_retriever = bm25_retriever
        self.dense_retriever = dense_retriever
        self.default_top_k = default_top_k
        self.parallel = parallel
        self.max_workers = max(1, max_workers)

    def _get_stream_retriever(self, stream_name: str) -> Any | None:
        if stream_name.endswith("_bm25"):
            return self.bm25_retriever
        elif stream_name.endswith("_dense"):
            return self.dense_retriever
        return None

    @staticmethod
    def _get_stream_language(stream_name: str) -> str:
        return stream_name.split("_")[0]

    @staticmethod
    def _get_stream_query(query_variants: QueryVariants, stream_name: str) -> str | None:
        """Resolve query text for a given stream with strict validation.

        If the corresponding language text is None, empty, or whitespace-only,
        the stream is omitted by returning None.
        """
        if stream_name == "vi_bm25":
            if not query_variants.vi or not query_variants.vi.strip():
                return None
            if query_variants.vi_condensed and query_variants.vi_condensed.strip():
                return query_variants.vi_condensed.strip()
            return query_variants.vi.strip()

        elif stream_name == "vi_dense":
            if not query_variants.vi or not query_variants.vi.strip():
                return None
            return query_variants.vi.strip()

        elif stream_name in ("en_bm25", "en_dense"):
            if not query_variants.en or not query_variants.en.strip():
                return None
            return query_variants.en.strip()

        elif stream_name in ("zh_bm25", "zh_dense"):
            if not query_variants.zh or not query_variants.zh.strip():
                return None
            return query_variants.zh.strip()

        return None

    def _invoke_retriever(
        self,
        retriever: Any,
        method_name: str,
        query: str,
        language: str,
        top_k: int,
        stream_name: str,
    ) -> list[SearchHit] | None:
        """Call search method with fallback and fault tolerance."""
        try:
            search_fn = getattr(retriever, method_name, None)
            if search_fn is None:
                # Defensive fallback: if search_documents missing but aggregate_documents exists
                if (
                    method_name == "search_documents"
                    and hasattr(retriever, "search_chunks")
                    and hasattr(retriever, "aggregate_documents")
                ):
                    chunk_hits = retriever.search_chunks(
                        query=query, language=language, top_k=max(top_k * 5, 50)
                    )
                    doc_hits = retriever.aggregate_documents(chunk_hits)[:top_k]
                    return doc_hits
                logger.warning(
                    "Retriever for stream '%s' does not implement '%s'",
                    stream_name,
                    method_name,
                )
                return None

            try:
                hits = search_fn(query=query, language=language, top_k=top_k)
            except TypeError:
                hits = search_fn(query, language, top_k)

            if hits is None:
                hits = []
            return hits
        except Exception as exc:
            logger.warning(
                "Stream '%s' failed during %s: %s",
                stream_name,
                method_name,
                exc,
                exc_info=True,
            )
            return None

    def _execute_retrievals(
        self,
        method_name: str,
        query_variants: QueryVariants,
        top_k: int | None = None,
        enabled_streams: list[str] | None = None,
    ) -> dict[str, list[SearchHit]]:
        """Filter eligible streams and execute retrievals either concurrently or sequentially."""
        if query_variants is None:
            return {}

        effective_top_k = self.default_top_k if top_k is None else top_k

        # Collect active stream candidates
        active_tasks: list[tuple[str, Any, str, str]] = []
        for stream_name in ALL_STREAMS:
            if enabled_streams is not None and stream_name not in enabled_streams:
                continue

            retriever = self._get_stream_retriever(stream_name)
            if retriever is None:
                continue

            query_text = self._get_stream_query(query_variants, stream_name)
            if query_text is None:
                continue

            language = self._get_stream_language(stream_name)
            active_tasks.append((stream_name, retriever, language, query_text))

        if not active_tasks:
            return {}

        results: dict[str, list[SearchHit]] = {}

        if self.parallel and len(active_tasks) > 1 and self.max_workers > 1:
            workers = min(self.max_workers, len(active_tasks))
            with ThreadPoolExecutor(max_workers=workers) as executor:
                # Submit tasks preserving deterministic mapping
                future_list = [
                    (
                        stream_name,
                        executor.submit(
                            self._invoke_retriever,
                            retriever=retr,
                            method_name=method_name,
                            query=query,
                            language=lang,
                            top_k=effective_top_k,
                            stream_name=stream_name,
                        ),
                    )
                    for stream_name, retr, lang, query in active_tasks
                ]
                for stream_name, future in future_list:
                    try:
                        hits = future.result()
                        if hits is not None:
                            results[stream_name] = hits
                    except Exception as exc:
                        logger.warning(
                            "Stream '%s' failed in worker thread: %s",
                            stream_name,
                            exc,
                            exc_info=True,
                        )
        else:
            for stream_name, retr, lang, query in active_tasks:
                try:
                    hits = self._invoke_retriever(
                        retriever=retr,
                        method_name=method_name,
                        query=query,
                        language=lang,
                        top_k=effective_top_k,
                        stream_name=stream_name,
                    )
                    if hits is not None:
                        results[stream_name] = hits
                except Exception as exc:
                    logger.warning(
                        "Stream '%s' failed in sequential execution: %s",
                        stream_name,
                        exc,
                        exc_info=True,
                    )

        return results

    def retrieve_documents(
        self,
        query_variants: QueryVariants,
        top_k: int | None = None,
        enabled_streams: list[str] | None = None,
    ) -> dict[str, list[SearchHit]]:
        """Retrieve candidate documents across all enabled streams."""
        return self._execute_retrievals(
            method_name="search_documents",
            query_variants=query_variants,
            top_k=top_k,
            enabled_streams=enabled_streams,
        )

    def retrieve_chunks(
        self,
        query_variants: QueryVariants,
        top_k: int | None = None,
        enabled_streams: list[str] | None = None,
    ) -> dict[str, list[SearchHit]]:
        """Retrieve candidate chunks across all enabled streams."""
        return self._execute_retrievals(
            method_name="search_chunks",
            query_variants=query_variants,
            top_k=top_k,
            enabled_streams=enabled_streams,
        )

    def retrieve_all_streams(
        self,
        query_variants: QueryVariants,
        top_k: int | None = None,
        enabled_streams: list[str] | None = None,
    ) -> StreamResults:
        """Execute both document and chunk candidate retrieval across all active streams."""
        doc_streams = self.retrieve_documents(
            query_variants=query_variants,
            top_k=top_k,
            enabled_streams=enabled_streams,
        )
        chunk_streams = self.retrieve_chunks(
            query_variants=query_variants,
            top_k=top_k,
            enabled_streams=enabled_streams,
        )
        return StreamResults(
            doc_streams=doc_streams,
            chunk_streams=chunk_streams,
        )
