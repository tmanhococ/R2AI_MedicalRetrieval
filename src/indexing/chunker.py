from typing import Protocol, runtime_checkable
from src.data.schemas import DocumentRecord, ChunkRecord


@runtime_checkable
class Chunker(Protocol):
    """Protocol defining the chunking interface."""

    def chunk_documents(
        self,
        documents: list[DocumentRecord],
        existing_chunks: list[ChunkRecord] | None = None,
    ) -> list[ChunkRecord]:
        """Chunk a list of documents into ChunkRecords."""
        ...


class AutoChunker:
    """Sliding-window chunker dividing documents into chunks."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_length: int = 10,
        strategy: str = "sliding_window",
    ):
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap must be strictly less than chunk_size: overlap={chunk_overlap}, size={chunk_size}"
            )
        if min_chunk_length < 0:
            raise ValueError(f"min_chunk_length must be non-negative, got {min_chunk_length}")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.strategy = strategy

    def chunk_documents(
        self,
        documents: list[DocumentRecord],
        existing_chunks: list[ChunkRecord] | None = None,
    ) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        step = self.chunk_size - self.chunk_overlap
        effective_min = min(self.min_chunk_length, self.chunk_size)

        for doc in documents:
            if doc.language == "zh" and " " not in doc.text:
                tokens = [char for char in doc.text if not char.isspace()]
                separator = ""
            else:
                tokens = doc.text.split()
                separator = " "

            if not tokens:
                continue

            idx = 0
            for start in range(0, len(tokens), step):
                chunk_tokens = tokens[start : start + self.chunk_size]
                if len(chunk_tokens) < effective_min:
                    continue

                chunk_text = separator.join(chunk_tokens)
                meta = dict(doc.metadata) if doc.metadata else {}
                meta["is_auto_generated"] = True

                chunks.append(
                    ChunkRecord(
                        chunk_id=f"{doc.doc_id}_c{idx}",
                        doc_id=doc.doc_id,
                        language=doc.language,
                        text=chunk_text,
                        chunk_index=idx,
                        metadata=meta,
                    )
                )
                idx += 1

        return chunks


class PrechunkedChunker:
    """Chunker that preserves pre-existing chunk records."""

    def __init__(
        self,
        default_chunks: list[ChunkRecord] | None = None,
        strict: bool = False,
    ):
        self.default_chunks = default_chunks
        self.strict = strict

    def chunk_documents(
        self,
        documents: list[DocumentRecord],
        existing_chunks: list[ChunkRecord] | None = None,
    ) -> list[ChunkRecord]:
        chunks_pool = existing_chunks if existing_chunks is not None else self.default_chunks
        if chunks_pool is None:
            if self.strict:
                raise ValueError("No existing chunks provided for prechunked mode")
            return []

        doc_ids = {doc.doc_id for doc in documents}
        return [c for c in chunks_pool if c.doc_id in doc_ids]


# Alias for backward compatibility / spec alternative naming
PrechunkedAdapter = PrechunkedChunker


class HybridChunker:
    """Chunker that preserves existing chunks for documents that have them,
    and falls back to AutoChunker for documents that do not.
    """

    def __init__(self, auto_chunker: AutoChunker | None = None):
        self.auto_chunker = auto_chunker or AutoChunker()

    def chunk_documents(
        self,
        documents: list[DocumentRecord],
        existing_chunks: list[ChunkRecord] | None = None,
    ) -> list[ChunkRecord]:
        existing_by_doc: dict[str, list[ChunkRecord]] = {}
        if existing_chunks:
            for c in existing_chunks:
                existing_by_doc.setdefault(c.doc_id, []).append(c)

        result: list[ChunkRecord] = []
        for doc in documents:
            if doc.doc_id in existing_by_doc and existing_by_doc[doc.doc_id]:
                result.extend(existing_by_doc[doc.doc_id])
            else:
                result.extend(self.auto_chunker.chunk_documents([doc]))

        return result


def create_chunker(config: dict) -> Chunker:
    """Factory function to create a Chunker from configuration dictionary."""
    if "indexing" in config and isinstance(config["indexing"], dict):
        chunk_cfg = config["indexing"].get("chunking", {})
    elif "chunking" in config and isinstance(config["chunking"], dict):
        chunk_cfg = config["chunking"]
    else:
        chunk_cfg = config

    mode = chunk_cfg.get("mode", "hybrid")
    auto_strategy = chunk_cfg.get("auto_strategy", {})

    chunk_size = auto_strategy.get("chunk_size", 512)
    chunk_overlap = auto_strategy.get("chunk_overlap", 64)
    min_chunk_length = auto_strategy.get("min_chunk_length", 10)
    strategy_type = auto_strategy.get("type", "sliding_window")

    if mode == "auto":
        return AutoChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_length=min_chunk_length,
            strategy=strategy_type,
        )
    elif mode == "hybrid":
        auto_chunker = AutoChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_length=min_chunk_length,
            strategy=strategy_type,
        )
        return HybridChunker(auto_chunker=auto_chunker)
    elif mode == "prechunked":
        strict = chunk_cfg.get("strict", False)
        return PrechunkedChunker(strict=strict)
    else:
        raise ValueError(
            f"Unknown chunking mode: '{mode}'. Supported modes: 'auto', 'hybrid', 'prechunked'"
        )
