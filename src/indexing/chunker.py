from __future__ import annotations

import logging
import re
from typing import Any, Iterator, Protocol, Sequence, runtime_checkable
from src.data.schemas import DocumentRecord, ChunkRecord
from src.data.vibiomir_contracts import (
    ExtractedBlock,
    ExtractedDocument,
    SpanChunk,
    ChunkConfig,
    MigrationResult,
)

logger = logging.getLogger(__name__)



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


def _count_tokens(tokenizer: Any, text: str) -> int:
    """Compute subword or word token count using provided tokenizer."""
    if not text:
        return 0
    if hasattr(tokenizer, "encode"):
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    elif callable(tokenizer):
        try:
            res = tokenizer(text, add_special_tokens=False)
            if isinstance(res, dict) and "input_ids" in res:
                return len(res["input_ids"])
            elif isinstance(res, (list, tuple)):
                return len(res)
        except Exception:
            pass
    return len(text.split())


def _get_token_offsets(tokenizer: Any, text: str) -> list[tuple[int, int]]:
    """Retrieve character offsets for tokens, falling back to regex word boundaries."""
    if not text:
        return []
    if callable(tokenizer):
        try:
            res = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            if isinstance(res, dict) and "offset_mapping" in res and res["offset_mapping"]:
                offsets = [(int(s), int(e)) for s, e in res["offset_mapping"] if e > s]
                if offsets:
                    return offsets
        except Exception:
            pass
    matches = list(re.finditer(r"\S+", text))
    return [(m.start(), m.end()) for m in matches]


def _build_contextual_header(
    title: str | None,
    section: str | None,
    dense_tokenizer: Any,
    reranker_tokenizer: Any,
    max_tokens: int,
) -> str:
    """Build a title/section contextual header prefix for index_text bounded by max_tokens."""
    if max_tokens <= 0:
        return ""
    parts = []
    if title and title.strip():
        parts.append(title.strip())
    if section and section.strip() and (not title or section.strip().lower() != title.strip().lower()):
        parts.append(section.strip())
    if not parts:
        return ""
    header_cand = " - ".join(parts)
    d_cnt = _count_tokens(dense_tokenizer, header_cand)
    r_cnt = _count_tokens(reranker_tokenizer, header_cand)
    if max(d_cnt, r_cnt) <= max_tokens:
        return header_cand
    offsets = _get_token_offsets(dense_tokenizer, header_cand)
    if not offsets:
        return header_cand[: max_tokens * 4]
    for end_idx in range(min(max_tokens, len(offsets)), 0, -1):
        sub_hdr = header_cand[: offsets[end_idx - 1][1]].strip()
        if max(_count_tokens(dense_tokenizer, sub_hdr), _count_tokens(reranker_tokenizer, sub_hdr)) <= max_tokens:
            return sub_hdr
    return header_cand[: offsets[0][1]].strip()


def chunk_source_document(
    document: ExtractedDocument,
    dense_tokenizer: Any,
    reranker_tokenizer: Any,
    config: ChunkConfig | None = None,
) -> Iterator[SpanChunk]:
    """Partition an ExtractedDocument into SpanChunks with exact provenance slices and dual token budgets.
    
    Guarantees:
    - Exact character span slice identity: chunk.text == document.source_text[chunk.start_char:chunk.end_char].
    - chunk.source_sha256 == document.source_sha256 and chunk.doc_id == document.doc_id.
    - Deterministic chunk_id format: f"{doc_id}:{source_sha256}:{chunk_index:06d}".
    - Dual tokenizer body limit: max tokens <= config.body_max_tokens under both tokenizers.
    - Long paragraphs split via tokenizer offsets with configured overlap.
    - Section boundaries strictly preserved.
    - Contextual headers appear ONLY in index_text, NEVER in raw slice text.
    - All content blocks covered without gaps or deletions.
    """
    if config is None:
        config = ChunkConfig()

    body_max = config.body_max_tokens
    long_overlap = config.long_block_overlap_tokens
    header_max = config.contextual_header_max_tokens

    if not document.source_text:
        return

    blocks = list(document.blocks)
    if not blocks:
        blocks = [
            ExtractedBlock(
                block_index=0,
                kind="paragraph",
                start_char=0,
                end_char=len(document.source_text),
                section=None,
            )
        ]

    chunk_idx = 0

    def make_chunk(start_c: int, end_c: int, sec: str | None) -> SpanChunk:
        nonlocal chunk_idx
        chunk_text = document.source_text[start_c:end_c]
        header = _build_contextual_header(
            title=document.title,
            section=sec,
            dense_tokenizer=dense_tokenizer,
            reranker_tokenizer=reranker_tokenizer,
            max_tokens=header_max,
        )
        idx_text = f"{header}\n\n{chunk_text}" if header else chunk_text
        c = SpanChunk(
            chunk_id=f"{document.doc_id}:{document.source_sha256}:{chunk_idx:06d}",
            doc_id=document.doc_id,
            source_version=document.source_version,
            source_sha256=document.source_sha256,
            start_char=start_c,
            end_char=end_c,
            chunk_index=chunk_idx,
            language=document.language,
            section=sec,
            text=chunk_text,
            index_text=idx_text,
            dense_token_count=_count_tokens(dense_tokenizer, chunk_text),
            reranker_token_count=_count_tokens(reranker_tokenizer, chunk_text),
        )
        chunk_idx += 1
        return c

    def split_long_block(blk: ExtractedBlock) -> Iterator[SpanChunk]:
        blk_text = document.source_text[blk.start_char : blk.end_char]
        offsets = _get_token_offsets(dense_tokenizer, blk_text)
        if not offsets:
            yield make_chunk(blk.start_char, blk.end_char, blk.section)
            return

        total_toks = len(offsets)
        token_pos = 0
        w_size = body_max
        step_overlap = min(long_overlap, max(1, w_size // 2))

        while token_pos < total_toks:
            end_pos = min(token_pos + w_size, total_toks)
            while end_pos > token_pos + 1:
                cand_s = offsets[token_pos][0]
                cand_e = offsets[end_pos - 1][1]
                cand_t = blk_text[cand_s:cand_e]
                if (
                    _count_tokens(dense_tokenizer, cand_t) <= body_max
                    and _count_tokens(reranker_tokenizer, cand_t) <= body_max
                ):
                    break
                end_pos -= 1

            sub_s = blk.start_char + offsets[token_pos][0]
            if token_pos == 0:
                sub_s = blk.start_char
            if end_pos >= total_toks:
                sub_e = blk.end_char
            else:
                sub_e = blk.start_char + offsets[end_pos - 1][1]

            yield make_chunk(sub_s, sub_e, blk.section)

            if end_pos >= total_toks:
                break
            step = max(1, (end_pos - token_pos) - step_overlap)
            token_pos += step

    curr_pack: list[ExtractedBlock] = []

    for b in blocks:
        b_text = document.source_text[b.start_char : b.end_char]
        b_dense = _count_tokens(dense_tokenizer, b_text)
        b_rerank = _count_tokens(reranker_tokenizer, b_text)
        is_long = max(b_dense, b_rerank) > body_max

        if is_long:
            if curr_pack:
                yield make_chunk(curr_pack[0].start_char, curr_pack[-1].end_char, curr_pack[0].section)
                curr_pack = []
            yield from split_long_block(b)
        else:
            if not curr_pack:
                curr_pack.append(b)
            else:
                same_sec = b.section == curr_pack[0].section
                if not same_sec:
                    yield make_chunk(curr_pack[0].start_char, curr_pack[-1].end_char, curr_pack[0].section)
                    curr_pack = [b]
                else:
                    combined_s = curr_pack[0].start_char
                    combined_e = b.end_char
                    comb_text = document.source_text[combined_s:combined_e]
                    c_dense = _count_tokens(dense_tokenizer, comb_text)
                    c_rerank = _count_tokens(reranker_tokenizer, comb_text)
                    if max(c_dense, c_rerank) <= body_max:
                        curr_pack.append(b)
                    else:
                        yield make_chunk(curr_pack[0].start_char, curr_pack[-1].end_char, curr_pack[0].section)
                        curr_pack = [b]

    if curr_pack:
        yield make_chunk(curr_pack[0].start_char, curr_pack[-1].end_char, curr_pack[0].section)


def migrate_legacy_chunk(
    legacy_chunk: ChunkRecord | dict[str, Any],
    document: ExtractedDocument,
    dense_tokenizer: Any = None,
    reranker_tokenizer: Any = None,
    config: ChunkConfig | None = None,
    on_ambiguous: str = "quarantine",
) -> SpanChunk | None:
    """Migrate a legacy chunk to SpanChunk with verified character span provenance.
    
    If provenance is already valid in metadata, preserves it.
    If only text is available, resolves only if an exact unique occurrence exists in source_text.
    Otherwise returns None (quarantine).
    """
    text = getattr(legacy_chunk, "text", None)
    if text is None and isinstance(legacy_chunk, dict):
        text = legacy_chunk.get("text", "")
    text = str(text or "")

    meta = getattr(legacy_chunk, "metadata", None)
    if meta is None and isinstance(legacy_chunk, dict):
        meta = legacy_chunk.get("metadata", {})
    meta = meta or {}

    chunk_idx = getattr(legacy_chunk, "chunk_index", None)
    if chunk_idx is None and isinstance(legacy_chunk, dict):
        chunk_idx = legacy_chunk.get("chunk_index", 0)
    chunk_idx = int(chunk_idx or 0)

    # 1. Existing valid provenance
    start_c = meta.get("start_char")
    end_c = meta.get("end_char")
    s_sha = meta.get("source_sha256")
    if (
        start_c is not None
        and end_c is not None
        and s_sha == document.source_sha256
        and 0 <= int(start_c) < int(end_c) <= len(document.source_text)
        and document.source_text[int(start_c) : int(end_c)] == text
    ):
        return SpanChunk(
            chunk_id=f"{document.doc_id}:{document.source_sha256}:{chunk_idx:06d}",
            doc_id=document.doc_id,
            source_version=document.source_version,
            source_sha256=document.source_sha256,
            start_char=int(start_c),
            end_char=int(end_c),
            chunk_index=chunk_idx,
            language=document.language,
            section=meta.get("section"),
            text=text,
            index_text=meta.get("index_text", text),
            dense_token_count=_count_tokens(dense_tokenizer, text),
            reranker_token_count=_count_tokens(reranker_tokenizer, text),
        )

    # 2. Search for exact occurrence in document.source_text
    matches = []
    pos = 0
    while True:
        found = document.source_text.find(text, pos)
        if found == -1:
            break
        matches.append(found)
        pos = found + 1

    if len(matches) == 1:
        s_idx = matches[0]
        e_idx = s_idx + len(text)
        return SpanChunk(
            chunk_id=f"{document.doc_id}:{document.source_sha256}:{chunk_idx:06d}",
            doc_id=document.doc_id,
            source_version=document.source_version,
            source_sha256=document.source_sha256,
            start_char=s_idx,
            end_char=e_idx,
            chunk_index=chunk_idx,
            language=document.language,
            section=None,
            text=text,
            index_text=text,
            dense_token_count=_count_tokens(dense_tokenizer, text),
            reranker_token_count=_count_tokens(reranker_tokenizer, text),
        )

    return None


def migrate_legacy_chunks(
    legacy_chunks: Sequence[ChunkRecord | dict[str, Any]],
    document: ExtractedDocument,
    dense_tokenizer: Any = None,
    reranker_tokenizer: Any = None,
    config: ChunkConfig | None = None,
    on_ambiguous: str = "quarantine",
) -> MigrationResult:
    """Migrate a sequence of legacy chunks, preserving unique matches and quarantining/regenerating ambiguous ones."""
    migrated: list[SpanChunk] = []
    quarantined: list[Any] = []
    has_ambiguous = False

    for c in legacy_chunks:
        res = migrate_legacy_chunk(
            c,
            document,
            dense_tokenizer=dense_tokenizer,
            reranker_tokenizer=reranker_tokenizer,
            config=config,
            on_ambiguous="quarantine",
        )
        if res is not None:
            migrated.append(res)
        else:
            quarantined.append(c)
            has_ambiguous = True

    regenerated: list[SpanChunk] = []
    if has_ambiguous and on_ambiguous == "regenerate":
        regenerated = list(
            chunk_source_document(
                document,
                dense_tokenizer=dense_tokenizer,
                reranker_tokenizer=reranker_tokenizer,
                config=config,
            )
        )

    return MigrationResult(
        migrated_chunks=tuple(migrated),
        quarantined_chunks=tuple(quarantined),
        regenerated_chunks=tuple(regenerated),
    )

