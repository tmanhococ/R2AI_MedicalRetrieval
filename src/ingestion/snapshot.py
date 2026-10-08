from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Sequence
import time
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.phase2_contracts import (
    SCOPE_FULL_BODY_INDEX,
    SCOPE_FULL_SOURCE_PASS_WITH_GAPS,
    SCOPE_PARTIAL_BODY_INDEX,
)
from src.data.vibiomir_contracts import (
    ChunkConfig,
    ExtractionConfig,
    ExtractedBlock,
    ExtractedDocument,
    SnapshotConfig,
    SnapshotManifest,
    SpanChunk,
)
from src.data.vibiomir_registry import SourceRegistry
from src.indexing.chunker import (
    _build_contextual_header,
    _count_tokens,
    _get_token_offsets,
    chunk_source_document,
)
from src.ingestion.extractors import extract_document
from src.ingestion.quality import (
    FastTextLanguageDetector,
    LanguageDetector,
    classify_quality,
)

logger = logging.getLogger(__name__)

DOCUMENTS_SCHEMA = pa.schema([
    pa.field("doc_id", pa.int64(), nullable=False),
    pa.field("source_version", pa.string(), nullable=False),
    pa.field("source_text", pa.string(), nullable=False),
    pa.field("source_sha256", pa.string(), nullable=False),
    pa.field("raw_sha256", pa.string(), nullable=False),
    pa.field("title", pa.string(), nullable=True),
    pa.field("language", pa.string(), nullable=False),
    pa.field("blocks_json", pa.string(), nullable=False),
    pa.field("quality_flags_json", pa.string(), nullable=False),
])

CHUNKS_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.string(), nullable=False),
    pa.field("doc_id", pa.int64(), nullable=False),
    pa.field("source_version", pa.string(), nullable=False),
    pa.field("source_sha256", pa.string(), nullable=False),
    pa.field("start_char", pa.int64(), nullable=False),
    pa.field("end_char", pa.int64(), nullable=False),
    pa.field("chunk_index", pa.int64(), nullable=False),
    pa.field("language", pa.string(), nullable=False),
    pa.field("section", pa.string(), nullable=True),
    pa.field("text", pa.string(), nullable=False),
    pa.field("index_text", pa.string(), nullable=False),
    pa.field("dense_token_count", pa.int64(), nullable=False),
    pa.field("reranker_token_count", pa.int64(), nullable=False),
])

GAPS_SCHEMA = pa.schema([
    pa.field("doc_id", pa.int64(), nullable=False),
    pa.field("original_url", pa.string(), nullable=False),
    pa.field("reason", pa.string(), nullable=False),
    pa.field("http_status", pa.int32(), nullable=True),
])


class ShardedParquetWriter:
    """Manages sharded Parquet writing with row-group limits, byte rotation, and atomic renaming."""

    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        schema: pa.Schema,
        row_group_size: int = 5000,
        target_shard_bytes: int = 67108864,
        start_shard_index: int = 0,
        completed_shards: list[dict[str, Any]] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.schema = schema
        self.row_group_size = row_group_size
        self.target_shard_bytes = target_shard_bytes

        self.shard_index = start_shard_index
        self.current_rows: list[dict[str, Any]] = []
        self.current_tmp_path: Path | None = None
        self.current_final_name: str | None = None
        self.current_writer: pq.ParquetWriter | None = None
        self.current_shard_rows = 0
        self.completed_shards: list[dict[str, Any]] = list(completed_shards) if completed_shards else []

    def _open_new_shard(self) -> None:
        self.current_final_name = f"{self.prefix}_{self.shard_index:05d}.parquet"
        self.current_tmp_path = self.output_dir / f"{self.current_final_name}.tmp_{uuid.uuid4().hex}"
        self.current_writer = pq.ParquetWriter(
            str(self.current_tmp_path),
            self.schema,
            compression="SNAPPY",
        )
        self.current_shard_rows = 0

    def add_row(self, row_dict: dict[str, Any]) -> tuple[str, int]:
        """Appends a row. Returns (shard_filename, row_index_in_shard)."""
        if self.current_writer is None:
            self._open_new_shard()

        row_idx = self.current_shard_rows
        self.current_rows.append(row_dict)
        self.current_shard_rows += 1

        if len(self.current_rows) >= self.row_group_size:
            self._flush_batch()

        # Check file size for shard rotation
        if self.current_tmp_path and self.current_tmp_path.exists():
            if self.current_tmp_path.stat().st_size >= self.target_shard_bytes:
                self._rotate_shard()

        assert self.current_final_name is not None
        return self.current_final_name, row_idx

    def _flush_batch(self) -> None:
        if not self.current_rows or self.current_writer is None:
            return
        batch = pa.RecordBatch.from_pylist(self.current_rows, schema=self.schema)
        self.current_writer.write_batch(batch)
        self.current_rows.clear()

    def _rotate_shard(self) -> None:
        self._flush_batch()
        if self.current_writer is not None:
            self.current_writer.close()
            self.current_writer = None

        if self.current_tmp_path and self.current_tmp_path.exists():
            final_path = self.output_dir / self.current_final_name
            # Atomically rename
            self.current_tmp_path.replace(final_path)

            file_bytes = final_path.stat().st_size
            file_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
            self.completed_shards.append({
                "path": self.current_final_name,
                "sha256": file_hash,
                "type": self.prefix,
                "rows": self.current_shard_rows,
                "bytes": file_bytes,
            })
            self.shard_index += 1

    def close(self) -> list[dict[str, Any]]:
        self._flush_batch()
        if self.current_writer is not None:
            self.current_writer.close()
            self.current_writer = None

        if self.current_tmp_path and self.current_tmp_path.exists():
            if self.current_shard_rows > 0:
                final_path = self.output_dir / self.current_final_name
                self.current_tmp_path.replace(final_path)

                file_bytes = final_path.stat().st_size
                file_hash = hashlib.sha256(final_path.read_bytes()).hexdigest()
                self.completed_shards.append({
                    "path": self.current_final_name,
                    "sha256": file_hash,
                    "type": self.prefix,
                    "rows": self.current_shard_rows,
                    "bytes": file_bytes,
                })
                self.shard_index += 1
            else:
                try:
                    self.current_tmp_path.unlink()
                except Exception:
                    pass

        return self.completed_shards


class FallbackLanguageDetector:
    """Heuristic detector when fasttext model file is not present."""

    def detect(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return ("und", 0.0)
        # Check Chinese characters
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return ("zh", 0.98)
        # Check Vietnamese diacritics
        vi_chars = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")
        if any(c in vi_chars for c in text.lower()):
            return ("vi", 0.96)
        return ("en", 0.85)


def _rechunk_oversized_span(
    sp: SpanChunk,
    doc: ExtractedDocument,
    dense_tokenizer: Any,
    reranker_tokenizer: Any,
    config: ChunkConfig,
) -> list[SpanChunk]:
    """Subdivides an oversized span deterministically to meet model token budget."""
    body_max = config.body_max_tokens
    long_overlap = config.long_block_overlap_tokens
    header_max = config.contextual_header_max_tokens

    span_text = sp.text
    offsets = _get_token_offsets(dense_tokenizer, span_text)
    if not offsets:
        step = max(50, len(span_text) // 2)
        offsets = [(i, min(i + step, len(span_text))) for i in range(0, len(span_text), step)]

    total_toks = len(offsets)
    token_pos = 0
    sub_chunks: list[SpanChunk] = []

    while token_pos < total_toks:
        end_pos = min(token_pos + body_max, total_toks)
        while end_pos > token_pos + 1:
            cand_s = offsets[token_pos][0]
            cand_e = offsets[end_pos - 1][1]
            cand_t = span_text[cand_s:cand_e]
            d_tok = _count_tokens(dense_tokenizer, cand_t)
            r_tok = _count_tokens(reranker_tokenizer, cand_t)
            if d_tok <= body_max and r_tok <= body_max:
                break
            end_pos -= 1

        sub_s = sp.start_char + offsets[token_pos][0]
        if token_pos == 0:
            sub_s = sp.start_char
        if end_pos >= total_toks:
            sub_e = sp.end_char
        else:
            sub_e = sp.start_char + offsets[end_pos - 1][1]

        sub_text = doc.source_text[sub_s:sub_e]
        header = _build_contextual_header(
            title=doc.title,
            section=sp.section,
            dense_tokenizer=dense_tokenizer,
            reranker_tokenizer=reranker_tokenizer,
            max_tokens=header_max,
        )
        idx_text = f"{header}\n\n{sub_text}" if header else sub_text
        sub_chunks.append(
            SpanChunk(
                chunk_id="",
                doc_id=doc.doc_id,
                source_version=doc.source_version,
                source_sha256=doc.source_sha256,
                start_char=sub_s,
                end_char=sub_e,
                chunk_index=0,
                language=doc.language,
                section=sp.section,
                text=sub_text,
                index_text=idx_text,
                dense_token_count=_count_tokens(dense_tokenizer, sub_text),
                reranker_token_count=_count_tokens(reranker_tokenizer, sub_text),
            )
        )

        if end_pos >= total_toks:
            break
        step_overlap = min(long_overlap, max(1, body_max // 2))
        step = max(1, (end_pos - token_pos) - step_overlap)
        token_pos += step

    return sub_chunks


def _validate_and_rechunk_spans(
    spans: Sequence[SpanChunk],
    doc: ExtractedDocument,
    dense_tokenizer: Any,
    reranker_tokenizer: Any,
    config: ChunkConfig,
) -> list[SpanChunk]:
    """Validates each chunk against token budgets and re-chunks any oversized spans prior to freeze."""
    body_max = config.body_max_tokens
    validated: list[SpanChunk] = []

    for sp in spans:
        d_cnt = _count_tokens(dense_tokenizer, sp.text)
        r_cnt = _count_tokens(reranker_tokenizer, sp.text)
        if d_cnt <= body_max and r_cnt <= body_max:
            validated.append(sp)
        else:
            logger.info(
                f"Re-chunking oversized span in doc {doc.doc_id}: tokens ({d_cnt}, {r_cnt}) > budget {body_max}"
            )
            sub_spans = _rechunk_oversized_span(sp, doc, dense_tokenizer, reranker_tokenizer, config)
            validated.extend(sub_spans)

    final_spans: list[SpanChunk] = []
    for idx, sp in enumerate(validated):
        final_spans.append(
            SpanChunk(
                chunk_id=f"{doc.doc_id}:{doc.source_sha256}:{idx:06d}",
                doc_id=doc.doc_id,
                source_version=sp.source_version,
                source_sha256=sp.source_sha256,
                start_char=sp.start_char,
                end_char=sp.end_char,
                chunk_index=idx,
                language=sp.language,
                section=sp.section,
                text=sp.text,
                index_text=sp.index_text,
                dense_token_count=_count_tokens(dense_tokenizer, sp.text),
                reranker_token_count=_count_tokens(reranker_tokenizer, sp.text),
            )
        )
    return final_spans


def build_snapshot(
    registry: SourceRegistry,
    config: SnapshotConfig,
    raw_root: Path | None = None,
    extraction_config: ExtractionConfig | None = None,
    chunk_config: ChunkConfig | None = None,
    dense_tokenizer: Any = None,
    reranker_tokenizer: Any = None,
    language_detector: LanguageDetector | None = None,
    queries_path: Path | None = None,
    source_limit: int | None = None,
    resume: bool = False,
) -> SnapshotManifest:
    """Builds an immutable frozen snapshot from a SourceRegistry with atomic publication, shard rotation, and resume checkpointing."""
    snapshot_dir = config.snapshot_root / config.name
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    ext_cfg = extraction_config or ExtractionConfig()
    chk_cfg = chunk_config or config.chunk_config or ChunkConfig()

    # Determine language detector
    if language_detector is None:
        if ext_cfg.language_model_path.exists():
            language_detector = FastTextLanguageDetector(ext_cfg.language_model_path)
        else:
            language_detector = FallbackLanguageDetector()

    import sqlite3
    lookup_db_path = snapshot_dir / "lookup.sqlite"
    lookup_conn = sqlite3.connect(str(lookup_db_path))
    with lookup_conn:
        lookup_conn.execute("PRAGMA journal_mode = WAL;")
        lookup_conn.execute("CREATE TABLE IF NOT EXISTS official_docs (doc_id INTEGER PRIMARY KEY);")
        lookup_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_locations (
                doc_id INTEGER PRIMARY KEY,
                shard TEXT NOT NULL,
                row_idx INTEGER NOT NULL
            );
            """
        )
        lookup_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_locations (
                chunk_id TEXT PRIMARY KEY,
                doc_id INTEGER NOT NULL,
                shard TEXT NOT NULL,
                row_idx INTEGER NOT NULL,
                chunk_index INTEGER
            );
            """
        )
        lookup_conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_locations_doc_id ON chunk_locations(doc_id);")
        lookup_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_gaps (
                doc_id INTEGER PRIMARY KEY,
                original_url TEXT NOT NULL,
                reason TEXT NOT NULL,
                http_status INTEGER
            );
            """
        )
        lookup_conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_checkpoint (
                key TEXT PRIMARY KEY,
                val TEXT NOT NULL
            );
            """
        )

    last_processed_doc_id = 0
    doc_completed_shards: list[dict[str, Any]] = []
    chunk_completed_shards: list[dict[str, Any]] = []

    if resume:
        # 1. Clean up orphaned tmp files
        for tmp_f in snapshot_dir.glob("*.tmp_*"):
            try:
                tmp_f.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove stale tmp file {tmp_f}: {e}")

        # 2. Discover finalized shards on disk
        existing_doc_shards = sorted(snapshot_dir.glob("documents_*.parquet"))
        existing_chunk_shards = sorted(snapshot_dir.glob("chunks_*.parquet"))

        valid_doc_shard_names = {f.name for f in existing_doc_shards}
        valid_chunk_shard_names = {f.name for f in existing_chunk_shards}

        for shard_f in existing_doc_shards:
            meta = pq.read_metadata(str(shard_f))
            doc_completed_shards.append({
                "path": shard_f.name,
                "sha256": hashlib.sha256(shard_f.read_bytes()).hexdigest(),
                "type": "documents",
                "rows": meta.num_rows,
                "bytes": shard_f.stat().st_size,
            })

        for shard_f in existing_chunk_shards:
            meta = pq.read_metadata(str(shard_f))
            chunk_completed_shards.append({
                "path": shard_f.name,
                "sha256": hashlib.sha256(shard_f.read_bytes()).hexdigest(),
                "type": "chunks",
                "rows": meta.num_rows,
                "bytes": shard_f.stat().st_size,
            })

        # 3. Clean up lookup.sqlite records pointing to non-existent shards
        if valid_doc_shard_names:
            placeholders = ",".join("?" * len(valid_doc_shard_names))
            with lookup_conn:
                lookup_conn.execute(
                    f"DELETE FROM document_locations WHERE shard NOT IN ({placeholders})",
                    list(valid_doc_shard_names),
                )
        else:
            with lookup_conn:
                lookup_conn.execute("DELETE FROM document_locations")

        if valid_chunk_shard_names:
            placeholders = ",".join("?" * len(valid_chunk_shard_names))
            with lookup_conn:
                lookup_conn.execute(
                    f"DELETE FROM chunk_locations WHERE shard NOT IN ({placeholders})",
                    list(valid_chunk_shard_names),
                )
        else:
            with lookup_conn:
                lookup_conn.execute("DELETE FROM chunk_locations")

        # 4. Find max doc_id safely finalized
        row = lookup_conn.execute(
            """
            SELECT COALESCE(MAX(doc_id), 0) FROM (
                SELECT MAX(doc_id) AS doc_id FROM document_locations
                UNION ALL
                SELECT MAX(doc_id) AS doc_id FROM snapshot_gaps
            )
            """
        ).fetchone()
        last_processed_doc_id = int(row[0]) if row and row[0] is not None else 0

        with lookup_conn:
            lookup_conn.execute("DELETE FROM snapshot_gaps WHERE doc_id > ?", (last_processed_doc_id,))
            lookup_conn.execute("DELETE FROM official_docs WHERE doc_id > ?", (last_processed_doc_id,))

        logger.info(
            f"Resuming snapshot '{config.name}' from doc_id > {last_processed_doc_id} "
            f"({len(doc_completed_shards)} doc shards, {len(chunk_completed_shards)} chunk shards)"
        )

    # Setup sharded writers
    doc_writer = ShardedParquetWriter(
        output_dir=snapshot_dir,
        prefix="documents",
        schema=DOCUMENTS_SCHEMA,
        row_group_size=config.parquet_row_group_size,
        target_shard_bytes=config.target_shard_bytes,
        start_shard_index=len(doc_completed_shards),
        completed_shards=doc_completed_shards,
    )
    chunk_writer = ShardedParquetWriter(
        output_dir=snapshot_dir,
        prefix="chunks",
        schema=CHUNKS_SCHEMA,
        row_group_size=config.parquet_row_group_size,
        target_shard_bytes=config.target_shard_bytes,
        start_shard_index=len(chunk_completed_shards),
        completed_shards=chunk_completed_shards,
    )

    source_count = 0
    ready_document_count = 0
    chunk_count = 0
    membership_hasher = hashlib.sha256()
    language_counts: dict[str, int] = Counter()
    quarantine_counts: dict[str, int] = Counter()

    if resume and last_processed_doc_id > 0:
        row_ready = lookup_conn.execute("SELECT COUNT(*) FROM document_locations").fetchone()
        ready_document_count = int(row_ready[0]) if row_ready else 0

        row_chunks = lookup_conn.execute("SELECT COUNT(*) FROM chunk_locations").fetchone()
        chunk_count = int(row_chunks[0]) if row_chunks else 0

        row_src = lookup_conn.execute("SELECT COUNT(*) FROM official_docs").fetchone()
        source_count = int(row_src[0]) if row_src else 0

        for (d_id,) in lookup_conn.execute("SELECT doc_id FROM document_locations ORDER BY doc_id ASC"):
            membership_hasher.update(f"{d_id}\n".encode("utf-8"))

        row_lang = lookup_conn.execute("SELECT val FROM snapshot_checkpoint WHERE key = 'language_counts'").fetchone()
        if row_lang:
            try:
                language_counts.update(json.loads(row_lang[0]))
            except Exception:
                pass
        row_quar = lookup_conn.execute("SELECT val FROM snapshot_checkpoint WHERE key = 'quarantine_counts'").fetchone()
        if row_quar:
            try:
                quarantine_counts.update(json.loads(row_quar[0]))
            except Exception:
                pass

    # Extraction cache to avoid duplicate work when alias doc_ids point to same raw hash
    extraction_cache: dict[tuple[str, str], ExtractedDocument] = {}

    last_seen_doc_id = last_processed_doc_id
    doc_loc_batch: list[tuple[int, str, int]] = []
    chunk_loc_batch: list[tuple[str, int, str, int, int]] = []
    official_doc_batch: list[tuple[int]] = []
    gap_batch: list[tuple[int, str, str, int]] = []

    try:
        t_start = time.time()
        sql = """
            SELECT s.doc_id, s.original_url, s.fetch_job_id,
                   f.state, f.raw_path, f.raw_sha256, f.mime_type, f.http_status
            FROM sources s
            LEFT JOIN fetch_jobs f ON s.fetch_job_id = f.job_id
            WHERE s.doc_id > ?
            ORDER BY s.doc_id ASC
        """
        params = [last_processed_doc_id]
        if source_limit is not None and source_limit > 0:
            sql += f" LIMIT {source_limit}"
        cursor = registry.conn.execute(sql, params)

        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break

            for doc_id, original_url, job_id, state, raw_path_str, raw_sha256, mime_type, http_status in rows:
                last_seen_doc_id = doc_id
                source_count += 1
                official_doc_batch.append((doc_id,))

                if state != "fetched" or not raw_path_str or not raw_sha256:
                    gap_reason = "unknown_fetch_failure"
                    if state in ("failed", "terminal_failure"):
                        gap_reason = "terminal_fetch_failure"
                    elif state == "blocked":
                        gap_reason = "robots_blocked"
                    elif state == "unsupported":
                        gap_reason = "unsupported_content"
                    elif state in ("pending", "leased", "retry_wait"):
                        gap_reason = f"unresolved_{state}"
                    else:
                        gap_reason = f"fetch_state_{state}"
                    gap_batch.append((doc_id, original_url, gap_reason, http_status if http_status is not None else -1))
                    continue

                raw_path = Path(raw_path_str)
                if not raw_path.exists() and raw_root is not None:
                    candidate = raw_root / raw_path_str
                    if candidate.exists():
                        raw_path = candidate

                if not raw_path.exists():
                    logger.warning(f"Raw file {raw_path_str} not found for doc {doc_id}")
                    gap_batch.append((doc_id, original_url, "missing_raw_file", http_status if http_status is not None else -1))
                    continue

                # Read raw content
                if str(raw_path).endswith(".gz"):
                    with gzip.open(raw_path, "rb") as gf:
                        raw_bytes = gf.read()
                else:
                    raw_bytes = raw_path.read_bytes()

                # Hash verification
                computed_raw_sha = hashlib.sha256(raw_bytes).hexdigest()
                if computed_raw_sha != raw_sha256:
                    raise ValueError(
                        f"Raw SHA256 mismatch for doc {doc_id}: expected {raw_sha256}, got {computed_raw_sha}"
                    )

                # Extract document (checking cache for alias doc_ids)
                cache_key = (raw_sha256, mime_type or "text/html")
                if cache_key in extraction_cache:
                    base_extracted = extraction_cache[cache_key]
                    extracted_doc = ExtractedDocument(
                        doc_id=doc_id,
                        source_version=base_extracted.source_version,
                        source_text=base_extracted.source_text,
                        source_sha256=base_extracted.source_sha256,
                        raw_sha256=base_extracted.raw_sha256,
                        title=base_extracted.title,
                        language=base_extracted.language,
                        blocks=base_extracted.blocks,
                        quality_flags=base_extracted.quality_flags,
                    )
                else:
                    extracted_doc = extract_document(
                        doc_id=doc_id,
                        raw=raw_bytes,
                        mime_type=mime_type or "text/html",
                        raw_sha256=raw_sha256,
                        config=ext_cfg,
                    )
                    if len(extraction_cache) < 256:
                        extraction_cache[cache_key] = extracted_doc

                # Classify quality
                quality = classify_quality(extracted_doc, language_detector, ext_cfg)

                # Check if usable und content should be preserved
                hard_failure_flags = {
                    "captcha",
                    "soft_404",
                    "empty_text",
                    "extraction_failed",
                    "unsupported_format",
                    "unsupported_mime_type",
                    "needs_ocr",
                }
                has_hard_failure = any(f in hard_failure_flags for f in quality.flags)
                is_usable_und = (
                    not has_hard_failure
                    and len(extracted_doc.source_text.strip()) >= 50
                    and (quality.language in ("und", "unknown", "") or "unknown_language" in quality.flags)
                )

                if quality.state == "ready" or is_usable_und:
                    resolved_lang = (
                        quality.language
                        if (quality.state == "ready" and quality.language in ("vi", "en", "zh"))
                        else "und"
                    )

                    ready_doc = ExtractedDocument(
                        doc_id=extracted_doc.doc_id,
                        source_version=extracted_doc.source_version,
                        source_text=extracted_doc.source_text,
                        source_sha256=extracted_doc.source_sha256,
                        raw_sha256=extracted_doc.raw_sha256,
                        title=extracted_doc.title,
                        language=resolved_lang,
                        blocks=extracted_doc.blocks,
                        quality_flags=quality.flags,
                    )

                    # Write ready document to parquet shard
                    blocks_data = [
                        {
                            "block_index": b.block_index,
                            "kind": b.kind,
                            "start_char": b.start_char,
                            "end_char": b.end_char,
                            "section": b.section,
                        }
                        for b in ready_doc.blocks
                    ]
                    doc_row = {
                        "doc_id": ready_doc.doc_id,
                        "source_version": ready_doc.source_version,
                        "source_text": ready_doc.source_text,
                        "source_sha256": ready_doc.source_sha256,
                        "raw_sha256": ready_doc.raw_sha256,
                        "title": ready_doc.title,
                        "language": ready_doc.language,
                        "blocks_json": json.dumps(blocks_data, ensure_ascii=False),
                        "quality_flags_json": json.dumps(list(ready_doc.quality_flags), ensure_ascii=False),
                    }
                    doc_shard_name, doc_row_idx = doc_writer.add_row(doc_row)
                    doc_loc_batch.append((ready_doc.doc_id, doc_shard_name, doc_row_idx))

                    membership_hasher.update(f"{ready_doc.doc_id}\n".encode("utf-8"))
                    ready_document_count += 1
                    language_counts[ready_doc.language] += 1

                    # Generate SpanChunks and validate model token budget prior to freeze
                    spans = list(chunk_source_document(ready_doc, dense_tokenizer, reranker_tokenizer, chk_cfg))
                    validated_spans = _validate_and_rechunk_spans(
                        spans=spans,
                        doc=ready_doc,
                        dense_tokenizer=dense_tokenizer,
                        reranker_tokenizer=reranker_tokenizer,
                        config=chk_cfg,
                    )
                    for sp in validated_spans:
                        chunk_row = {
                            "chunk_id": sp.chunk_id,
                            "doc_id": sp.doc_id,
                            "source_version": sp.source_version,
                            "source_sha256": sp.source_sha256,
                            "start_char": sp.start_char,
                            "end_char": sp.end_char,
                            "chunk_index": sp.chunk_index,
                            "language": sp.language,
                            "section": sp.section,
                            "text": sp.text,
                            "index_text": sp.index_text,
                            "dense_token_count": sp.dense_token_count,
                            "reranker_token_count": sp.reranker_token_count,
                        }
                        chunk_shard_name, chunk_row_idx = chunk_writer.add_row(chunk_row)
                        chunk_loc_batch.append(
                            (sp.chunk_id, sp.doc_id, chunk_shard_name, chunk_row_idx, sp.chunk_index)
                        )
                        chunk_count += 1
                else:
                    for flg in quality.flags:
                        quarantine_counts[flg] += 1
                    gap_batch.append((
                        doc_id,
                        original_url,
                        f"quarantined:{','.join(quality.flags)}" if quality.flags else "quarantined",
                        http_status if http_status is not None else -1,
                    ))

            # Commit batches to lookup db periodically
            if len(official_doc_batch) >= 1000:
                with lookup_conn:
                    lookup_conn.executemany("INSERT OR IGNORE INTO official_docs (doc_id) VALUES (?)", official_doc_batch)
                official_doc_batch.clear()

            if len(doc_loc_batch) >= 500:
                with lookup_conn:
                    lookup_conn.executemany(
                        "INSERT OR REPLACE INTO document_locations (doc_id, shard, row_idx) VALUES (?, ?, ?)",
                        doc_loc_batch,
                    )
                    lookup_conn.executemany(
                        "INSERT OR REPLACE INTO chunk_locations (chunk_id, doc_id, shard, row_idx, chunk_index) VALUES (?, ?, ?, ?, ?)",
                        chunk_loc_batch,
                    )
                doc_loc_batch.clear()
                chunk_loc_batch.clear()

            if len(gap_batch) >= 500:
                with lookup_conn:
                    lookup_conn.executemany(
                        "INSERT OR REPLACE INTO snapshot_gaps (doc_id, original_url, reason, http_status) VALUES (?, ?, ?, ?)",
                        gap_batch,
                    )
                gap_batch.clear()

            # Periodic progress reporting
            if source_count % 10000 < len(rows) or source_count == len(rows):
                elapsed = max(time.time() - t_start, 0.1)
                rate = source_count / elapsed
                pct = source_count / 4394718 * 100
                rem_s = (4394718 - source_count) / max(rate, 0.1)
                with lookup_conn:
                    lookup_conn.execute(
                        "INSERT OR REPLACE INTO snapshot_checkpoint (key, val) VALUES ('language_counts', ?)",
                        (json.dumps(dict(language_counts)),),
                    )
                    lookup_conn.execute(
                        "INSERT OR REPLACE INTO snapshot_checkpoint (key, val) VALUES ('quarantine_counts', ?)",
                        (json.dumps(dict(quarantine_counts)),),
                    )
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Snapshot: {source_count:,}/4,394,718 ({pct:.1f}%) | "
                    f"ready={ready_document_count:,} chunks={chunk_count:,} | "
                    f"{rate:.1f} src/s | ETA={rem_s/3600:.1f}h",
                    flush=True,
                )

        # Final commit to lookup db
        if official_doc_batch:
            with lookup_conn:
                lookup_conn.executemany("INSERT OR IGNORE INTO official_docs (doc_id) VALUES (?)", official_doc_batch)
            official_doc_batch.clear()

        if doc_loc_batch or chunk_loc_batch:
            with lookup_conn:
                lookup_conn.executemany(
                    "INSERT OR REPLACE INTO document_locations (doc_id, shard, row_idx) VALUES (?, ?, ?)",
                    doc_loc_batch,
                )
                lookup_conn.executemany(
                    "INSERT OR REPLACE INTO chunk_locations (chunk_id, doc_id, shard, row_idx, chunk_index) VALUES (?, ?, ?, ?, ?)",
                    chunk_loc_batch,
                )
            doc_loc_batch.clear()
            chunk_loc_batch.clear()

        if gap_batch:
            with lookup_conn:
                lookup_conn.executemany(
                    "INSERT OR REPLACE INTO snapshot_gaps (doc_id, original_url, reason, http_status) VALUES (?, ?, ?, ?)",
                    gap_batch,
                )
            gap_batch.clear()

        with lookup_conn:
            lookup_conn.execute(
                "INSERT OR REPLACE INTO snapshot_checkpoint (key, val) VALUES ('language_counts', ?)",
                (json.dumps(dict(language_counts)),),
            )
            lookup_conn.execute(
                "INSERT OR REPLACE INTO snapshot_checkpoint (key, val) VALUES ('quarantine_counts', ?)",
                (json.dumps(dict(quarantine_counts)),),
            )

        # Close writers and finalize all shards
        completed_doc_shards = doc_writer.close()
        completed_chunk_shards = chunk_writer.close()
        all_shards = completed_doc_shards + completed_chunk_shards

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received! Finalizing shards before exit...")
        if official_doc_batch:
            with lookup_conn:
                lookup_conn.executemany("INSERT OR IGNORE INTO official_docs (doc_id) VALUES (?)", official_doc_batch)
        if doc_loc_batch or chunk_loc_batch:
            with lookup_conn:
                lookup_conn.executemany("INSERT OR REPLACE INTO document_locations (doc_id, shard, row_idx) VALUES (?, ?, ?)", doc_loc_batch)
                lookup_conn.executemany("INSERT OR REPLACE INTO chunk_locations (chunk_id, doc_id, shard, row_idx, chunk_index) VALUES (?, ?, ?, ?, ?)", chunk_loc_batch)
        if gap_batch:
            with lookup_conn:
                lookup_conn.executemany("INSERT OR REPLACE INTO snapshot_gaps (doc_id, original_url, reason, http_status) VALUES (?, ?, ?, ?)", gap_batch)
        doc_writer.close()
        chunk_writer.close()
        print(f"\n[PAUSED] Snapshot gracefully paused at doc_id={last_seen_doc_id}. Shards finalized. Resume anytime with --resume.")
        raise
    finally:
        # Load all gaps from SQLite table before closing lookup_conn
        gap_rows = lookup_conn.execute("SELECT doc_id, original_url, reason, http_status FROM snapshot_gaps ORDER BY doc_id ASC").fetchall()
        gap_records = [
            {"doc_id": r[0], "original_url": r[1], "reason": r[2], "http_status": r[3] if r[3] is not None else -1}
            for r in gap_rows
        ]
        lookup_conn.close()

    # Calculate gap statistics and source scope
    unresolved_count = sum(1 for g in gap_records if g["reason"].startswith("unresolved_"))
    terminal_gap_count = len(gap_records) - unresolved_count
    if unresolved_count > 0:
        source_scope = SCOPE_PARTIAL_BODY_INDEX
    elif len(gap_records) > 0:
        source_scope = SCOPE_FULL_SOURCE_PASS_WITH_GAPS
    else:
        source_scope = SCOPE_FULL_BODY_INDEX

    # Write source_gaps.parquet if gaps exist
    if gap_records:
        gaps_table = pa.Table.from_pylist(gap_records, schema=GAPS_SCHEMA)
        gaps_parquet_name = "source_gaps.parquet"
        gaps_parquet_path = snapshot_dir / gaps_parquet_name
        pq.write_table(gaps_table, str(gaps_parquet_path), compression="SNAPPY")
        gaps_sha = hashlib.sha256(gaps_parquet_path.read_bytes()).hexdigest()
        all_shards.append({
            "path": gaps_parquet_name,
            "sha256": gaps_sha,
            "type": "gaps",
            "rows": len(gap_records),
            "bytes": gaps_parquet_path.stat().st_size,
        })

    # Write gaps.json
    gaps_summary = {
        "snapshot_id": config.name,
        "total_gaps": len(gap_records),
        "unresolved_count": unresolved_count,
        "terminal_gap_count": terminal_gap_count,
        "source_scope": source_scope,
        "gap_counts_by_reason": dict(Counter(g["reason"] for g in gap_records)),
    }
    with open(snapshot_dir / "gaps.json", "w", encoding="utf-8") as f:
        json.dump(gaps_summary, f, indent=2, ensure_ascii=False)

    # Calculate fingerprints and metadata
    created_at = datetime.now(timezone.utc).isoformat()
    ingest_state = registry.get_ingest_state()
    registry_sha256 = ingest_state[0] if ingest_state else hashlib.sha256(b"").hexdigest()

    queries_sha256 = hashlib.sha256(b"").hexdigest()
    if queries_path and Path(queries_path).exists():
        queries_sha256 = hashlib.sha256(Path(queries_path).read_bytes()).hexdigest()

    source_membership_sha256 = membership_hasher.hexdigest()

    extractor_fingerprint = hashlib.sha256(
        f"{ext_cfg.html_backend}:{ext_cfg.pdf_backend}:{ext_cfg.language_confidence_min}".encode("utf-8")
    ).hexdigest()

    chunker_fingerprint = hashlib.sha256(
        f"{chk_cfg.strategy_type}:{chk_cfg.body_max_tokens}:{chk_cfg.long_block_overlap_tokens}:{chk_cfg.contextual_header_max_tokens}".encode(
            "utf-8"
        )
    ).hexdigest()

    manifest = SnapshotManifest(
        snapshot_id=config.name,
        created_at=created_at,
        registry_sha256=registry_sha256,
        queries_sha256=queries_sha256,
        source_membership_sha256=source_membership_sha256,
        extractor_fingerprint=extractor_fingerprint,
        chunker_fingerprint=chunker_fingerprint,
        source_count=source_count,
        ready_document_count=ready_document_count,
        chunk_count=chunk_count,
        language_counts=dict(language_counts),
        quarantine_counts=dict(quarantine_counts),
        shard_paths_and_hashes=all_shards,
        source_scope=source_scope,
        gap_count=len(gap_records),
    )

    # Publish manifest atomically last
    manifest_tmp = snapshot_dir / f"manifest.json.tmp_{uuid.uuid4().hex}"
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2, ensure_ascii=False)
    manifest_final = snapshot_dir / "manifest.json"
    manifest_tmp.replace(manifest_final)

    logger.info(
        f"Snapshot '{config.name}' published successfully: {ready_document_count} docs, "
        f"{chunk_count} chunks across {len(all_shards)} shards (scope: {source_scope}, gaps: {len(gap_records)})."
    )

    return manifest
