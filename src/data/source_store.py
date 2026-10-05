from __future__ import annotations

from collections import OrderedDict, defaultdict
import json
import logging
from pathlib import Path
import sqlite3
import threading
from typing import Any, Callable, Generic, Iterator, Sequence, TypeVar

import pyarrow.parquet as pq

from src.data.phase2_contracts import EvidenceSpan
from src.data.vibiomir_contracts import (
    ExtractedBlock,
    ExtractedDocument,
    SourceStore,
    SpanChunk,
)

logger = logging.getLogger(__name__)

K = TypeVar("K")
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    """Thread-safe and memory-bounded LRU cache supporting both entry count and byte capacity."""

    def __init__(
        self,
        max_entries: int = 2048,
        max_bytes: int = 256 * 1024 * 1024,
        size_fn: Callable[[V], int] | None = None,
    ) -> None:
        self.max_entries = max(1, max_entries)
        self.max_bytes = max(1, max_bytes)
        self.size_fn = size_fn or (lambda _: 1)
        self._cache: OrderedDict[K, tuple[V, int]] = OrderedDict()
        self._current_bytes = 0
        self._lock = threading.RLock()

    def get(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            if key not in self._cache:
                return default
            self._cache.move_to_end(key)
            val, _ = self._cache[key]
            return val

    def __contains__(self, key: K) -> bool:
        with self._lock:
            return key in self._cache

    def __getitem__(self, key: K) -> V:
        with self._lock:
            if key not in self._cache:
                raise KeyError(key)
            self._cache.move_to_end(key)
            val, _ = self._cache[key]
            return val

    def __setitem__(self, key: K, value: V) -> None:
        with self._lock:
            size = self.size_fn(value)
            if key in self._cache:
                _, old_size = self._cache[key]
                self._current_bytes -= old_size
                del self._cache[key]

            self._cache[key] = (value, size)
            self._current_bytes += size
            self._cache.move_to_end(key)
            self._evict()

    def pop(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            if key not in self._cache:
                return default
            val, size = self._cache.pop(key)
            self._current_bytes -= size
            return val

    def _evict(self) -> None:
        # Called within self._lock
        while self._cache and (
            len(self._cache) > self.max_entries or self._current_bytes > self.max_bytes
        ):
            _, (_, size) = self._cache.popitem(last=False)
            self._current_bytes -= size

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)

    @property
    def current_bytes(self) -> int:
        with self._lock:
            return self._current_bytes


def _doc_size(doc: ExtractedDocument) -> int:
    return len(doc.source_text) * 2 + len(doc.blocks) * 64 + 512


def _chunk_size(chunk: SpanChunk) -> int:
    return len(chunk.text) * 2 + len(chunk.index_text) * 2 + 256


def _row_to_document(row_dict: dict[str, Any]) -> ExtractedDocument:
    blocks_data = json.loads(row_dict["blocks_json"])
    blocks = tuple(
        ExtractedBlock(
            block_index=b["block_index"],
            kind=b["kind"],
            start_char=b["start_char"],
            end_char=b["end_char"],
            section=b.get("section"),
        )
        for b in blocks_data
    )
    quality_flags = tuple(json.loads(row_dict["quality_flags_json"]))

    return ExtractedDocument(
        doc_id=int(row_dict["doc_id"]),
        source_version=str(row_dict["source_version"]),
        source_text=str(row_dict["source_text"]),
        source_sha256=str(row_dict["source_sha256"]),
        raw_sha256=str(row_dict["raw_sha256"]),
        title=str(row_dict["title"]) if row_dict.get("title") is not None else None,
        language=str(row_dict["language"]),
        blocks=blocks,
        quality_flags=quality_flags,
    )


def _row_to_chunk(row_dict: dict[str, Any]) -> SpanChunk:
    return SpanChunk(
        chunk_id=str(row_dict["chunk_id"]),
        doc_id=int(row_dict["doc_id"]),
        source_version=str(row_dict["source_version"]),
        source_sha256=str(row_dict["source_sha256"]),
        start_char=int(row_dict["start_char"]),
        end_char=int(row_dict["end_char"]),
        chunk_index=int(row_dict["chunk_index"]),
        language=str(row_dict["language"]),
        section=str(row_dict["section"]) if row_dict.get("section") is not None else None,
        text=str(row_dict["text"]),
        index_text=str(row_dict["index_text"]),
        dense_token_count=int(row_dict["dense_token_count"]),
        reranker_token_count=int(row_dict["reranker_token_count"]),
    )


class SnapshotSourceStore:
    """Reader implementing SourceStore protocol over an immutable frozen snapshot."""

    def __init__(
        self,
        snapshot_path: Path | str,
        cache_size: int = 2048,
        max_cache_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        self.snapshot_path = Path(snapshot_path)
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {self.snapshot_path}")

        manifest_file = self.snapshot_path / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Snapshot manifest not found at {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            self.manifest_data = json.load(f)

        self.lookup_db_path = self.snapshot_path / "lookup.sqlite"
        if not self.lookup_db_path.exists():
            self._build_missing_lookup_index()

        # Open SQLite lookup in read-only mode with URI
        uri = f"file:{self.lookup_db_path.resolve().as_posix()}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)

        self._doc_cache: LRUCache[int, ExtractedDocument] = LRUCache(
            max_entries=cache_size,
            max_bytes=max_cache_bytes,
            size_fn=_doc_size,
        )
        self._chunk_cache: LRUCache[str, SpanChunk] = LRUCache(
            max_entries=cache_size * 4,
            max_bytes=max_cache_bytes,
            size_fn=_chunk_size,
        )
        self._cache_size = cache_size

    def _build_missing_lookup_index(self) -> None:
        """Fallback to index Parquet shards if lookup.sqlite is missing."""
        logger.info(f"Building index for snapshot at {self.snapshot_path}")
        conn = sqlite3.connect(str(self.lookup_db_path))
        with conn:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("CREATE TABLE IF NOT EXISTS official_docs (doc_id INTEGER PRIMARY KEY);")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS document_locations (
                    doc_id INTEGER PRIMARY KEY,
                    shard TEXT NOT NULL,
                    row_idx INTEGER NOT NULL
                );
                """
            )
            conn.execute(
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
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_locations_doc_id ON chunk_locations(doc_id);")

            # Index document shards
            for doc_file in sorted(self.snapshot_path.glob("documents_*.parquet")):
                table = pq.read_table(doc_file, columns=["doc_id"])
                doc_ids = table.column("doc_id").to_pylist()
                rows = [(d, doc_file.name, idx) for idx, d in enumerate(doc_ids)]
                conn.executemany(
                    "INSERT OR REPLACE INTO document_locations (doc_id, shard, row_idx) VALUES (?, ?, ?)",
                    rows,
                )
                conn.executemany(
                    "INSERT OR IGNORE INTO official_docs (doc_id) VALUES (?)",
                    [(d,) for d in doc_ids],
                )

            # Index chunk shards
            for chunk_file in sorted(self.snapshot_path.glob("chunks_*.parquet")):
                table = pq.read_table(chunk_file, columns=["chunk_id", "doc_id", "chunk_index"])
                chunk_ids = table.column("chunk_id").to_pylist()
                c_doc_ids = table.column("doc_id").to_pylist()
                chunk_indices = table.column("chunk_index").to_pylist()
                rows = [
                    (cid, did, chunk_file.name, idx, cidx)
                    for idx, (cid, did, cidx) in enumerate(zip(chunk_ids, c_doc_ids, chunk_indices))
                ]
                conn.executemany(
                    "INSERT OR REPLACE INTO chunk_locations (chunk_id, doc_id, shard, row_idx, chunk_index) VALUES (?, ?, ?, ?, ?)",
                    rows,
                )
        conn.close()

    def contains_official_doc(self, doc_id: int) -> bool:
        """Returns True if doc_id was part of the official registry when snapshot was taken."""
        row = self._conn.execute("SELECT 1 FROM official_docs WHERE doc_id = ?", (doc_id,)).fetchone()
        return row is not None

    def get_document(self, doc_id: int) -> ExtractedDocument:
        """Retrieves exact ExtractedDocument for a doc_id present in this snapshot."""
        if doc_id in self._doc_cache:
            return self._doc_cache[doc_id]
        docs = self.get_documents([doc_id])
        return docs[0]

    def get_documents(self, doc_ids: Sequence[int]) -> list[ExtractedDocument]:
        """Batch-retrieves ExtractedDocuments grouped by shard, preserving input order and caching results."""
        if not doc_ids:
            return []

        result_map: dict[int, ExtractedDocument] = {}
        missing_doc_ids: list[int] = []

        for doc_id in set(doc_ids):
            if doc_id in self._doc_cache:
                result_map[doc_id] = self._doc_cache[doc_id]
            else:
                missing_doc_ids.append(doc_id)

        if missing_doc_ids:
            # Batch query document locations from SQLite in chunks of 500
            shard_to_ids: dict[str, list[int]] = defaultdict(list)
            found_ids: set[int] = set()

            for i in range(0, len(missing_doc_ids), 500):
                batch = missing_doc_ids[i : i + 500]
                placeholders = ",".join("?" for _ in batch)
                cursor = self._conn.execute(
                    f"SELECT doc_id, shard FROM document_locations WHERE doc_id IN ({placeholders})",
                    batch,
                )
                for d_id, shard in cursor.fetchall():
                    shard_to_ids[shard].append(d_id)
                    found_ids.add(d_id)

            unresolved = set(missing_doc_ids) - found_ids
            if unresolved:
                missing_example = next(iter(unresolved))
                raise KeyError(f"Document {missing_example} not found in snapshot {self.snapshot_path.name}")

            # Read each shard exactly once for the batch
            for shard_name, s_doc_ids in shard_to_ids.items():
                shard_path = self.snapshot_path / shard_name
                table = pq.read_table(shard_path, filters=[("doc_id", "in", s_doc_ids)])
                rows = table.to_pylist()
                found_in_shard = set()
                for row_dict in rows:
                    doc = _row_to_document(row_dict)
                    self._doc_cache[doc.doc_id] = doc
                    result_map[doc.doc_id] = doc
                    found_in_shard.add(doc.doc_id)

                missing_in_shard = set(s_doc_ids) - found_in_shard
                if missing_in_shard:
                    m_id = next(iter(missing_in_shard))
                    raise KeyError(f"Document {m_id} not found in shard {shard_name}")

        return [result_map[doc_id] for doc_id in doc_ids]

    def get_chunk(self, chunk_id: str) -> SpanChunk:
        """Retrieves exact SpanChunk for a chunk_id present in this snapshot."""
        if chunk_id in self._chunk_cache:
            return self._chunk_cache[chunk_id]
        chunks = self.get_chunks([chunk_id])
        return chunks[0]

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[SpanChunk]:
        """Batch-retrieves SpanChunks grouped by shard, preserving input order and caching in bounded LRU."""
        if not chunk_ids:
            return []

        result_map: dict[str, SpanChunk] = {}
        missing_chunk_ids: list[str] = []

        for cid in set(chunk_ids):
            if cid in self._chunk_cache:
                result_map[cid] = self._chunk_cache[cid]
            else:
                missing_chunk_ids.append(cid)

        if missing_chunk_ids:
            # Batch query chunk locations from SQLite in chunks of 500
            shard_to_cids: dict[str, list[str]] = defaultdict(list)
            found_cids: set[str] = set()

            for i in range(0, len(missing_chunk_ids), 500):
                batch = missing_chunk_ids[i : i + 500]
                placeholders = ",".join("?" for _ in batch)
                cursor = self._conn.execute(
                    f"SELECT chunk_id, shard FROM chunk_locations WHERE chunk_id IN ({placeholders})",
                    batch,
                )
                for cid, shard in cursor.fetchall():
                    shard_to_cids[shard].append(cid)
                    found_cids.add(cid)

            unresolved = set(missing_chunk_ids) - found_cids
            if unresolved:
                missing_example = next(iter(unresolved))
                raise KeyError(f"Chunk {missing_example} not found in snapshot {self.snapshot_path.name}")

            # Read each shard exactly once for the batch
            for shard_name, s_chunk_ids in shard_to_cids.items():
                shard_path = self.snapshot_path / shard_name
                table = pq.read_table(shard_path, filters=[("chunk_id", "in", s_chunk_ids)])
                rows = table.to_pylist()
                found_in_shard = set()
                for row_dict in rows:
                    chunk = _row_to_chunk(row_dict)
                    self._chunk_cache[chunk.chunk_id] = chunk
                    result_map[chunk.chunk_id] = chunk
                    found_in_shard.add(chunk.chunk_id)

                missing_in_shard = set(s_chunk_ids) - found_in_shard
                if missing_in_shard:
                    m_cid = next(iter(missing_in_shard))
                    raise KeyError(f"Chunk {m_cid} not found in shard {shard_name}")

        return [result_map[cid] for cid in chunk_ids]

    def iter_chunks_for_doc(self, doc_id: int) -> Iterator[SpanChunk]:
        """Queries and yields SpanChunks for doc_id strictly in ascending chunk_index order."""
        cursor = self._conn.execute(
            "SELECT chunk_id FROM chunk_locations WHERE doc_id = ?",
            (doc_id,),
        )
        rows = cursor.fetchall()
        if not rows:
            # Check if doc exists at all
            if not self.contains_official_doc(doc_id):
                doc_exists = self._conn.execute(
                    "SELECT 1 FROM document_locations WHERE doc_id = ?", (doc_id,)
                ).fetchone()
                if not doc_exists:
                    raise KeyError(f"Document {doc_id} not found in snapshot {self.snapshot_path.name}")
            return

        chunk_ids = [r[0] for r in rows]
        chunks = self.get_chunks(chunk_ids)
        chunks.sort(key=lambda c: c.chunk_index)
        for chunk in chunks:
            yield chunk

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> SnapshotSourceStore:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()


class EvidenceSourceStore:
    """Wrapper implementing SourceStore protocol over an underlying SourceStore.
    
    Supports transparent resolution of both standard SpanChunk IDs and Phase 2 EvidenceSpan IDs
    via document character slicing and registration.
    """

    def __init__(self, underlying: SourceStore) -> None:
        self.underlying = underlying
        self._registered_spans: dict[str, EvidenceSpan] = {}
        self._span_chunk_cache: LRUCache[str, SpanChunk] = LRUCache(
            max_entries=4096,
            max_bytes=64 * 1024 * 1024,
            size_fn=_chunk_size,
        )

    def register_evidence_span(self, span: EvidenceSpan) -> None:
        """Register a Phase 2 EvidenceSpan for resolution."""
        self._registered_spans[span.span_id] = span

    def register_evidence_spans(self, spans: Sequence[EvidenceSpan]) -> None:
        """Batch-register Phase 2 EvidenceSpans."""
        for span in spans:
            self._registered_spans[span.span_id] = span

    def contains_official_doc(self, doc_id: int) -> bool:
        return self.underlying.contains_official_doc(doc_id)

    def get_document(self, doc_id: int) -> ExtractedDocument:
        return self.underlying.get_document(doc_id)

    def get_documents(self, doc_ids: Sequence[int]) -> list[ExtractedDocument]:
        if hasattr(self.underlying, "get_documents"):
            return self.underlying.get_documents(doc_ids)
        return [self.underlying.get_document(d) for d in doc_ids]

    def iter_chunks_for_doc(self, doc_id: int) -> Iterator[SpanChunk]:
        if hasattr(self.underlying, "iter_chunks_for_doc"):
            return self.underlying.iter_chunks_for_doc(doc_id)
        raise NotImplementedError("Underlying store does not support iter_chunks_for_doc")

    def _resolve_evidence_span(self, chunk_id: str) -> SpanChunk | None:
        if chunk_id in self._span_chunk_cache:
            return self._span_chunk_cache[chunk_id]

        if chunk_id in self._registered_spans:
            span = self._registered_spans[chunk_id]
            doc = self.get_document(span.doc_id)
            if not (0 <= span.start_char < span.end_char <= len(doc.source_text)):
                raise ValueError(
                    f"Invalid span bounds for {chunk_id}: [{span.start_char}:{span.end_char}] "
                    f"in document of length {len(doc.source_text)}"
                )
            text = doc.source_text[span.start_char : span.end_char]
            tok_count = text.count(" ") + 1 if text.strip() else 0
            chunk = SpanChunk(
                chunk_id=span.span_id,
                doc_id=span.doc_id,
                source_version=span.source_version,
                source_sha256=span.source_sha256,
                start_char=span.start_char,
                end_char=span.end_char,
                chunk_index=-1,
                language=doc.language,
                section=None,
                text=text,
                index_text=text,
                dense_token_count=tok_count,
                reranker_token_count=tok_count,
            )
            self._span_chunk_cache[chunk_id] = chunk
            return chunk

        if chunk_id.startswith("span:"):
            parts = chunk_id.split(":")
            if len(parts) >= 4:
                try:
                    doc_id = int(parts[1])
                    start_c = int(parts[2])
                    end_c = int(parts[3])
                except (ValueError, IndexError):
                    return None

                doc = self.get_document(doc_id)
                if not (0 <= start_c < end_c <= len(doc.source_text)):
                    raise ValueError(
                        f"Invalid span bounds for {chunk_id}: [{start_c}:{end_c}] "
                        f"in document of length {len(doc.source_text)}"
                    )
                text = doc.source_text[start_c:end_c]
                tok_count = text.count(" ") + 1 if text.strip() else 0
                chunk = SpanChunk(
                    chunk_id=chunk_id,
                    doc_id=doc.doc_id,
                    source_version=doc.source_version,
                    source_sha256=doc.source_sha256,
                    start_char=start_c,
                    end_char=end_c,
                    chunk_index=-1,
                    language=doc.language,
                    section=None,
                    text=text,
                    index_text=text,
                    dense_token_count=tok_count,
                    reranker_token_count=tok_count,
                )
                self._span_chunk_cache[chunk_id] = chunk
                return chunk

        return None

    def get_chunk(self, chunk_id: str) -> SpanChunk:
        """Retrieves chunk by resolving evidence span or querying underlying store."""
        span_chunk = self._resolve_evidence_span(chunk_id)
        if span_chunk is not None:
            return span_chunk
        return self.underlying.get_chunk(chunk_id)

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[SpanChunk]:
        """Batch-retrieves chunks, resolving evidence spans and delegating standard chunks."""
        if not chunk_ids:
            return []

        resolved: dict[str, SpanChunk] = {}
        underlying_ids: list[str] = []

        for cid in set(chunk_ids):
            sc = self._resolve_evidence_span(cid)
            if sc is not None:
                resolved[cid] = sc
            else:
                underlying_ids.append(cid)

        if underlying_ids:
            if hasattr(self.underlying, "get_chunks"):
                underlying_chunks = self.underlying.get_chunks(underlying_ids)
                for c in underlying_chunks:
                    resolved[c.chunk_id] = c
            else:
                for uid in underlying_ids:
                    resolved[uid] = self.underlying.get_chunk(uid)

        return [resolved[cid] for cid in chunk_ids]

    def close(self) -> None:
        if hasattr(self.underlying, "close"):
            self.underlying.close()

    def __enter__(self) -> EvidenceSourceStore:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
