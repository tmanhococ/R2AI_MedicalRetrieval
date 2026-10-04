from __future__ import annotations

import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

import pyarrow.parquet as pq

from src.data.vibiomir_contracts import (
    ExtractedBlock,
    ExtractedDocument,
    SourceStore,
    SpanChunk,
)

logger = logging.getLogger(__name__)


class SnapshotSourceStore:
    """Reader implementing SourceStore protocol over an immutable frozen snapshot."""

    def __init__(self, snapshot_path: Path | str, cache_size: int = 2048) -> None:
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

        self._doc_cache: dict[int, ExtractedDocument] = {}
        self._chunk_cache: dict[str, SpanChunk] = {}
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
                    row_idx INTEGER NOT NULL
                );
                """
            )

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
                table = pq.read_table(chunk_file, columns=["chunk_id", "doc_id"])
                chunk_ids = table.column("chunk_id").to_pylist()
                c_doc_ids = table.column("doc_id").to_pylist()
                rows = [(cid, did, chunk_file.name, idx) for idx, (cid, did) in enumerate(zip(chunk_ids, c_doc_ids))]
                conn.executemany(
                    "INSERT OR REPLACE INTO chunk_locations (chunk_id, doc_id, shard, row_idx) VALUES (?, ?, ?, ?)",
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

        row = self._conn.execute(
            "SELECT shard FROM document_locations WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"Document {doc_id} not found in snapshot {self.snapshot_path.name}")

        shard_name = row[0]
        shard_path = self.snapshot_path / shard_name
        table = pq.read_table(shard_path, filters=[("doc_id", "==", doc_id)])
        if table.num_rows == 0:
            raise KeyError(f"Document {doc_id} not found in shard {shard_name}")

        row_dict = table.to_pylist()[0]
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

        doc = ExtractedDocument(
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

        if len(self._doc_cache) >= self._cache_size:
            self._doc_cache.clear()
        self._doc_cache[doc_id] = doc
        return doc

    def get_chunk(self, chunk_id: str) -> SpanChunk:
        """Retrieves exact SpanChunk for a chunk_id present in this snapshot."""
        if chunk_id in self._chunk_cache:
            return self._chunk_cache[chunk_id]

        row = self._conn.execute(
            "SELECT shard FROM chunk_locations WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        if not row:
            raise KeyError(f"Chunk {chunk_id} not found in snapshot {self.snapshot_path.name}")

        shard_name = row[0]
        shard_path = self.snapshot_path / shard_name
        table = pq.read_table(shard_path, filters=[("chunk_id", "==", chunk_id)])
        if table.num_rows == 0:
            raise KeyError(f"Chunk {chunk_id} not found in shard {shard_name}")

        row_dict = table.to_pylist()[0]
        chunk = SpanChunk(
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

        if len(self._chunk_cache) >= self._cache_size:
            self._chunk_cache.clear()
        self._chunk_cache[chunk_id] = chunk
        return chunk

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SnapshotSourceStore":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
