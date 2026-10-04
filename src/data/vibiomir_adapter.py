import hashlib
import json
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.dataset_adapter import _normalize_language
from src.data.schemas import ChunkRecord, DocumentRecord, QueryRecord
from src.data.vibiomir_contracts import (
    EvaluationRecord,
    OfficialQuery,
    OfficialSource,
    ReferenceChunk,
    SnapshotManifest,
)


def compute_file_sha256(path: Path | str, chunk_size: int = 65536) -> str:
    """Computes the SHA-256 hex digest of a file."""
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


class OfficialSourceIterator:
    """Streaming iterator over official sources in Parquet with lazy SHA256 computation."""

    def __init__(self, path: Path | str, batch_size: int = 10000) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Corpus file not found: {self.path}")
        self.batch_size = batch_size
        self._sha256: str | None = None

    @property
    def input_sha256(self) -> str:
        if self._sha256 is None:
            self._sha256 = compute_file_sha256(self.path)
        return self._sha256

    def __iter__(self) -> Iterator[OfficialSource]:
        parquet_file = pq.ParquetFile(self.path)
        schema = parquet_file.schema_arrow

        if "id" not in schema.names or "url" not in schema.names:
            raise ValueError(
                f"Parquet file must contain 'id' and 'url' columns, got {schema.names}"
            )

        id_field = schema.field("id")
        url_field = schema.field("url")

        if not pa.types.is_integer(id_field.type):
            raise ValueError(
                f"Column 'id' must be integer type (signed int64), got {id_field.type}"
            )
        if not (pa.types.is_string(url_field.type) or pa.types.is_large_string(url_field.type)):
            raise ValueError(
                f"Column 'url' must be string type, got {url_field.type}"
            )

        for batch in parquet_file.iter_batches(
            batch_size=self.batch_size, columns=["id", "url"]
        ):
            id_col = batch.column("id")
            url_col = batch.column("url")

            if id_col.null_count > 0:
                raise ValueError("Column 'id' contains null values")
            if url_col.null_count > 0:
                raise ValueError("Column 'url' contains null values")

            for raw_id, raw_url in zip(id_col.to_pylist(), url_col.to_pylist()):
                if isinstance(raw_id, bool) or not isinstance(raw_id, int):
                    raise ValueError(f"doc_id must be integer, got {type(raw_id).__name__}: {raw_id}")
                if not (-9223372036854775808 <= raw_id <= 9223372036854775807):
                    raise ValueError(f"doc_id out of signed int64 range: {raw_id}")
                if not isinstance(raw_url, str) or not raw_url.strip():
                    raise ValueError(f"url must be non-empty string, got {raw_url!r}")

                yield OfficialSource(doc_id=raw_id, url=raw_url.strip())


def iter_official_sources(
    path: Path | str, batch_size: int = 10000
) -> OfficialSourceIterator:
    """Returns a streaming iterator over official sources in Parquet."""
    return OfficialSourceIterator(path=path, batch_size=batch_size)


def load_official_queries(
    path: Path | str, expected_count: int | None = None
) -> tuple[OfficialQuery, ...]:
    """Loads and validates official queries from query.parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Query file not found: {path}")

    parquet_file = pq.ParquetFile(path)
    schema = parquet_file.schema_arrow

    if "id" not in schema.names or "query" not in schema.names:
        raise ValueError(
            f"Parquet file must contain 'id' and 'query' columns, got {schema.names}"
        )

    id_field = schema.field("id")
    query_field = schema.field("query")

    if not pa.types.is_integer(id_field.type):
        raise ValueError(
            f"Column 'id' must be integer type (signed int64), got {id_field.type}"
        )
    if not (pa.types.is_string(query_field.type) or pa.types.is_large_string(query_field.type)):
        raise ValueError(
            f"Column 'query' must be string type, got {query_field.type}"
        )

    table = parquet_file.read(columns=["id", "query"])
    id_col = table.column("id")
    query_col = table.column("query")

    if id_col.null_count > 0:
        raise ValueError("Column 'id' contains null values")
    if query_col.null_count > 0:
        raise ValueError("Column 'query' contains null values")

    queries: list[OfficialQuery] = []
    seen_ids: set[int] = set()

    for raw_id, raw_query in zip(id_col.to_pylist(), query_col.to_pylist()):
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise ValueError(f"query_id must be integer, got {type(raw_id).__name__}: {raw_id}")
        if not (-9223372036854775808 <= raw_id <= 9223372036854775807):
            raise ValueError(f"query_id out of signed int64 range: {raw_id}")
        if not isinstance(raw_query, str) or not raw_query.strip():
            raise ValueError(f"query must be non-empty string, got {raw_query!r}")

        if raw_id in seen_ids:
            raise ValueError(f"Duplicate query_id found: {raw_id}")
        seen_ids.add(raw_id)

        queries.append(OfficialQuery(query_id=raw_id, query=raw_query.strip()))

    if expected_count is not None and len(queries) != expected_count:
        raise ValueError(
            f"Expected {expected_count} queries, but found {len(queries)} in {path}"
        )

    return tuple(queries)


class ViBioMIRAdapter:
    """Dataset adapter streaming documents, chunks, queries, and ground truth from a frozen snapshot."""

    def __init__(
        self,
        snapshot_path: Path | str,
        queries_path: Path | str | None = None,
        qrels_path: Path | str | None = None,
        batch_size: int = 5000,
    ) -> None:
        self.snapshot_path = Path(snapshot_path)
        if not self.snapshot_path.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {self.snapshot_path}")

        manifest_file = self.snapshot_path / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Snapshot manifest not found at {manifest_file}")

        with open(manifest_file, "r", encoding="utf-8") as f:
            m_data = json.load(f)

        self.manifest = SnapshotManifest(
            snapshot_id=m_data["snapshot_id"],
            created_at=m_data["created_at"],
            registry_sha256=m_data["registry_sha256"],
            queries_sha256=m_data["queries_sha256"],
            source_membership_sha256=m_data["source_membership_sha256"],
            extractor_fingerprint=m_data["extractor_fingerprint"],
            chunker_fingerprint=m_data["chunker_fingerprint"],
            source_count=m_data["source_count"],
            ready_document_count=m_data["ready_document_count"],
            chunk_count=m_data["chunk_count"],
            language_counts=m_data["language_counts"],
            quarantine_counts=m_data["quarantine_counts"],
            shard_paths_and_hashes=m_data["shard_paths_and_hashes"],
        )

        self.queries_path = Path(queries_path) if queries_path is not None else None
        self.qrels_path = Path(qrels_path) if qrels_path is not None else None
        self.batch_size = batch_size

    def iter_documents(self) -> Iterator[DocumentRecord]:
        doc_shards = sorted(self.snapshot_path.glob("documents_*.parquet"))
        for shard_file in doc_shards:
            pfile = pq.ParquetFile(shard_file)
            for batch in pfile.iter_batches(batch_size=self.batch_size):
                b_dict = batch.to_pydict()
                for i in range(batch.num_rows):
                    official_id = b_dict["doc_id"][i]
                    lang_raw = b_dict["language"][i]
                    norm_lang = _normalize_language(lang_raw)
                    yield DocumentRecord(
                        doc_id=str(official_id),
                        language=norm_lang,
                        text=b_dict["source_text"][i],
                        title=b_dict["title"][i],
                        metadata={
                            "official_doc_id": official_id,
                            "source_sha256": b_dict["source_sha256"][i],
                            "raw_sha256": b_dict["raw_sha256"][i],
                            "source_version": b_dict["source_version"][i],
                        },
                    )

    def iter_chunks(self) -> Iterator[ChunkRecord]:
        chunk_shards = sorted(self.snapshot_path.glob("chunks_*.parquet"))
        for shard_file in chunk_shards:
            pfile = pq.ParquetFile(shard_file)
            for batch in pfile.iter_batches(batch_size=self.batch_size):
                b_dict = batch.to_pydict()
                for i in range(batch.num_rows):
                    official_id = b_dict["doc_id"][i]
                    lang_raw = b_dict["language"][i]
                    norm_lang = _normalize_language(lang_raw)
                    yield ChunkRecord(
                        chunk_id=str(b_dict["chunk_id"][i]),
                        doc_id=str(official_id),
                        language=norm_lang,
                        text=str(b_dict["text"][i]),
                        chunk_index=int(b_dict["chunk_index"][i]),
                        metadata={
                            "official_doc_id": official_id,
                            "source_sha256": b_dict["source_sha256"][i],
                            "start_char": b_dict["start_char"][i],
                            "end_char": b_dict["end_char"][i],
                            "section": b_dict["section"][i],
                            "index_text": b_dict["index_text"][i],
                            "dense_token_count": b_dict["dense_token_count"][i],
                            "reranker_token_count": b_dict["reranker_token_count"][i],
                        },
                    )

    def load_documents(self) -> list[DocumentRecord]:
        return list(self.iter_documents())

    def load_chunks(self) -> list[ChunkRecord]:
        return list(self.iter_chunks())

    def load_queries(self, split: str = "test") -> list[QueryRecord]:
        q_path = self.queries_path
        if q_path is None:
            cand1 = self.snapshot_path / f"query_{split}.parquet"
            cand2 = self.snapshot_path / "query.parquet"
            cand3 = self.snapshot_path / "queries.parquet"
            for c in (cand1, cand2, cand3):
                if c.exists():
                    q_path = c
                    break

        if q_path is None or not q_path.exists():
            return []

        official_queries = load_official_queries(q_path)
        return [
            QueryRecord(
                query_id=str(q.query_id),
                text_vi=q.query,
                metadata={"official_query_id": q.query_id},
            )
            for q in official_queries
        ]

    def load_ground_truth(self, split: str = "dev") -> list[EvaluationRecord]:
        if self.qrels_path is None or not self.qrels_path.exists():
            return []

        results: list[EvaluationRecord] = []
        with open(self.qrels_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                qid = int(data["query_id"])
                docs = frozenset(int(d) for d in data.get("relevant_docs", []))
                chunks = tuple(
                    ReferenceChunk(doc_id=int(c["doc_id"]), chunk_text=str(c["chunk_text"]))
                    for c in data.get("relevant_chunks", [])
                )
                results.append(EvaluationRecord(query_id=qid, relevant_doc_ids=docs, relevant_chunks=chunks))
        return results


def validate_index_manifest(index_path: Path | str, snapshot: SnapshotManifest) -> None:
    """Validates that an index manifest aligns with the given snapshot manifest."""
    path = Path(index_path)
    if path.is_file():
        manifest_files = [path]
    elif path.is_dir():
        manifest_files = list(path.rglob("manifest.json"))
        if not manifest_files:
            raise FileNotFoundError(f"No manifest.json found in {index_path}")
    else:
        raise FileNotFoundError(f"Index path does not exist: {index_path}")

    for mf in manifest_files:
        with open(mf, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "snapshot_id" not in data:
            raise ValueError(f"Index manifest {mf.name} missing required field 'snapshot_id'")

        if data["snapshot_id"] != snapshot.snapshot_id:
            raise ValueError(
                f"Snapshot ID mismatch in {mf}: index has '{data['snapshot_id']}', expected '{snapshot.snapshot_id}'"
            )

        if "source_membership_sha256" in data and data["source_membership_sha256"] != snapshot.source_membership_sha256:
            raise ValueError(
                f"Source membership SHA-256 mismatch in {mf}: index has '{data['source_membership_sha256']}', expected '{snapshot.source_membership_sha256}'"
            )

        if "chunker_fingerprint" in data and data["chunker_fingerprint"] != snapshot.chunker_fingerprint:
            raise ValueError(
                f"Chunker fingerprint mismatch in {mf}: index has '{data['chunker_fingerprint']}', expected '{snapshot.chunker_fingerprint}'"
            )
