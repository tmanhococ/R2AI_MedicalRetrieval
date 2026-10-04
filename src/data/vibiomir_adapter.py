import hashlib
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from src.data.vibiomir_contracts import (
    OfficialQuery,
    OfficialSource,
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
