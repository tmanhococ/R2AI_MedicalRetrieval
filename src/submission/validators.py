from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import zipfile

from src.data.vibiomir_contracts import (
    OfficialQuery,
    SourceStore,
    SubmissionChunk,
    SubmissionRow,
    ValidationReport,
)

INT64_MIN = -9223372036854775808
INT64_MAX = 9223372036854775807


class SubmissionValidationError(ValueError):
    """Raised when submission rows or zip archives fail strict validation."""

    def __init__(self, message: str, report: ValidationReport) -> None:
        super().__init__(message)
        self.report = report


def _is_int64(val: Any) -> bool:
    """Returns True if val is a signed 64-bit integer (rejecting bool)."""
    return isinstance(val, int) and not isinstance(val, bool) and (INT64_MIN <= val <= INT64_MAX)


def _extract_snapshot_id(store: SourceStore) -> str | None:
    """Extracts snapshot identifier from SourceStore instance if available."""
    if hasattr(store, "snapshot_id"):
        return str(store.snapshot_id)
    if hasattr(store, "manifest_data") and isinstance(store.manifest_data, dict):
        return store.manifest_data.get("snapshot_id")
    return None


def validate_predictions(
    rows: Sequence[SubmissionRow] | Any,
    queries: Sequence[OfficialQuery],
    store: SourceStore,
    expected_count: int | None = None,
) -> ValidationReport:
    """Strictly validates submission rows against official queries and SourceStore snapshot.

    Checks:
    1. Top-level list and sequence structure.
    2. Exact query coverage: no missing, extra, or duplicate query IDs.
    3. Signed int64 IDs: rejects bool, string, float, out-of-range IDs.
    4. Relevant docs/chunks list types: rejects missing or non-list types.
    5. Registry membership: rejects foreign doc IDs not in store.
    6. Deduplication: rejects duplicate doc IDs and duplicate (doc_id, chunk_text) in same query.
    7. Exact source substring provenance: verifies chunk text is verbatim substring of doc source_text.
    8. Valid empty predictions pass with empty lists.
    9. Chunk doc_id not in relevant_docs is accepted if valid official doc.
    10. Non-negative integer chunk_order (rejecting bool, negative, float).
    """
    errors: list[str] = []
    warnings: list[str] = []
    snapshot_id = _extract_snapshot_id(store)

    if not isinstance(rows, (list, tuple)):
        return ValidationReport(
            valid=False,
            query_count=0,
            document_count=0,
            chunk_count=0,
            errors=(f"Submission rows must be a list/sequence, got {type(rows).__name__}",),
            snapshot_id=snapshot_id,
        )

    expected_query_ids = {q.query_id for q in queries}
    if expected_count is not None and len(rows) != expected_count:
        errors.append(f"Expected query count was {expected_count}, got {len(rows)}")


    seen_query_ids: set[int] = set()
    duplicate_query_ids: set[int] = set()
    total_docs = 0
    total_chunks = 0

    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"Row at index {idx} is not a dictionary: {type(row).__name__}")
            continue

        # Validate Query ID
        if "id" not in row:
            errors.append(f"Row at index {idx} missing required key 'id'")
            continue

        raw_id = row["id"]
        if not _is_int64(raw_id):
            errors.append(
                f"Row at index {idx}: query ID must be signed int64, got {type(raw_id).__name__} ({raw_id!r})"
            )
            continue

        query_id = int(raw_id)
        if query_id in seen_query_ids:
            duplicate_query_ids.add(query_id)
        seen_query_ids.add(query_id)

        # Validate relevant_docs
        if "relevant_docs" not in row:
            errors.append(f"Query {query_id}: missing 'relevant_docs' field")
        elif not isinstance(row["relevant_docs"], list):
            errors.append(
                f"Query {query_id}: 'relevant_docs' must be a list, got {type(row['relevant_docs']).__name__}"
            )
        else:
            rel_docs = row["relevant_docs"]
            total_docs += len(rel_docs)
            seen_doc_ids: set[int] = set()
            for doc_idx, raw_doc_id in enumerate(rel_docs):
                if not _is_int64(raw_doc_id):
                    errors.append(
                        f"Query {query_id}: relevant_docs[{doc_idx}] doc_id must be integer (signed int64), "
                        f"got {type(raw_doc_id).__name__} ({raw_doc_id!r})"
                    )
                    continue

                doc_id = int(raw_doc_id)
                if doc_id in seen_doc_ids:
                    errors.append(f"Query {query_id}: duplicate document ID {doc_id} in relevant_docs")
                seen_doc_ids.add(doc_id)

                if not store.contains_official_doc(doc_id):
                    errors.append(f"Query {query_id}: foreign doc ID {doc_id} not in official registry/snapshot")

        # Validate relevant_chunks
        if "relevant_chunks" not in row:
            errors.append(f"Query {query_id}: missing 'relevant_chunks' field")
        elif not isinstance(row["relevant_chunks"], list):
            errors.append(
                f"Query {query_id}: 'relevant_chunks' must be a list, got {type(row['relevant_chunks']).__name__}"
            )
        else:
            rel_chunks = row["relevant_chunks"]
            total_chunks += len(rel_chunks)
            seen_chunks: set[tuple[int, str]] = set()

            for chunk_idx, chunk in enumerate(rel_chunks):
                if not isinstance(chunk, dict):
                    errors.append(
                        f"Query {query_id}: relevant_chunks[{chunk_idx}] must be dict, got {type(chunk).__name__}"
                    )
                    continue

                # doc_id in chunk
                if "doc_id" not in chunk:
                    errors.append(f"Query {query_id}: relevant_chunks[{chunk_idx}] missing 'doc_id'")
                    continue
                raw_chunk_doc_id = chunk["doc_id"]
                if not _is_int64(raw_chunk_doc_id):
                    errors.append(
                        f"Query {query_id}: relevant_chunks[{chunk_idx}] doc_id must be integer (signed int64), "
                        f"got {type(raw_chunk_doc_id).__name__} ({raw_chunk_doc_id!r})"
                    )
                    continue

                chunk_doc_id = int(raw_chunk_doc_id)

                # chunk_text
                if "chunk_text" not in chunk:
                    errors.append(f"Query {query_id}: relevant_chunks[{chunk_idx}] missing 'chunk_text'")
                    continue
                chunk_text = chunk["chunk_text"]
                if not isinstance(chunk_text, str) or not chunk_text.strip():
                    errors.append(
                        f"Query {query_id}: relevant_chunks[{chunk_idx}] 'chunk_text' must be non-empty string"
                    )
                    continue

                # Duplicate chunk pair
                chunk_pair = (chunk_doc_id, chunk_text)
                if chunk_pair in seen_chunks:
                    errors.append(
                        f"Query {query_id}: duplicate chunk (doc_id={chunk_doc_id}, chunk_text) in relevant_chunks"
                    )
                seen_chunks.add(chunk_pair)

                # Official registry check
                if not store.contains_official_doc(chunk_doc_id):
                    errors.append(
                        f"Query {query_id}: foreign chunk doc_id {chunk_doc_id} not in official registry/snapshot"
                    )
                    continue

                # chunk_order check if present
                if "chunk_order" in chunk:
                    order = chunk["chunk_order"]
                    if isinstance(order, bool) or not isinstance(order, int) or order < 0:
                        errors.append(
                            f"Query {query_id}: chunk_order must be non-negative integer, got {order!r}"
                        )

                # Exact substring match in snapshot source_text
                try:
                    doc = store.get_document(chunk_doc_id)
                    if chunk_text not in doc.source_text:
                        errors.append(
                            f"Query {query_id}: chunk_text is not a substring of source document {chunk_doc_id} in snapshot"
                        )
                except KeyError:
                    errors.append(
                        f"Query {query_id}: source document {chunk_doc_id} not found in snapshot store"
                    )
                except Exception as e:
                    errors.append(
                        f"Query {query_id}: failed to load document {chunk_doc_id} from snapshot: {e}"
                    )

    # Check query coverage
    missing_queries = expected_query_ids - seen_query_ids
    if missing_queries:
        sample_missing = sorted(list(missing_queries))[:5]
        errors.append(f"Missing {len(missing_queries)} queries (sample: {sample_missing})")

    extra_queries = seen_query_ids - expected_query_ids
    if extra_queries:
        sample_extra = sorted(list(extra_queries))[:5]
        errors.append(f"Found {len(extra_queries)} extra/unknown queries (sample: {sample_extra})")

    if duplicate_query_ids:
        sample_dup = sorted(list(duplicate_query_ids))[:5]
        errors.append(f"Found {len(duplicate_query_ids)} duplicate queries (sample: {sample_dup})")

    is_valid = len(errors) == 0
    return ValidationReport(
        valid=is_valid,
        query_count=len(rows),
        document_count=total_docs,
        chunk_count=total_chunks,
        errors=tuple(errors),
        warnings=tuple(warnings),
        snapshot_id=snapshot_id,
    )


def validate_zip(
    path: Path | str,
    queries: Sequence[OfficialQuery],
    store: SourceStore,
    expected_count: int | None = None,
) -> ValidationReport:
    """Strictly validates a ViBioMIR submission ZIP archive.

    Checks:
    1. ZIP exists and is a valid non-empty archive.
    2. Contains exactly one regular entry named 'results.json' at root.
    3. No directories, subdirectories, extra files, or encryption.
    4. results.json decompresses into UTF-8 JSON array.
    5. Runs validate_predictions on parsed content.
    """
    zip_path = Path(path)
    snapshot_id = _extract_snapshot_id(store)

    if not zip_path.exists():
        return ValidationReport(
            valid=False,
            query_count=0,
            document_count=0,
            chunk_count=0,
            errors=(f"Submission ZIP file not found: {zip_path}",),
            snapshot_id=snapshot_id,
        )

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infolist = zf.infolist()
            if not infolist:
                return ValidationReport(
                    valid=False,
                    query_count=0,
                    document_count=0,
                    chunk_count=0,
                    errors=("ZIP archive is empty",),
                    snapshot_id=snapshot_id,
                )

            if len(infolist) > 1:
                filenames = [info.filename for info in infolist]
                return ValidationReport(
                    valid=False,
                    query_count=0,
                    document_count=0,
                    chunk_count=0,
                    errors=(
                        f"ZIP archive must contain exactly one root file named 'results.json', "
                        f"found {len(infolist)} entries: {filenames}",
                    ),
                    snapshot_id=snapshot_id,
                )

            entry = infolist[0]
            if entry.is_dir():
                return ValidationReport(
                    valid=False,
                    query_count=0,
                    document_count=0,
                    chunk_count=0,
                    errors=("ZIP archive entry cannot be a directory",),
                    snapshot_id=snapshot_id,
                )

            # Check filename: must be strictly "results.json" without directory prefixes
            if "/" in entry.filename or "\\" in entry.filename or entry.filename != "results.json":
                return ValidationReport(
                    valid=False,
                    query_count=0,
                    document_count=0,
                    chunk_count=0,
                    errors=(
                        f"ZIP entry must be exactly 'results.json' at root (no subdirectories), "
                        f"got {entry.filename!r}",
                    ),
                    snapshot_id=snapshot_id,
                )

            # Check encryption flag
            if entry.flag_bits & 0x1:
                return ValidationReport(
                    valid=False,
                    query_count=0,
                    document_count=0,
                    chunk_count=0,
                    errors=("ZIP entry 'results.json' is encrypted",),
                    snapshot_id=snapshot_id,
                )

            raw_bytes = zf.read(entry)

    except zipfile.BadZipFile as e:
        return ValidationReport(
            valid=False,
            query_count=0,
            document_count=0,
            chunk_count=0,
            errors=(f"Corrupt or invalid ZIP archive: {e}",),
            snapshot_id=snapshot_id,
        )
    except Exception as e:
        return ValidationReport(
            valid=False,
            query_count=0,
            document_count=0,
            chunk_count=0,
            errors=(f"Error reading ZIP archive: {e}",),
            snapshot_id=snapshot_id,
        )

    # Decompress and parse results.json
    try:
        decoded_text = raw_bytes.decode("utf-8")
        data = json.loads(decoded_text)
    except UnicodeDecodeError as e:
        return ValidationReport(
            valid=False,
            query_count=0,
            document_count=0,
            chunk_count=0,
            errors=(f"results.json is not valid UTF-8: {e}",),
            snapshot_id=snapshot_id,
        )
    except json.JSONDecodeError as e:
        return ValidationReport(
            valid=False,
            query_count=0,
            document_count=0,
            chunk_count=0,
            errors=(f"Failed to parse results.json as JSON: {e}",),
            snapshot_id=snapshot_id,
        )

    if not isinstance(data, list):
        return ValidationReport(
            valid=False,
            query_count=0,
            document_count=0,
            chunk_count=0,
            errors=(f"results.json root element must be a JSON array, got {type(data).__name__}",),
            snapshot_id=snapshot_id,
        )

    # Delegate predictions validation
    return validate_predictions(
        rows=data,
        queries=queries,
        store=store,
        expected_count=expected_count,
    )
