"""Development query records loader and provenance validator for ViBioMIR.

Enforces strict provenance, anti-leakage, and schema constraints:
- Rejects dev query IDs that collide with official query IDs (namespace separation).
- Rejects qrels marked with test, public, or private splits to prevent data leakage.
- Validates metadata, non-empty reference chunk text, and source existence in SourceStore.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.vibiomir_contracts import (
    DevQueryMetadata,
    EvaluationRecord,
    ReferenceChunk,
    SourceStore,
)

FORBIDDEN_SPLITS = frozenset({"test", "public", "private", "official_test", "test_set"})


def load_dev_records(
    path: Path,
    official_query_ids: set[int],
    store: SourceStore,
) -> tuple[EvaluationRecord, ...]:
    """Load development evaluation records with strict provenance checks.

    Args:
        path: Path to development JSONL or JSON file.
        official_query_ids: Set of official query IDs (1..1,200) to check namespace separation.
        store: SourceStore containing frozen snapshot documents for reference verification.

    Returns:
        Tuple of validated EvaluationRecord instances.

    Raises:
        ValueError: If namespace collides, split is forbidden, chunk text is empty,
                    or referenced doc_id is missing from SourceStore.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dev records file not found at {path}")

    # Read records from JSONL or JSON
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        return ()

    raw_records: list[dict[str, Any]] = []
    if content.startswith("["):
        raw_records = json.loads(content)
    else:
        for line_num, line in enumerate(content.splitlines(), start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                raw_records.append(json.loads(line_str))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSON on line {line_num} of {path}: {exc}"
                ) from exc

    records: list[EvaluationRecord] = []
    seen_query_ids: set[int] = set()

    for item in raw_records:
        query_id = item.get("query_id", item.get("id"))
        if query_id is None or isinstance(query_id, bool) or not isinstance(query_id, int):
            raise ValueError(f"Invalid query_id in dev record: {query_id!r}")

        # 1. Namespace Separation: Must not collide with official query IDs
        if query_id in official_query_ids:
            raise ValueError(
                f"Dev query ID {query_id} collides with official query namespace (size {len(official_query_ids)})"
            )

        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate dev query ID found: {query_id}")
        seen_query_ids.add(query_id)

        # 2. Extract and Validate Metadata
        raw_meta = item.get("metadata")
        if raw_meta is not None:
            split = str(raw_meta.get("split", "")).strip().lower()
            origin = str(raw_meta.get("origin", "")).strip()
            query_text = str(raw_meta.get("query_text", item.get("query", ""))).strip()
            source_doc_ids = tuple(int(d) for d in raw_meta.get("source_doc_ids", []))
            topic = str(raw_meta.get("topic", "")).strip()
            question_style = str(raw_meta.get("question_style", "")).strip()
            pool_version = str(raw_meta.get("pool_version", "")).strip()
            judgment_coverage = float(raw_meta.get("judgment_coverage", 1.0))
        else:
            split = str(item.get("split", "")).strip().lower()
            origin = str(item.get("origin", "")).strip()
            query_text = str(item.get("query_text", item.get("query", ""))).strip()
            source_doc_ids = tuple(int(d) for d in item.get("source_doc_ids", []))
            topic = str(item.get("topic", "")).strip()
            question_style = str(item.get("question_style", "")).strip()
            pool_version = str(item.get("pool_version", "")).strip()
            judgment_coverage = float(item.get("judgment_coverage", 1.0))

        # 3. Anti-leakage: Reject test/public/private splits
        if split in FORBIDDEN_SPLITS:
            raise ValueError(
                f"Forbidden split '{split}' in dev records for query {query_id}: "
                "test/public/private qrels are rejected to prevent evaluation leakage."
            )

        dev_metadata = DevQueryMetadata(
            query_id=query_id,
            query_text=query_text,
            origin=origin,
            source_doc_ids=source_doc_ids,
            topic=topic,
            question_style=question_style,
            split=split,
            pool_version=pool_version,
            judgment_coverage=judgment_coverage,
        )

        # 4. Extract Relevant Docs and Validate Store Existence
        raw_doc_ids = item.get("relevant_doc_ids", item.get("relevant_docs", []))
        relevant_doc_ids = frozenset(int(d) for d in raw_doc_ids)

        for doc_id in relevant_doc_ids:
            if not store.contains_official_doc(doc_id):
                raise ValueError(
                    f"Document {doc_id} referenced in dev query {query_id} not found in SourceStore"
                )

        # 5. Extract Reference Chunks and Validate Non-Empty and Store Existence
        raw_chunks = item.get("relevant_chunks", [])
        ref_chunks: list[ReferenceChunk] = []

        for c_item in raw_chunks:
            c_doc_id = int(c_item["doc_id"])
            c_text = str(c_item.get("chunk_text", "")).strip()

            if not c_text:
                raise ValueError(
                    f"Reference chunk for query {query_id}, doc {c_doc_id} has empty chunk_text"
                )

            if not store.contains_official_doc(c_doc_id):
                raise ValueError(
                    f"Document {c_doc_id} in reference chunk for query {query_id} not found in SourceStore"
                )

            c_start = c_item.get("char_start")
            c_end = c_item.get("char_end")
            ref_chunks.append(
                ReferenceChunk(
                    doc_id=c_doc_id,
                    chunk_text=c_text,
                    char_start=int(c_start) if c_start is not None else None,
                    char_end=int(c_end) if c_end is not None else None,
                )
            )

        records.append(
            EvaluationRecord(
                query_id=query_id,
                relevant_doc_ids=relevant_doc_ids,
                relevant_chunks=tuple(ref_chunks),
                metadata=dev_metadata,
            )
        )

    return tuple(records)
