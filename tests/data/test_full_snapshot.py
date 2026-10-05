import gzip
import hashlib
import json
from pathlib import Path
import pytest

from src.data.phase2_contracts import EvidenceSpan, SCOPE_FULL_SOURCE_PASS_WITH_GAPS, SCOPE_FULL_BODY_INDEX
from src.data.source_store import SnapshotSourceStore, EvidenceSourceStore, LRUCache
from src.data.vibiomir_contracts import (
    ChunkConfig,
    ExtractionConfig,
    ExtractedDocument,
    FetchResult,
    OfficialSource,
    SnapshotConfig,
    SourceStore,
    SpanChunk,
)
from src.data.vibiomir_registry import SourceRegistry, ingest_sources
from src.ingestion.fetcher import save_raw_atomic
from src.ingestion.snapshot import build_snapshot


class DummyLanguageDetector:
    def __init__(self, mapping: dict[str, tuple[str, float]] | None = None, default_lang: str = "vi"):
        self.mapping = mapping or {}
        self.default_lang = default_lang

    def detect(self, text: str) -> tuple[str, float]:
        for key, res in self.mapping.items():
            if key in text:
                return res
        return (self.default_lang, 0.95)


def _setup_multi_shard_snapshot(tmp_path: Path) -> Path:
    """Helper to create a snapshot with small shard size so documents/chunks span multiple shards."""
    db_path = tmp_path / "registry.db"
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry(db_path)

    sources = []
    for i in range(1, 7):
        sources.append(OfficialSource(doc_id=i, url=f"https://example.com/article_{i}"))
    ingest_sources(sources, registry)

    cursor = registry.conn.execute("SELECT job_id, url FROM fetch_jobs")
    for job_id, url in cursor.fetchall():
        doc_num = url.split("_")[-1]
        html_content = (
            f"<html><head><title>Article {doc_num}</title></head>"
            f"<body><article><h1>Heading {doc_num}</h1>"
            f"<p>Paragraph 1 for article {doc_num} contains important medical descriptions.</p>"
            f"<p>Paragraph 2 for article {doc_num} details dosage and contraindications.</p>"
            f"</article></body></html>"
        ).encode("utf-8")
        raw_path, raw_sha256, _ = save_raw_atomic(raw_root, html_content)
        registry.finish_fetch(
            FetchResult(
                job_id=job_id,
                state="fetched",
                raw_path=str(raw_path),
                raw_sha256=raw_sha256,
                final_url=url,
                mime_type="text/html",
                http_status=200,
                fetched_at="2026-10-05T00:00:00Z",
                error_code=None,
                bytes_written=len(html_content),
            )
        )

    # Use small target_shard_bytes and row_group_size to force multiple shards
    config = SnapshotConfig(
        snapshot_root=tmp_path / "snapshots",
        name="multi_shard_snap",
        parquet_row_group_size=2,
        target_shard_bytes=512,  # tiny size to rotate shards
        chunk_config=ChunkConfig(body_max_tokens=10),
    )
    build_snapshot(
        registry=registry,
        config=config,
        raw_root=raw_root,
        language_detector=DummyLanguageDetector(),
    )
    return tmp_path / "snapshots" / "multi_shard_snap"


def test_batched_get_documents_order_and_grouping(tmp_path: Path):
    """SnapshotSourceStore.get_documents retrieves documents in exact requested order and caches them."""
    snap_dir = _setup_multi_shard_snapshot(tmp_path)
    store = SnapshotSourceStore(snap_dir, cache_size=10)

    # Request documents out of order and with duplicate
    req_ids = [4, 1, 6, 1, 3]
    docs = store.get_documents(req_ids)

    assert len(docs) == len(req_ids)
    assert [d.doc_id for d in docs] == req_ids
    assert docs[0].doc_id == 4
    assert docs[1].doc_id == 1
    assert docs[2].doc_id == 6
    assert docs[3].doc_id == 1
    assert docs[4].doc_id == 3

    # Subsequent request should hit cache
    docs2 = store.get_documents([1, 4])
    assert [d.doc_id for d in docs2] == [1, 4]

    # Non-existent doc_id raises KeyError
    with pytest.raises(KeyError):
        store.get_documents([1, 9999])

    # Empty list returns empty list
    assert store.get_documents([]) == []
    store.close()


def test_batched_get_chunks_order_and_grouping(tmp_path: Path):
    """SnapshotSourceStore.get_chunks retrieves chunks in exact requested order and caches them."""
    snap_dir = _setup_multi_shard_snapshot(tmp_path)
    store = SnapshotSourceStore(snap_dir, cache_size=20)

    doc1 = store.get_document(1)
    doc2 = store.get_document(2)

    chunks1 = list(store.iter_chunks_for_doc(1))
    chunks2 = list(store.iter_chunks_for_doc(2))
    assert len(chunks1) > 0
    assert len(chunks2) > 0

    c1_id = chunks1[0].chunk_id
    c2_id = chunks2[0].chunk_id

    req_chunk_ids = [c2_id, c1_id, c2_id]
    chunks = store.get_chunks(req_chunk_ids)

    assert len(chunks) == len(req_chunk_ids)
    assert [c.chunk_id for c in chunks] == req_chunk_ids
    assert chunks[0].doc_id == 2
    assert chunks[1].doc_id == 1
    assert chunks[2].doc_id == 2

    # Non-existent chunk_id raises KeyError
    with pytest.raises(KeyError):
        store.get_chunks(["non_existent_chunk_id"])

    # Empty list returns empty list
    assert store.get_chunks([]) == []
    store.close()


def test_iter_chunks_for_doc_ordered(tmp_path: Path):
    """iter_chunks_for_doc yields chunks in strictly ascending chunk_index order."""
    snap_dir = _setup_multi_shard_snapshot(tmp_path)
    store = SnapshotSourceStore(snap_dir)

    chunks = list(store.iter_chunks_for_doc(1))
    assert len(chunks) >= 2
    indices = [c.chunk_index for c in chunks]
    assert indices == sorted(indices)
    assert indices == list(range(len(chunks)))

    for c in chunks:
        assert c.doc_id == 1
        assert c.text in store.get_document(1).source_text

    # Non-existent doc_id raises KeyError
    with pytest.raises(KeyError):
        list(store.iter_chunks_for_doc(8888))

    store.close()


def test_evidence_source_store_resolves_standard_and_evidence_spans(tmp_path: Path):
    """EvidenceSourceStore implements SourceStore and resolves standard chunks and custom EvidenceSpans."""
    snap_dir = _setup_multi_shard_snapshot(tmp_path)
    raw_store = SnapshotSourceStore(snap_dir)
    evidence_store = EvidenceSourceStore(raw_store)

    assert isinstance(evidence_store, SourceStore)
    assert evidence_store.contains_official_doc(1)
    assert not evidence_store.contains_official_doc(9999)

    doc1 = evidence_store.get_document(1)
    assert doc1.doc_id == 1

    # 1. Standard chunk lookup
    std_chunks = list(raw_store.iter_chunks_for_doc(1))
    std_chunk = std_chunks[0]
    resolved_std = evidence_store.get_chunk(std_chunk.chunk_id)
    assert resolved_std.chunk_id == std_chunk.chunk_id
    assert resolved_std.text == std_chunk.text

    # 2. Registered custom EvidenceSpan lookup
    # Slice first 30 chars of doc 1
    slice_text = doc1.source_text[5:35]
    custom_span = EvidenceSpan(
        span_id="custom_span_101",
        doc_id=1,
        source_version=doc1.source_version,
        source_sha256=doc1.source_sha256,
        start_char=5,
        end_char=35,
        seed_chunk_ids=(std_chunk.chunk_id,),
        variant_kind="base",
    )
    evidence_store.register_evidence_span(custom_span)

    resolved_span = evidence_store.get_chunk("custom_span_101")
    assert resolved_span.chunk_id == "custom_span_101"
    assert resolved_span.doc_id == 1
    assert resolved_span.start_char == 5
    assert resolved_span.end_char == 35
    assert resolved_span.text == slice_text

    # 3. Dynamic span: pattern lookup span:{doc_id}:{start}:{end}
    dynamic_span_id = "span:1:10:40"
    resolved_dyn = evidence_store.get_chunk(dynamic_span_id)
    assert resolved_dyn.chunk_id == dynamic_span_id
    assert resolved_dyn.doc_id == 1
    assert resolved_dyn.text == doc1.source_text[10:40]

    # 4. Batched get_chunks mixing standard and evidence span IDs
    batch_ids = [std_chunk.chunk_id, "custom_span_101", dynamic_span_id, std_chunk.chunk_id]
    batch_chunks = evidence_store.get_chunks(batch_ids)
    assert len(batch_chunks) == 4
    assert [c.chunk_id for c in batch_chunks] == batch_ids
    assert batch_chunks[0].text == std_chunk.text
    assert batch_chunks[1].text == slice_text
    assert batch_chunks[2].text == doc1.source_text[10:40]
    assert batch_chunks[3].text == std_chunk.text

    # 5. Batched get_documents delegation
    batch_docs = evidence_store.get_documents([1, 2])
    assert [d.doc_id for d in batch_docs] == [1, 2]

    # 6. iter_chunks_for_doc delegation
    iter_chunks = list(evidence_store.iter_chunks_for_doc(1))
    assert len(iter_chunks) == len(std_chunks)

    evidence_store.close()


def test_build_snapshot_preserves_usable_und_language(tmp_path: Path):
    """build_snapshot preserves articles detected as 'und' if usable rather than quarantining them."""
    db_path = tmp_path / "registry.db"
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry(db_path)

    sources = [
        OfficialSource(doc_id=1, url="https://example.com/vi_article"),
        OfficialSource(doc_id=2, url="https://example.com/und_article"),
    ]
    ingest_sources(sources, registry)

    # Job 1: Vietnamese
    # Job 2: Ambiguous text detected as "und" but legitimate medical content
    for doc_id, url, lang, text in [
        (1, "https://example.com/vi_article", "vi", "Bệnh viêm phổi do virus ở trẻ em."),
        (2, "https://example.com/und_article", "und", "12345 ABC XYZ Medical Formula and Code: Ph 7.4, Temp 37.0 C."),
    ]:
        html = f"<html><body><article><p>{text}</p></article></body></html>".encode("utf-8")
        raw_p, raw_hash, _ = save_raw_atomic(raw_root, html)
        cursor = registry.conn.execute("SELECT fetch_job_id FROM sources WHERE doc_id = ?", (doc_id,))
        job_id = cursor.fetchone()[0]
        registry.finish_fetch(
            FetchResult(
                job_id=job_id,
                state="fetched",
                raw_path=str(raw_p),
                raw_sha256=raw_hash,
                final_url=url,
                mime_type="text/html",
                http_status=200,
                fetched_at="2026-10-05T00:00:00Z",
                error_code=None,
                bytes_written=len(html),
            )
        )

    detector = DummyLanguageDetector(mapping={"Formula": ("und", 0.0)}, default_lang="vi")
    config = SnapshotConfig(snapshot_root=tmp_path / "snapshots", name="snap_und_test")
    manifest = build_snapshot(
        registry=registry,
        config=config,
        raw_root=raw_root,
        language_detector=detector,
    )

    assert manifest.ready_document_count == 2
    assert "und" in manifest.language_counts
    assert manifest.language_counts["und"] == 1

    store = SnapshotSourceStore(tmp_path / "snapshots" / "snap_und_test")
    doc_und = store.get_document(2)
    assert doc_und.language == "und"
    chunks_und = list(store.iter_chunks_for_doc(2))
    assert len(chunks_und) > 0
    assert chunks_und[0].language == "und"
    store.close()


def test_build_snapshot_handles_alias_doc_ids(tmp_path: Path):
    """When multiple official doc_ids map to the same URL/job, all get their own ExtractedDocument and chunks."""
    db_path = tmp_path / "registry.db"
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry(db_path)

    # doc 10 and doc 20 have the same URL
    sources = [
        OfficialSource(doc_id=10, url="https://example.com/shared_article"),
        OfficialSource(doc_id=20, url="https://example.com/shared_article"),
    ]
    ingest_sources(sources, registry)

    html = b"<html><body><article><p>Shared medical content for multiple IDs.</p></article></body></html>"
    raw_p, raw_hash, _ = save_raw_atomic(raw_root, html)
    cursor = registry.conn.execute("SELECT fetch_job_id FROM sources WHERE doc_id = 10")
    job_id = cursor.fetchone()[0]
    registry.finish_fetch(
        FetchResult(
            job_id=job_id,
            state="fetched",
            raw_path=str(raw_p),
            raw_sha256=raw_hash,
            final_url="https://example.com/shared_article",
            mime_type="text/html",
            http_status=200,
            fetched_at="2026-10-05T00:00:00Z",
            error_code=None,
            bytes_written=len(html),
        )
    )

    config = SnapshotConfig(snapshot_root=tmp_path / "snapshots", name="snap_alias_test")
    manifest = build_snapshot(
        registry=registry,
        config=config,
        raw_root=raw_root,
        language_detector=DummyLanguageDetector(),
    )

    assert manifest.ready_document_count == 2
    store = SnapshotSourceStore(tmp_path / "snapshots" / "snap_alias_test")
    d10 = store.get_document(10)
    d20 = store.get_document(20)
    assert d10.doc_id == 10
    assert d20.doc_id == 20
    assert d10.source_text == d20.source_text

    c10 = list(store.iter_chunks_for_doc(10))
    c20 = list(store.iter_chunks_for_doc(20))
    assert len(c10) > 0
    assert len(c20) > 0
    assert c10[0].doc_id == 10
    assert c20[0].doc_id == 20
    assert c10[0].chunk_id.startswith("10:")
    assert c20[0].chunk_id.startswith("20:")
    store.close()


def test_build_snapshot_records_terminal_gaps_and_scope(tmp_path: Path):
    """build_snapshot records terminal gaps to source_gaps.parquet and gaps.json, setting source_scope."""
    db_path = tmp_path / "registry.db"
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)
    registry = SourceRegistry(db_path)

    sources = [
        OfficialSource(doc_id=1, url="https://example.com/success"),
        OfficialSource(doc_id=2, url="https://example.com/failed_fetch"),
    ]
    ingest_sources(sources, registry)

    # Job 1 succeeds
    html = b"<html><body><article><p>Valid article content.</p></article></body></html>"
    raw_p, raw_hash, _ = save_raw_atomic(raw_root, html)
    cursor = registry.conn.execute("SELECT fetch_job_id FROM sources WHERE doc_id = 1")
    job1_id = cursor.fetchone()[0]
    registry.finish_fetch(
        FetchResult(
            job_id=job1_id,
            state="fetched",
            raw_path=str(raw_p),
            raw_sha256=raw_hash,
            final_url="https://example.com/success",
            mime_type="text/html",
            http_status=200,
            fetched_at="2026-10-05T00:00:00Z",
            error_code=None,
            bytes_written=len(html),
        )
    )

    # Job 2 fails terminally
    cursor = registry.conn.execute("SELECT fetch_job_id FROM sources WHERE doc_id = 2")
    job2_id = cursor.fetchone()[0]
    registry.finish_fetch(
        FetchResult(
            job_id=job2_id,
            state="failed",
            raw_path=None,
            raw_sha256=None,
            final_url="https://example.com/failed_fetch",
            mime_type=None,
            http_status=404,
            fetched_at="2026-10-05T00:00:00Z",
            error_code="http_404",
            bytes_written=0,
        )
    )

    config = SnapshotConfig(snapshot_root=tmp_path / "snapshots", name="snap_gaps_test")
    manifest = build_snapshot(
        registry=registry,
        config=config,
        raw_root=raw_root,
        language_detector=DummyLanguageDetector(),
    )

    assert manifest.source_count == 2
    assert manifest.ready_document_count == 1

    snap_dir = tmp_path / "snapshots" / "snap_gaps_test"
    gaps_json_file = snap_dir / "gaps.json"
    assert gaps_json_file.exists()
    gaps_data = json.loads(gaps_json_file.read_text(encoding="utf-8"))
    assert gaps_data["total_gaps"] == 1
    assert gaps_data["source_scope"] == SCOPE_FULL_SOURCE_PASS_WITH_GAPS

    # Verify manifest.json contains scope and gap metadata
    manifest_file = snap_dir / "manifest.json"
    m_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert m_data.get("source_scope") == SCOPE_FULL_SOURCE_PASS_WITH_GAPS
    assert m_data.get("gap_count") == 1

    # Verify source_gaps.parquet exists
    gaps_parquet = snap_dir / "source_gaps.parquet"
    assert gaps_parquet.exists()


def test_lru_cache_bounds_and_eviction():
    """LRUCache evicts least-recently-used items when max_entries or max_bytes is exceeded."""
    # 1. Capacity eviction by count
    cache = LRUCache[str, str](max_entries=2, max_bytes=10000)
    cache["a"] = "apple"
    cache["b"] = "banana"
    assert len(cache) == 2
    assert "a" in cache
    assert "b" in cache

    # Access "a" to make it most recently used
    _ = cache["a"]

    # Insert "c", which should evict "b"
    cache["c"] = "cherry"
    assert len(cache) == 2
    assert "a" in cache
    assert "c" in cache
    assert "b" not in cache

    # 2. Capacity eviction by byte size
    # size_fn returns length of string
    byte_cache = LRUCache[str, str](max_entries=100, max_bytes=10, size_fn=lambda s: len(s))
    byte_cache["k1"] = "12345"  # 5 bytes
    byte_cache["k2"] = "67890"  # 5 bytes -> total 10 bytes
    assert byte_cache.current_bytes == 10
    assert len(byte_cache) == 2

    # Insert k3 with 4 bytes -> total 14 > 10, evicts k1
    byte_cache["k3"] = "abcd"
    assert "k1" not in byte_cache
    assert "k2" in byte_cache
    assert "k3" in byte_cache
    assert byte_cache.current_bytes == 9  # 5 + 4
