import gzip
import hashlib
import json
from pathlib import Path
import pytest

from src.data.dataset_adapter import DatasetAdapter
from src.data.schemas import ChunkRecord, DocumentRecord, QueryRecord
from src.data.source_store import SnapshotSourceStore
from src.data.vibiomir_adapter import ViBioMIRAdapter, validate_index_manifest
from src.data.vibiomir_contracts import (
    ChunkConfig,
    ExtractionConfig,
    ExtractedDocument,
    OfficialQuery,
    OfficialSource,
    SnapshotConfig,
    SnapshotManifest,
    SourceStore,
    SpanChunk,
)
from src.data.vibiomir_registry import SourceRegistry, ingest_sources
from src.indexing.dense_index import DenseIndexer
from src.ingestion.fetcher import save_raw_atomic
from src.ingestion.snapshot import build_snapshot


class DummyLanguageDetector:
    def __init__(self, language: str = "vi", score: float = 0.95):
        self.language = language
        self.score = score

    def detect(self, text: str) -> tuple[str, float]:
        if not text.strip():
            return ("und", 0.0)
        return (self.language, self.score)


class OnePassIterator:
    """An iterator that can strictly be iterated only once."""

    def __init__(self, items):
        self.items = list(items)
        self.iterated = False

    def __iter__(self):
        if self.iterated:
            raise RuntimeError("OnePassIterator cannot be iterated more than once!")
        self.iterated = True
        return iter(self.items)


def create_sample_registry_and_raw(
    tmp_path: Path, num_sources: int = 5, same_url_pair: bool = False, uncrawled_ids: list[int] | None = None
) -> tuple[SourceRegistry, Path, list[int]]:
    """Helper to set up a registry with fetched raw HTML files and optional pending uncrawled sources."""
    db_path = tmp_path / "registry.db"
    raw_root = tmp_path / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    registry = SourceRegistry(db_path)
    sources = []
    doc_ids = []

    for i in range(1, num_sources + 1):
        doc_ids.append(i)
        if same_url_pair and i in (1, 2):
            url = "https://example.com/same_content"
        else:
            url = f"https://example.com/page_{i}"
        sources.append(OfficialSource(doc_id=i, url=url))

    if uncrawled_ids:
        for u_id in uncrawled_ids:
            doc_ids.append(u_id)
            sources.append(OfficialSource(doc_id=u_id, url=f"https://example.com/uncrawled_{u_id}"))

    ingest_sources(sources, registry)

    # For each job belonging to crawled sources, simulate a successful fetch with raw HTML
    cursor = registry.conn.execute("SELECT job_id, url FROM fetch_jobs WHERE url NOT LIKE '%uncrawled%'")
    for job_id, url in cursor.fetchall():
        html_content = (
            f"<html><head><title>Title for {url}</title></head>"
            f"<body><article><h1>Heading for {url}</h1>"
            f"<p>This is paragraph 1 with medical information for {url}. Triệu chứng bệnh viêm phổi.</p>"
            f"<p>This is paragraph 2 with detailed guidance for {url}. Điều trị kháng sinh cần tuân thủ phác đồ.</p>"
            f"</article></body></html>"
        ).encode("utf-8")

        raw_path, raw_sha256, _ = save_raw_atomic(raw_root, html_content)
        registry.finish_fetch(
            type("FetchResultObj", (), {
                "job_id": job_id,
                "state": "fetched",
                "raw_path": raw_path,
                "raw_sha256": raw_sha256,
                "final_url": url,
                "mime_type": "text/html",
                "http_status": 200,
                "error_code": None,
                "retry_at": None,
            })()
        )

    return registry, raw_root, doc_ids


def test_shard_rotation_and_atomic_publication(tmp_path: Path):
    """Test that shards are rotated based on target bytes/rows, written atomically, and manifest published last."""
    registry, raw_root, _ = create_sample_registry_and_raw(tmp_path, num_sources=10)

    snapshot_dir = tmp_path / "snapshots"
    # Set small row group and small target shard bytes to force rotation
    config = SnapshotConfig(
        snapshot_root=snapshot_dir,
        name="test_snapshot_rotation",
        parquet_row_group_size=2,
        target_shard_bytes=500,  # small enough to trigger rotation
        chunk_config=ChunkConfig(body_max_tokens=50),
    )

    manifest = build_snapshot(
        registry=registry,
        config=config,
        raw_root=raw_root,
        language_detector=DummyLanguageDetector(),
    )

    target_dir = snapshot_dir / "test_snapshot_rotation"
    assert target_dir.exists()

    # Manifest file must exist
    manifest_file = target_dir / "manifest.json"
    assert manifest_file.exists()

    # No leftover .tmp files
    tmp_files = list(target_dir.glob("*.tmp*"))
    assert len(tmp_files) == 0, f"Found uncommitted tmp files: {tmp_files}"

    # Shards should have rotated (more than 1 shard for docs or chunks)
    doc_shards = list(target_dir.glob("documents_*.parquet"))
    chunk_shards = list(target_dir.glob("chunks_*.parquet"))
    assert len(doc_shards) >= 2, f"Expected multiple document shards, got {doc_shards}"
    assert len(chunk_shards) >= 2, f"Expected multiple chunk shards, got {chunk_shards}"

    # Verify manifest fields
    assert manifest.source_count == 10
    assert manifest.ready_document_count == 10
    assert manifest.chunk_count > 0
    assert len(manifest.shard_paths_and_hashes) == len(doc_shards) + len(chunk_shards)

    for shard_info in manifest.shard_paths_and_hashes:
        shard_path = target_dir / shard_info["path"]
        assert shard_path.exists()
        # Verify hash matches actual file hash
        actual_hash = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        assert actual_hash == shard_info["sha256"]


def test_atomic_publication_failure_does_not_publish_manifest(tmp_path: Path, monkeypatch):
    """If an error occurs before completing shards, manifest.json must NOT be published."""
    registry, raw_root, _ = create_sample_registry_and_raw(tmp_path, num_sources=4)

    snapshot_dir = tmp_path / "snapshots"
    config = SnapshotConfig(
        snapshot_root=snapshot_dir,
        name="test_snapshot_failed",
        parquet_row_group_size=2,
    )

    # Monkeypatch extraction to fail mid-process
    import src.ingestion.snapshot as snap_mod

    def broken_extract(*args, **kwargs):
        raise RuntimeError("Simulated extraction crash during snapshot build")

    monkeypatch.setattr(snap_mod, "extract_document", broken_extract)

    with pytest.raises(RuntimeError, match="Simulated extraction crash"):
        build_snapshot(
            registry=registry,
            config=config,
            raw_root=raw_root,
            language_detector=DummyLanguageDetector(),
        )

    target_dir = snapshot_dir / "test_snapshot_failed"
    manifest_file = target_dir / "manifest.json"
    assert not manifest_file.exists(), "manifest.json should not exist if snapshot build failed"


def test_snapshot_immutability_invariant(tmp_path: Path):
    """Invariant: Snapshot A does not change when new crawl fetch jobs occur and Snapshot B is built."""
    registry, raw_root, _ = create_sample_registry_and_raw(tmp_path, num_sources=3, uncrawled_ids=[4, 5])

    config_a = SnapshotConfig(
        snapshot_root=tmp_path / "snapshots",
        name="snapshot_a",
    )
    manifest_a = build_snapshot(
        registry=registry,
        config=config_a,
        raw_root=raw_root,
        language_detector=DummyLanguageDetector(),
    )

    dir_a = tmp_path / "snapshots" / "snapshot_a"
    files_a_before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dir_a.iterdir()}

    # Now crawl the pending sources 4 and 5
    cursor = registry.conn.execute("SELECT job_id, url FROM fetch_jobs WHERE state = 'pending'")
    for job_id, url in cursor.fetchall():
        html_bytes = f"<html><body><p>Content for new url {url}</p></body></html>".encode("utf-8")
        raw_path, raw_sha256, _ = save_raw_atomic(raw_root, html_bytes)
        registry.finish_fetch(
            type("FetchResultObj", (), {
                "job_id": job_id,
                "state": "fetched",
                "raw_path": raw_path,
                "raw_sha256": raw_sha256,
                "final_url": url,
                "mime_type": "text/html",
                "http_status": 200,
                "error_code": None,
                "retry_at": None,
            })()
        )

    # Build Snapshot B
    config_b = SnapshotConfig(
        snapshot_root=tmp_path / "snapshots",
        name="snapshot_b",
    )
    manifest_b = build_snapshot(
        registry=registry,
        config=config_b,
        raw_root=raw_root,
        language_detector=DummyLanguageDetector(),
    )

    # Snapshot B has 5 documents
    assert manifest_b.ready_document_count == 5

    # Check Snapshot A directory files: 100% bit-for-bit unchanged
    files_a_after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in dir_a.iterdir()}
    assert files_a_before == files_a_after
    assert manifest_a.ready_document_count == 3


def test_same_url_keeps_two_distinct_official_doc_ids(tmp_path: Path):
    """Same URL keeps two distinct official doc IDs in snapshot documents and chunks."""
    registry, raw_root, _ = create_sample_registry_and_raw(tmp_path, num_sources=3, same_url_pair=True)

    config = SnapshotConfig(
        snapshot_root=tmp_path / "snapshots",
        name="snapshot_same_url",
    )
    manifest = build_snapshot(
        registry=registry,
        config=config,
        raw_root=raw_root,
        language_detector=DummyLanguageDetector(),
    )

    store = SnapshotSourceStore(tmp_path / "snapshots" / "snapshot_same_url")

    # Both doc_id 1 and 2 exist in store
    assert store.contains_official_doc(1)
    assert store.contains_official_doc(2)

    doc1 = store.get_document(1)
    doc2 = store.get_document(2)

    assert doc1.doc_id == 1
    assert doc2.doc_id == 2
    # Both have the same text because they share URL and raw content
    assert doc1.source_text == doc2.source_text
    assert doc1.source_sha256 == doc2.source_sha256

    # Adapter yields both distinct DocumentRecords
    adapter = ViBioMIRAdapter(tmp_path / "snapshots" / "snapshot_same_url")
    docs = list(adapter.iter_documents())
    doc_ids_in_adapter = {d.doc_id for d in docs}
    assert "1" in doc_ids_in_adapter
    assert "2" in doc_ids_in_adapter

    # Chunks for both doc IDs exist with distinct chunk IDs
    chunks = list(adapter.iter_chunks())
    chunk_doc_ids = {c.doc_id for c in chunks}
    assert "1" in chunk_doc_ids
    assert "2" in chunk_doc_ids
    assert any(c.chunk_id.startswith("1:") for c in chunks)
    assert any(c.chunk_id.startswith("2:") for c in chunks)


def test_snapshot_source_store_lookups(tmp_path: Path):
    """SnapshotSourceStore retrieves exact ExtractedDocument and SpanChunk by ID, and checks official IDs."""
    registry, raw_root, _ = create_sample_registry_and_raw(tmp_path, num_sources=3, uncrawled_ids=[99])

    config = SnapshotConfig(
        snapshot_root=tmp_path / "snapshots",
        name="snapshot_store_test",
    )
    manifest = build_snapshot(
        registry=registry,
        config=config,
        raw_root=raw_root,
        language_detector=DummyLanguageDetector(),
    )

    store = SnapshotSourceStore(tmp_path / "snapshots" / "snapshot_store_test")
    assert isinstance(store, SourceStore)

    # Crawled docs are present
    assert store.contains_official_doc(1)
    assert store.contains_official_doc(2)
    assert store.contains_official_doc(3)
    # Uncrawled doc is in registry, so contains_official_doc is True
    assert store.contains_official_doc(99)
    # Non-existent doc is False
    assert not store.contains_official_doc(9999)

    doc1 = store.get_document(1)
    assert isinstance(doc1, ExtractedDocument)
    assert doc1.doc_id == 1
    assert "Triệu chứng bệnh viêm phổi" in doc1.source_text
    assert len(doc1.blocks) > 0

    # Uncrawled doc raises KeyError on get_document
    with pytest.raises(KeyError, match="not found in snapshot"):
        store.get_document(99)

    with pytest.raises(KeyError, match="not found in snapshot"):
        store.get_document(9999)

    # Test chunk retrieval
    adapter = ViBioMIRAdapter(tmp_path / "snapshots" / "snapshot_store_test")
    chunks = list(adapter.iter_chunks())
    assert len(chunks) > 0
    sample_chunk_id = chunks[0].chunk_id

    chunk_retrieved = store.get_chunk(sample_chunk_id)
    assert isinstance(chunk_retrieved, SpanChunk)
    assert chunk_retrieved.chunk_id == sample_chunk_id
    assert chunk_retrieved.text == chunks[0].text

    with pytest.raises(KeyError, match="Chunk .* not found"):
        store.get_chunk("nonexistent:sha:000000")


def test_vibiomir_adapter_protocol_and_id_string_conversion(tmp_path: Path):
    """ViBioMIRAdapter converts official int IDs (583) to core strings ('583') and implements DatasetAdapter."""
    registry, raw_root, _ = create_sample_registry_and_raw(tmp_path, num_sources=3)

    config = SnapshotConfig(
        snapshot_root=tmp_path / "snapshots",
        name="snapshot_adapter_test",
    )
    manifest = build_snapshot(
        registry=registry,
        config=config,
        raw_root=raw_root,
        language_detector=DummyLanguageDetector(),
    )

    snapshot_path = tmp_path / "snapshots" / "snapshot_adapter_test"
    adapter = ViBioMIRAdapter(snapshot_path)
    assert isinstance(adapter, DatasetAdapter)

    # Test streaming documents
    doc_iter = adapter.iter_documents()
    first_doc = next(doc_iter)
    assert isinstance(first_doc, DocumentRecord)
    assert isinstance(first_doc.doc_id, str)
    assert first_doc.doc_id == "1"
    assert first_doc.language in ("vi", "en", "zh")
    assert first_doc.metadata is not None
    assert first_doc.metadata["official_doc_id"] == 1

    # Test streaming chunks
    chunk_iter = adapter.iter_chunks()
    first_chunk = next(chunk_iter)
    assert isinstance(first_chunk, ChunkRecord)
    assert isinstance(first_chunk.doc_id, str)
    assert first_chunk.doc_id == "1"
    assert isinstance(first_chunk.chunk_id, str)
    assert first_chunk.metadata is not None
    assert first_chunk.metadata["official_doc_id"] == 1
    assert "start_char" in first_chunk.metadata

    # Test load_documents and load_chunks compatibility methods
    all_docs = adapter.load_documents()
    all_chunks = adapter.load_chunks()
    assert len(all_docs) == 3
    assert len(all_chunks) > 0

    # Test load_queries with query parquet
    queries_file = tmp_path / "query.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq

    q_table = pa.table({
        "id": pa.array([101, 102], type=pa.int64()),
        "query": pa.array(["Triệu chứng viêm phổi?", "Cách điều trị kháng sinh?"], type=pa.string()),
    })
    pq.write_table(q_table, queries_file)

    adapter_with_queries = ViBioMIRAdapter(snapshot_path, queries_path=queries_file)
    queries = adapter_with_queries.load_queries("test")
    assert len(queries) == 2
    assert isinstance(queries[0], QueryRecord)
    assert queries[0].query_id == "101"
    assert queries[0].text_vi == "Triệu chứng viêm phổi?"


def test_one_pass_stream_consumption(tmp_path: Path):
    """Embedding/indexing path consumes chunks in one pass without loading full corpus into a single list."""
    chunks = [
        ChunkRecord(chunk_id="c1", doc_id="1", language="vi", text="đoạn văn bản y tế thứ nhất"),
        ChunkRecord(chunk_id="c2", doc_id="2", language="vi", text="đoạn văn bản y tế thứ hai"),
        ChunkRecord(chunk_id="c3", doc_id="3", language="vi", text="đoạn văn bản y tế thứ ba"),
    ]

    one_pass_stream = OnePassIterator(chunks)

    class MockEncoder:
        def encode(self, texts, **kwargs):
            import numpy as np
            # Verify batch of texts is small, not entire corpus at once if batched
            return np.ones((len(texts), 32), dtype=np.float32)

    indexer = DenseIndexer(encoder=MockEncoder(), backend="numpy", batch_size=2)
    output_dir = tmp_path / "dense_out"

    # Must succeed with one-pass stream without raising RuntimeError
    manifest = indexer.build_index(one_pass_stream, output_dir=output_dir)
    assert manifest["backend"] == "numpy"
    assert manifest["partitions"]["vi"]["count"] == 3
    assert "resource_estimate" in manifest
    assert manifest["resource_estimate"]["vector_count"] == 3


def test_validate_index_manifest(tmp_path: Path):
    """validate_index_manifest verifies snapshot_id and fingerprints match snapshot manifest."""
    snapshot_manifest = SnapshotManifest(
        snapshot_id="snap_20261004_v1",
        created_at="2026-10-04T12:00:00Z",
        registry_sha256="abc123reg",
        queries_sha256="q123sha",
        source_membership_sha256="membersha999",
        extractor_fingerprint="ext_trafilatura_v1",
        chunker_fingerprint="chunk_strat_v1",
        source_count=100,
        ready_document_count=95,
        chunk_count=350,
        language_counts={"vi": 95},
        quarantine_counts={},
        shard_paths_and_hashes=[],
    )

    # Valid index manifest
    idx_dir = tmp_path / "valid_index"
    idx_dir.mkdir()
    valid_manifest_content = {
        "snapshot_id": "snap_20261004_v1",
        "source_membership_sha256": "membersha999",
        "chunker_fingerprint": "chunk_strat_v1",
        "backend": "faiss",
    }
    with open(idx_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(valid_manifest_content, f)

    # Should validate cleanly
    validate_index_manifest(idx_dir, snapshot_manifest)

    # Mismatched snapshot_id
    bad_id_dir = tmp_path / "bad_id_index"
    bad_id_dir.mkdir()
    with open(bad_id_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({**valid_manifest_content, "snapshot_id": "different_snap_id"}, f)

    with pytest.raises(ValueError, match="Snapshot ID mismatch"):
        validate_index_manifest(bad_id_dir, snapshot_manifest)

    # Mismatched source_membership_sha256
    bad_membership_dir = tmp_path / "bad_mem_index"
    bad_membership_dir.mkdir()
    with open(bad_membership_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({**valid_manifest_content, "source_membership_sha256": "corrupted_membership"}, f)

    with pytest.raises(ValueError, match="Source membership SHA-256 mismatch"):
        validate_index_manifest(bad_membership_dir, snapshot_manifest)

    # Mismatched chunker_fingerprint
    bad_chunker_dir = tmp_path / "bad_chunker_index"
    bad_chunker_dir.mkdir()
    with open(bad_chunker_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({**valid_manifest_content, "chunker_fingerprint": "other_chunker"}, f)

    with pytest.raises(ValueError, match="Chunker fingerprint mismatch"):
        validate_index_manifest(bad_chunker_dir, snapshot_manifest)

    # Missing snapshot_id entirely
    missing_id_dir = tmp_path / "missing_id_index"
    missing_id_dir.mkdir()
    with open(missing_id_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"backend": "faiss"}, f)

    with pytest.raises(ValueError, match="missing required field 'snapshot_id'"):
        validate_index_manifest(missing_id_dir, snapshot_manifest)
