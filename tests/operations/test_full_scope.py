from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.data.phase2_contracts import (
    CoverageLedger,
    FullCampaignConfig,
    IndexManifest,
    ScopeViolationError,
)
from src.data.vibiomir_contracts import SnapshotManifest
from src.data.vibiomir_registry import SourceRegistry
from src.operations.coverage import (
    build_coverage_ledger,
    require_scope,
)


def _compute_membership_sha(doc_ids: list[int]) -> str:
    membership_bytes = "".join(f"{d}\n" for d in sorted(doc_ids)).encode("utf-8")
    return hashlib.sha256(membership_bytes).hexdigest()


def _make_snapshot(
    snapshot_id: str = "snap_phase2_01",
    ready_doc_ids: list[int] | None = None,
    chunk_count: int = 200,
    quarantine_counts: dict[str, int] | None = None,
    registry_sha256: str = "reg_sha_canonical",
) -> SnapshotManifest:
    if ready_doc_ids is None:
        ready_doc_ids = list(range(1, 101))
    membership_sha = _compute_membership_sha(ready_doc_ids)
    return SnapshotManifest(
        snapshot_id=snapshot_id,
        created_at="2026-10-05T00:00:00Z",
        registry_sha256=registry_sha256,
        queries_sha256="queries_sha",
        source_membership_sha256=membership_sha,
        extractor_fingerprint="extractor_fp",
        chunker_fingerprint="chunker_fp",
        source_count=len(ready_doc_ids),
        ready_document_count=len(ready_doc_ids),
        chunk_count=chunk_count,
        language_counts={"vi": len(ready_doc_ids)},
        quarantine_counts=quarantine_counts or {},
        shard_paths_and_hashes=[],
    )


def _make_index_manifest(
    snapshot_id: str = "snap_phase2_01",
    item_type: str = "document",
    member_count: int = 100,
    membership_sha256: str = "",
    backend: str = "lucene",
    language: str = "all",
) -> IndexManifest:
    return IndexManifest(
        snapshot_id=snapshot_id,
        item_type=item_type,
        language=language,
        member_count=member_count,
        membership_sha256=membership_sha256,
        backend=backend,
        model_fingerprint=None if backend == "lucene" else "intfloat/multilingual-e5-base",
        tokenizer_fingerprint="whitespace",
        parameters={"k1": 1.2, "b": 0.75},
        artifact_paths_and_hashes={},
    )


def _populate_registry(
    db_path: Path,
    sources: list[tuple[int, str, str]],
    registry_hash: str = "reg_sha_canonical",
) -> SourceRegistry:
    """Populates SourceRegistry with doc_id, url, and fetch_job state."""
    registry = SourceRegistry(db_path)
    registry.init_ingest_state(registry_hash)

    for doc_id, url, state in sources:
        job_id = hashlib.sha256(url.encode("utf-8")).hexdigest()
        with registry.conn:
            registry.conn.execute(
                "INSERT INTO sources (doc_id, original_url, fetch_job_id) VALUES (?, ?, ?)",
                (doc_id, url, job_id),
            )
            registry.conn.execute(
                """
                INSERT OR REPLACE INTO fetch_jobs (
                    job_id, url, state, attempts, created_at, updated_at
                ) VALUES (?, ?, ?, 1, '2026-10-05T00:00:00Z', '2026-10-05T00:00:00Z')
                """,
                (job_id, url, state),
            )
    registry.mark_ingest_complete(registry_hash, len(sources))
    return registry


# ---------------------------------------------------------------------------
# Test 1: Full registry (4.39M) + pilot (465 bodies) labels PARTIAL_BODY_INDEX
# ---------------------------------------------------------------------------
def test_full_registry_plus_pilot_bodies_labels_partial(tmp_path: Path):
    """A registry with 4.39M official sources but only 465 ready bodies MUST label PARTIAL_BODY_INDEX."""
    # Using mock registry for 4.39M scale
    mock_registry = MagicMock(spec=SourceRegistry)
    mock_registry.conn = None
    mock_registry.count_sources.return_value = 4394718
    mock_registry.count_unique_urls.return_value = 3980000
    mock_registry.get_ingest_state.return_value = ("reg_sha_canonical", 4394718, True)

    pilot_ids = list(range(1, 466))
    pilot_snapshot = _make_snapshot(
        snapshot_id="pilot_snap",
        ready_doc_ids=pilot_ids,
        chunk_count=930,
    )
    pilot_doc_index = _make_index_manifest(
        snapshot_id="pilot_snap",
        item_type="document",
        member_count=465,
        membership_sha256=_compute_membership_sha(pilot_ids),
    )
    pilot_chunk_index = _make_index_manifest(
        snapshot_id="pilot_snap",
        item_type="chunk",
        member_count=930,
        membership_sha256="chunk_hash",
    )

    ledger = build_coverage_ledger(
        mock_registry,
        pilot_snapshot,
        [pilot_doc_index, pilot_chunk_index],
    )

    assert ledger.official_count == 4394718
    assert ledger.unique_url_count == 3980000
    assert ledger.ready_doc_count == 465
    assert ledger.unresolved_count == 4394718 - 465
    assert ledger.terminal_gap_count == 0
    assert ledger.source_scope == "PARTIAL_BODY_INDEX"

    # Strict scope guard: requiring full source pass or full body MUST raise
    with pytest.raises(ScopeViolationError, match="PARTIAL_BODY_INDEX"):
        require_scope(ledger, "FULL_SOURCE_PASS_WITH_GAPS")

    with pytest.raises(ScopeViolationError, match="PARTIAL_BODY_INDEX"):
        require_scope(ledger, "full_source_pass")

    with pytest.raises(ScopeViolationError, match="PARTIAL_BODY_INDEX"):
        require_scope(ledger, "FULL_BODY_INDEX")


# ---------------------------------------------------------------------------
# Test 2: Pending doc blocks full source pass
# ---------------------------------------------------------------------------
def test_pending_doc_blocks_full_source_pass(tmp_path: Path):
    """Even a single pending or in-flight doc blocks full source pass."""
    db_path = tmp_path / "test_reg.db"
    # 10 sources: 9 fetched, 1 pending
    sources = [(i, f"https://example.com/doc_{i}", "fetched") for i in range(1, 10)]
    sources.append((10, "https://example.com/doc_10", "pending"))
    registry = _populate_registry(db_path, sources)

    ready_ids = list(range(1, 10))
    snapshot = _make_snapshot(
        ready_doc_ids=ready_ids,
        chunk_count=20,
    )
    doc_index = _make_index_manifest(
        item_type="document",
        member_count=9,
        membership_sha256=_compute_membership_sha(ready_ids),
    )
    chunk_index = _make_index_manifest(
        item_type="chunk",
        member_count=20,
    )

    ledger = build_coverage_ledger(registry, snapshot, [doc_index, chunk_index])

    assert ledger.official_count == 10
    assert ledger.ready_doc_count == 9
    assert ledger.unresolved_count == 1
    assert ledger.terminal_gap_count == 0
    assert ledger.source_scope == "PARTIAL_BODY_INDEX"

    with pytest.raises(ScopeViolationError, match="Scope violation"):
        require_scope(ledger, "FULL_SOURCE_PASS_WITH_GAPS")


# ---------------------------------------------------------------------------
# Test 3: Terminal failures reported properly (FULL_SOURCE_PASS_WITH_GAPS)
# ---------------------------------------------------------------------------
def test_terminal_failures_reported_properly(tmp_path: Path):
    """When all sources are resolved (some ready, some terminal failures), labels FULL_SOURCE_PASS_WITH_GAPS."""
    db_path = tmp_path / "test_reg.db"
    # 100 sources: 80 fetched, 15 fetch_failed, 5 blocked. No pending.
    sources = []
    for i in range(1, 81):
        sources.append((i, f"https://example.com/doc_{i}", "fetched"))
    for i in range(81, 96):
        sources.append((i, f"https://example.com/doc_{i}", "fetch_failed"))
    for i in range(96, 101):
        sources.append((i, f"https://example.com/doc_{i}", "blocked"))

    registry = _populate_registry(db_path, sources)

    ready_ids = list(range(1, 81))
    snapshot = _make_snapshot(
        ready_doc_ids=ready_ids,
        chunk_count=160,
    )
    doc_index = _make_index_manifest(
        item_type="document",
        member_count=80,
        membership_sha256=_compute_membership_sha(ready_ids),
    )
    chunk_index = _make_index_manifest(
        item_type="chunk",
        member_count=160,
    )

    ledger = build_coverage_ledger(registry, snapshot, [doc_index, chunk_index])

    assert ledger.official_count == 100
    assert ledger.ready_doc_count == 80
    assert ledger.terminal_gap_count == 20
    assert ledger.unresolved_count == 0
    # Invariant: N = ready + terminal + unresolved
    assert ledger.official_count == ledger.ready_doc_count + ledger.terminal_gap_count + ledger.unresolved_count
    assert ledger.source_scope == "FULL_SOURCE_PASS_WITH_GAPS"

    # Satisfies full source pass
    require_scope(ledger, "FULL_SOURCE_PASS_WITH_GAPS")
    require_scope(ledger, "full_source_pass")

    # Does NOT satisfy full body index (because terminal gaps exist)
    with pytest.raises(ScopeViolationError, match="terminal gaps"):
        require_scope(ledger, "FULL_BODY_INDEX")


# ---------------------------------------------------------------------------
# Test 4: Ready doc or chunk omitted from index blocks validation
# ---------------------------------------------------------------------------
def test_ready_doc_or_chunk_omitted_blocks_validation(tmp_path: Path):
    """Ready docs or chunks omitted from sparse indexes MUST force PARTIAL_BODY_INDEX."""
    db_path = tmp_path / "test_reg.db"
    sources = [(i, f"https://example.com/doc_{i}", "fetched") for i in range(1, 51)]
    registry = _populate_registry(db_path, sources)

    ready_ids = list(range(1, 51))
    snapshot = _make_snapshot(ready_doc_ids=ready_ids, chunk_count=100)

    # Case A: Document index only has 49 out of 50 docs
    incomplete_doc_ids = ready_ids[:49]
    incomplete_doc_index = _make_index_manifest(
        item_type="document",
        member_count=49,
        membership_sha256=_compute_membership_sha(incomplete_doc_ids),
    )
    full_chunk_index = _make_index_manifest(item_type="chunk", member_count=100)

    ledger_a = build_coverage_ledger(
        registry, snapshot, [incomplete_doc_index, full_chunk_index]
    )
    assert ledger_a.sparse_doc_count == 49
    assert ledger_a.ready_doc_count == 50
    assert ledger_a.source_scope == "PARTIAL_BODY_INDEX"
    with pytest.raises(ScopeViolationError):
        require_scope(ledger_a, "FULL_SOURCE_PASS_WITH_GAPS")

    # Case B: Doc index is complete (50 docs), but chunk index has 95 out of 100 chunks
    full_doc_index = _make_index_manifest(
        item_type="document",
        member_count=50,
        membership_sha256=_compute_membership_sha(ready_ids),
    )
    incomplete_chunk_index = _make_index_manifest(item_type="chunk", member_count=95)

    ledger_b = build_coverage_ledger(
        registry, snapshot, [full_doc_index, incomplete_chunk_index]
    )
    assert ledger_b.sparse_chunk_count == 95
    assert ledger_b.chunk_count == 100
    assert ledger_b.source_scope == "PARTIAL_BODY_INDEX"
    with pytest.raises(ScopeViolationError):
        require_scope(ledger_b, "FULL_SOURCE_PASS_WITH_GAPS")


# ---------------------------------------------------------------------------
# Test 5: Equal count with wrong membership IDs fails verification
# ---------------------------------------------------------------------------
def test_equal_count_wrong_membership_rejects(tmp_path: Path):
    """An index with equal count (e.g. 50) but different membership hash MUST be rejected."""
    db_path = tmp_path / "test_reg.db"
    sources = [(i, f"https://example.com/doc_{i}", "fetched") for i in range(1, 51)]
    registry = _populate_registry(db_path, sources)

    ready_ids = list(range(1, 51))
    snapshot = _make_snapshot(ready_doc_ids=ready_ids, chunk_count=100)

    # Mismatched IDs (51..100) having the exact same count (50)
    fraudulent_ids = list(range(51, 101))
    fraudulent_hash = _compute_membership_sha(fraudulent_ids)

    fraudulent_doc_index = _make_index_manifest(
        item_type="document",
        member_count=50,
        membership_sha256=fraudulent_hash,  # Count is 50, but hash doesn't match snapshot!
    )
    chunk_index = _make_index_manifest(item_type="chunk", member_count=100)

    with pytest.raises(ScopeViolationError, match="membership hash mismatch"):
        build_coverage_ledger(
            registry, snapshot, [fraudulent_doc_index, chunk_index]
        )


# ---------------------------------------------------------------------------
# Test 6: Snapshot ID mismatch in index is rejected
# ---------------------------------------------------------------------------
def test_snapshot_id_mismatch_rejects(tmp_path: Path):
    """An index built for a different snapshot_id MUST be rejected."""
    db_path = tmp_path / "test_reg.db"
    sources = [(i, f"https://example.com/doc_{i}", "fetched") for i in range(1, 11)]
    registry = _populate_registry(db_path, sources)

    snapshot = _make_snapshot(snapshot_id="snap_A", ready_doc_ids=list(range(1, 11)))
    doc_index = _make_index_manifest(snapshot_id="snap_B", member_count=10)

    with pytest.raises(ScopeViolationError, match="snapshot_id mismatch"):
        build_coverage_ledger(registry, snapshot, [doc_index])


# ---------------------------------------------------------------------------
# Test 7: Distinguish URL rows from unique URLs
# ---------------------------------------------------------------------------
def test_url_rows_vs_unique_urls_distinction(tmp_path: Path):
    """Distinguishes between official URL rows (N) and unique URLs (U)."""
    db_path = tmp_path / "test_reg.db"
    # 5 docs sharing 2 URLs (aliases)
    sources = [
        (1, "https://example.com/pageA", "fetched"),
        (2, "https://example.com/pageA", "fetched"),
        (3, "https://example.com/pageA", "fetched"),
        (4, "https://example.com/pageB", "fetched"),
        (5, "https://example.com/pageB", "fetched"),
    ]
    registry = _populate_registry(db_path, sources)

    ready_ids = [1, 2, 3, 4, 5]
    snapshot = _make_snapshot(ready_doc_ids=ready_ids, chunk_count=10)
    doc_index = _make_index_manifest(
        member_count=5,
        membership_sha256=_compute_membership_sha(ready_ids),
    )
    chunk_index = _make_index_manifest(item_type="chunk", member_count=10)

    ledger = build_coverage_ledger(registry, snapshot, [doc_index, chunk_index])

    assert ledger.official_count == 5
    assert ledger.unique_url_count == 2


# ---------------------------------------------------------------------------
# Test 8: Source limit or domain whitelist forces PARTIAL_BODY_INDEX
# ---------------------------------------------------------------------------
def test_campaign_filters_force_partial(tmp_path: Path):
    """Configuring source_limit or domain_whitelist automatically sets PARTIAL_BODY_INDEX."""
    db_path = tmp_path / "test_reg.db"
    sources = [(i, f"https://example.com/doc_{i}", "fetched") for i in range(1, 21)]
    registry = _populate_registry(db_path, sources)

    ready_ids = list(range(1, 21))
    snapshot = _make_snapshot(ready_doc_ids=ready_ids, chunk_count=40)
    doc_index = _make_index_manifest(
        member_count=20,
        membership_sha256=_compute_membership_sha(ready_ids),
    )
    chunk_index = _make_index_manifest(item_type="chunk", member_count=40)

    # With source_limit set
    cfg_limit = FullCampaignConfig(source_limit=10)
    ledger_limit = build_coverage_ledger(
        registry, snapshot, [doc_index, chunk_index], campaign_config=cfg_limit
    )
    assert ledger_limit.source_scope == "PARTIAL_BODY_INDEX"

    # With domain_whitelist set
    cfg_whitelist = FullCampaignConfig(domain_whitelist=["example.com"])
    ledger_whitelist = build_coverage_ledger(
        registry, snapshot, [doc_index, chunk_index], campaign_config=cfg_whitelist
    )
    assert ledger_whitelist.source_scope == "PARTIAL_BODY_INDEX"


# ---------------------------------------------------------------------------
# Test 9: FULL_BODY_INDEX when R = N (no gaps, no unresolved)
# ---------------------------------------------------------------------------
def test_full_body_index_label(tmp_path: Path):
    """When all official sources are content-ready without gaps, labels FULL_BODY_INDEX."""
    db_path = tmp_path / "test_reg.db"
    sources = [(i, f"https://example.com/doc_{i}", "fetched") for i in range(1, 21)]
    registry = _populate_registry(db_path, sources)

    ready_ids = list(range(1, 21))
    snapshot = _make_snapshot(ready_doc_ids=ready_ids, chunk_count=40)
    doc_index = _make_index_manifest(
        member_count=20,
        membership_sha256=_compute_membership_sha(ready_ids),
    )
    chunk_index = _make_index_manifest(item_type="chunk", member_count=40)

    ledger = build_coverage_ledger(registry, snapshot, [doc_index, chunk_index])

    assert ledger.official_count == 20
    assert ledger.ready_doc_count == 20
    assert ledger.terminal_gap_count == 0
    assert ledger.unresolved_count == 0
    assert ledger.source_scope == "FULL_BODY_INDEX"

    require_scope(ledger, "FULL_BODY_INDEX")
    require_scope(ledger, "FULL_SOURCE_PASS_WITH_GAPS")


# ---------------------------------------------------------------------------
# Test 10: FULL_HYBRID and FULL_HYBRID_WITH_GAPS with dense index and reranker
# ---------------------------------------------------------------------------
def test_full_hybrid_labels(tmp_path: Path):
    """Dense index and reranker enable FULL_HYBRID or FULL_HYBRID_WITH_GAPS."""
    db_path = tmp_path / "test_reg.db"
    # 20 sources: 18 fetched, 2 fetch_failed
    sources = [(i, f"https://example.com/doc_{i}", "fetched") for i in range(1, 19)]
    sources.extend([(19, "https://example.com/doc_19", "fetch_failed"), (20, "https://example.com/doc_20", "fetch_failed")])
    registry = _populate_registry(db_path, sources)

    ready_ids = list(range(1, 19))
    snapshot = _make_snapshot(ready_doc_ids=ready_ids, chunk_count=36)
    doc_index = _make_index_manifest(
        member_count=18,
        membership_sha256=_compute_membership_sha(ready_ids),
    )
    chunk_index = _make_index_manifest(item_type="chunk", member_count=36)
    dense_index = _make_index_manifest(
        item_type="chunk",
        member_count=36,
        backend="faiss_cpu_ivfpq",
    )

    ledger_with_gaps = build_coverage_ledger(
        registry, snapshot, [doc_index, chunk_index, dense_index], has_real_reranker=True
    )
    assert ledger_with_gaps.dense_row_count == 36
    assert ledger_with_gaps.source_scope == "FULL_HYBRID_WITH_GAPS"
    require_scope(ledger_with_gaps, "FULL_HYBRID_WITH_GAPS")
    require_scope(ledger_with_gaps, "FULL_SOURCE_PASS_WITH_GAPS")

    # Now with 0 gaps: FULL_HYBRID
    db_path_full = tmp_path / "full_reg.db"
    sources_full = [(i, f"https://example.com/doc_{i}", "fetched") for i in range(1, 19)]
    reg_full = _populate_registry(db_path_full, sources_full)
    snapshot_full = _make_snapshot(ready_doc_ids=ready_ids, chunk_count=36)

    ledger_full = build_coverage_ledger(
        reg_full, snapshot_full, [doc_index, chunk_index, dense_index], has_real_reranker=True
    )
    assert ledger_full.source_scope == "FULL_HYBRID"
    require_scope(ledger_full, "FULL_HYBRID")
    require_scope(ledger_full, "FULL_BODY_INDEX")


# ---------------------------------------------------------------------------
# Test 11: Document index with empty hash rejected when full count
# ---------------------------------------------------------------------------
def test_doc_manifest_empty_hash_rejected_when_full_count(tmp_path: Path):
    """Document index manifest with empty or missing membership hash MUST be rejected when count matches."""
    db_path = tmp_path / "test_reg.db"
    sources = [(i, f"https://example.com/doc_{i}", "fetched") for i in range(1, 11)]
    registry = _populate_registry(db_path, sources)

    ready_ids = list(range(1, 11))
    snapshot = _make_snapshot(ready_doc_ids=ready_ids, chunk_count=20)

    # Empty membership_sha256 while member_count == 10
    doc_index_empty_hash = _make_index_manifest(
        item_type="document",
        member_count=10,
        membership_sha256="",
    )
    chunk_index = _make_index_manifest(item_type="chunk", member_count=20)

    with pytest.raises(ScopeViolationError, match="membership hash mismatch"):
        build_coverage_ledger(registry, snapshot, [doc_index_empty_hash, chunk_index])


# ---------------------------------------------------------------------------
# Test 12: Registry ingest hash mismatch rejected
# ---------------------------------------------------------------------------
def test_registry_hash_mismatch_rejected(tmp_path: Path):
    """Registry with divergent input_sha256 compared to snapshot MUST be rejected."""
    mock_registry = MagicMock(spec=SourceRegistry)
    mock_registry.conn = None
    mock_registry.count_sources.return_value = 10
    mock_registry.count_unique_urls.return_value = 10
    mock_registry.get_ingest_state.return_value = ("divergent_hash_xyz", 10, True)

    ready_ids = list(range(1, 11))
    snapshot = _make_snapshot(ready_doc_ids=ready_ids, registry_sha256="expected_canonical_hash")

    doc_index = _make_index_manifest(
        item_type="document",
        member_count=10,
        membership_sha256=_compute_membership_sha(ready_ids),
    )

    with pytest.raises(ScopeViolationError, match="registry_sha256 mismatch"):
        build_coverage_ledger(mock_registry, snapshot, [doc_index])


# ---------------------------------------------------------------------------
# Test 13: Explicit scope check when target requires hybrid
# ---------------------------------------------------------------------------
def test_require_scope_explicit_scope_disallowed(tmp_path: Path):
    """Target requiring FULL_HYBRID or FULL_HYBRID_WITH_GAPS must fail if ledger lacks hybrid."""
    db_path = tmp_path / "test_reg.db"
    sources = [(i, f"https://example.com/doc_{i}", "fetched") for i in range(1, 11)]
    registry = _populate_registry(db_path, sources)

    ready_ids = list(range(1, 11))
    snapshot = _make_snapshot(ready_doc_ids=ready_ids, chunk_count=20)
    doc_index = _make_index_manifest(
        member_count=10,
        membership_sha256=_compute_membership_sha(ready_ids),
    )
    chunk_index = _make_index_manifest(item_type="chunk", member_count=20)

    # Ledger is FULL_BODY_INDEX (no dense index, no reranker)
    ledger = build_coverage_ledger(registry, snapshot, [doc_index, chunk_index])
    assert ledger.source_scope == "FULL_BODY_INDEX"

    # Requiring FULL_HYBRID must fail with explicit allowed scopes error
    with pytest.raises(ScopeViolationError, match="requires one of"):
        require_scope(ledger, "FULL_HYBRID")

    # Requiring FULL_HYBRID_WITH_GAPS must also fail
    with pytest.raises(ScopeViolationError, match="requires one of"):
        require_scope(ledger, "FULL_HYBRID_WITH_GAPS")

