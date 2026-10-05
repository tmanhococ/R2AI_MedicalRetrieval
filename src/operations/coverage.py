from __future__ import annotations

import logging
from typing import Sequence

from src.data.phase2_contracts import (
    CoverageLedger,
    FullCampaignConfig,
    IndexManifest,
    ScopeViolationError,
    SCOPE_FULL_BODY_INDEX,
    SCOPE_FULL_HYBRID,
    SCOPE_FULL_HYBRID_WITH_GAPS,
    SCOPE_FULL_SOURCE_PASS_WITH_GAPS,
    SCOPE_PARTIAL_BODY_INDEX,
    VALID_SCOPES,
)
from src.data.vibiomir_contracts import SnapshotManifest
from src.data.vibiomir_registry import SourceRegistry

logger = logging.getLogger(__name__)

# Known terminal failure states for fetch jobs in registry
TERMINAL_STATES = frozenset({
    "fetch_failed",
    "terminal_fetch_failure",
    "blocked",
    "robots_blocked",
    "unsupported",
    "unsupported_content",
    "unsupported_language",
    "terminal_extraction_failure",
    "invalid_url",
    "http_error",
    "permanent_error",
})

# Known unresolved / in-flight states
UNRESOLVED_STATES = frozenset({
    "pending",
    "leased",
    "fetching",
    "retry_wait",
    "needs_review",
})

_NORMALIZED_TARGETS = {
    "PARTIAL_BODY_INDEX": SCOPE_PARTIAL_BODY_INDEX,
    "PARTIAL": SCOPE_PARTIAL_BODY_INDEX,
    "FULL_SOURCE_PASS_WITH_GAPS": SCOPE_FULL_SOURCE_PASS_WITH_GAPS,
    "FULL_SOURCE_PASS": SCOPE_FULL_SOURCE_PASS_WITH_GAPS,
    "FULL_BODY_INDEX": SCOPE_FULL_BODY_INDEX,
    "FULL_BODY": SCOPE_FULL_BODY_INDEX,
    "STRICT_FULL_BODY": SCOPE_FULL_BODY_INDEX,
    "FULL_HYBRID_WITH_GAPS": SCOPE_FULL_HYBRID_WITH_GAPS,
    "FULL_HYBRID": SCOPE_FULL_HYBRID,
}

ALLOWED_SCOPES_FOR_TARGET: dict[str, tuple[str, ...]] = {
    SCOPE_PARTIAL_BODY_INDEX: VALID_SCOPES,
    SCOPE_FULL_SOURCE_PASS_WITH_GAPS: (
        SCOPE_FULL_SOURCE_PASS_WITH_GAPS,
        SCOPE_FULL_BODY_INDEX,
        SCOPE_FULL_HYBRID_WITH_GAPS,
        SCOPE_FULL_HYBRID,
    ),
    SCOPE_FULL_BODY_INDEX: (
        SCOPE_FULL_BODY_INDEX,
        SCOPE_FULL_HYBRID,
    ),
    SCOPE_FULL_HYBRID_WITH_GAPS: (
        SCOPE_FULL_HYBRID_WITH_GAPS,
        SCOPE_FULL_HYBRID,
    ),
    SCOPE_FULL_HYBRID: (
        SCOPE_FULL_HYBRID,
    ),
}


def build_coverage_ledger(
    registry: SourceRegistry,
    snapshot: SnapshotManifest,
    index_manifests: Sequence[IndexManifest],
    campaign_config: FullCampaignConfig | None = None,
    has_real_reranker: bool = False,
) -> CoverageLedger:
    """Build and verify full-corpus coverage ledger across registry, snapshot, and indexes.

    Enforces invariants:
    1. index_manifest.snapshot_id == snapshot.snapshot_id
    2. index_manifest (document) membership_sha256 strictly non-empty and matches
       snapshot.source_membership_sha256 when member_count >= ready_document_count
    3. official_count == ready_doc_count + terminal_gap_count + unresolved_count
    """
    # 1. Manifest invariants
    for manifest in index_manifests:
        if manifest.snapshot_id != snapshot.snapshot_id:
            raise ScopeViolationError(
                f"snapshot_id mismatch: index manifest has '{manifest.snapshot_id}', "
                f"expected snapshot '{snapshot.snapshot_id}'"
            )
        if (
            manifest.item_type == "document"
            and manifest.member_count >= snapshot.ready_document_count
        ):
            if not manifest.membership_sha256 or manifest.membership_sha256 != snapshot.source_membership_sha256:
                raise ScopeViolationError(
                    f"membership hash mismatch: index has '{manifest.membership_sha256}', "
                    f"expected '{snapshot.source_membership_sha256}'"
                )

    # 2. Count indexed members across manifests
    sparse_doc_count = 0
    sparse_chunk_count = 0
    dense_row_count = 0

    for manifest in index_manifests:
        is_dense = (
            manifest.backend.startswith("faiss")
            or "dense" in manifest.backend.lower()
            or (manifest.model_fingerprint is not None and manifest.backend not in ("lucene", "bm25", "bm25s"))
        )
        if is_dense:
            dense_row_count += manifest.member_count
        elif manifest.item_type == "document":
            sparse_doc_count += manifest.member_count
        elif manifest.item_type == "chunk":
            sparse_chunk_count += manifest.member_count

    # 3. Query source registry
    official_count = registry.count_sources()
    unique_url_count = registry.count_unique_urls()
    ready_doc_count = snapshot.ready_document_count
    chunk_count = snapshot.chunk_count

    terminal_gap_count = 0
    if getattr(registry, "conn", None) is not None:
        cursor = registry.conn.execute(
            """
            SELECT j.state, count(*)
            FROM sources s
            LEFT JOIN fetch_jobs j ON s.fetch_job_id = j.job_id
            GROUP BY j.state
            """
        )
        state_counts = dict(cursor.fetchall())
        terminal_gap_count = sum(cnt for st, cnt in state_counts.items() if st in TERMINAL_STATES)
        unresolved_state_count = sum(
            cnt for st, cnt in state_counts.items()
            if st in UNRESOLVED_STATES or st is None or (st not in TERMINAL_STATES and st not in ("fetched", "content_ready", "ready"))
        )
        logger.debug(
            "Registry state summary: terminal=%d, unresolved_db=%d, details=%s",
            terminal_gap_count,
            unresolved_state_count,
            state_counts,
        )

    unresolved_count = official_count - ready_doc_count - terminal_gap_count
    if unresolved_count < 0:
        raise ScopeViolationError(
            f"Corrupt coverage state: ready docs ({ready_doc_count}) + terminal gaps ({terminal_gap_count}) "
            f"> official sources ({official_count})"
        )

    # Invariant: official_count == ready_doc_count + terminal_gap_count + unresolved_count
    assert official_count == ready_doc_count + terminal_gap_count + unresolved_count

    # Registry hash verification
    reg_hash = ""
    try:
        ingest_state = registry.get_ingest_state()
        if ingest_state:
            reg_hash = ingest_state[0]
            if snapshot.registry_sha256 and reg_hash and reg_hash != snapshot.registry_sha256:
                raise ScopeViolationError(
                    f"registry_sha256 mismatch: registry has '{reg_hash}', "
                    f"expected snapshot '{snapshot.registry_sha256}'"
                )
    except ScopeViolationError:
        raise
    except Exception:
        pass
    if not reg_hash:
        reg_hash = snapshot.registry_sha256

    # 4. Scope labeling
    forced_partial = False
    if campaign_config is not None:
        if campaign_config.source_limit is not None:
            forced_partial = True
        if campaign_config.domain_whitelist:
            forced_partial = True
        if campaign_config.scope != "all":
            forced_partial = True

    docs_fully_indexed = sparse_doc_count >= ready_doc_count
    chunks_fully_indexed = sparse_chunk_count >= chunk_count
    has_unresolved = unresolved_count > 0

    if forced_partial or has_unresolved or not docs_fully_indexed or not chunks_fully_indexed:
        source_scope = SCOPE_PARTIAL_BODY_INDEX
    else:
        is_hybrid = (
            has_real_reranker
            and dense_row_count >= chunk_count
            and chunk_count > 0
        )
        if terminal_gap_count > 0:
            source_scope = SCOPE_FULL_HYBRID_WITH_GAPS if is_hybrid else SCOPE_FULL_SOURCE_PASS_WITH_GAPS
        else:
            source_scope = SCOPE_FULL_HYBRID if is_hybrid else SCOPE_FULL_BODY_INDEX

    return CoverageLedger(
        official_count=official_count,
        unique_url_count=unique_url_count,
        ready_doc_count=ready_doc_count,
        terminal_gap_count=terminal_gap_count,
        unresolved_count=unresolved_count,
        chunk_count=chunk_count,
        sparse_doc_count=sparse_doc_count,
        sparse_chunk_count=sparse_chunk_count,
        dense_row_count=dense_row_count,
        source_scope=source_scope,
        registry_hash=reg_hash,
        snapshot_id=snapshot.snapshot_id,
    )


def require_scope(ledger: CoverageLedger, target: str) -> None:
    """Verify that a coverage ledger meets or exceeds the required target scope.

    Raises ScopeViolationError if the target scope is not satisfied.
    """
    normalized_key = target.strip().upper()
    target_scope = _NORMALIZED_TARGETS.get(normalized_key, normalized_key)

    if target_scope not in VALID_SCOPES:
        raise ValueError(f"Unknown scope target: '{target}' (valid: {VALID_SCOPES})")

    # If target is PARTIAL_BODY_INDEX, any ledger satisfies it
    if target_scope == SCOPE_PARTIAL_BODY_INDEX:
        return

    # If actual scope is PARTIAL_BODY_INDEX, it cannot satisfy any higher scope
    if ledger.source_scope == SCOPE_PARTIAL_BODY_INDEX:
        raise ScopeViolationError(
            f"Scope violation: ledger has {ledger.source_scope}, cannot satisfy '{target}'. "
            f"Unresolved sources: {ledger.unresolved_count}, "
            f"docs indexed: {ledger.sparse_doc_count}/{ledger.ready_doc_count}, "
            f"chunks indexed: {ledger.sparse_chunk_count}/{ledger.chunk_count}."
        )

    # Check terminal gaps requirement: FULL_BODY_INDEX and FULL_HYBRID do not allow gaps
    if target_scope in (SCOPE_FULL_BODY_INDEX, SCOPE_FULL_HYBRID) and ledger.terminal_gap_count > 0:
        raise ScopeViolationError(
            f"Scope violation: target '{target}' requires complete body coverage, "
            f"but terminal gaps exist ({ledger.terminal_gap_count} gaps)."
        )

    # Explicitly check that ledger.source_scope is in the allowed scopes for this target
    allowed = ALLOWED_SCOPES_FOR_TARGET[target_scope]
    if ledger.source_scope not in allowed:
        raise ScopeViolationError(
            f"Scope violation: target '{target}' requires one of {allowed}, "
            f"but actual scope is '{ledger.source_scope}'."
        )
