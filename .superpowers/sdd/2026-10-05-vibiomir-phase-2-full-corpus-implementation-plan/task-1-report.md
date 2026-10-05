# Task 1 Report: Full coverage ledger và strict scope guards

## 1. What was implemented

- **`src/operations/coverage.py`**:
  - Implemented `build_coverage_ledger(registry, snapshot, index_manifests, campaign_config=None, has_real_reranker=False) -> CoverageLedger`:
    - Validates manifest invariants: checks `index_manifest.snapshot_id == snapshot.snapshot_id`, raising `ScopeViolationError("snapshot_id mismatch")` on discrepancy.
    - Validates document membership hashes: checks `index_manifest.membership_sha256 == snapshot.source_membership_sha256` when index member count equals or exceeds ready document count, raising `ScopeViolationError("membership hash mismatch")` on forgery/drift.
    - Resolves official sources vs unique URLs (`official_count`, `unique_url_count`).
    - Inspects source registry state joined with `fetch_jobs`, partitioning into ready documents, terminal failure gaps (`fetch_failed`, `blocked`, `robots_blocked`, `unsupported`, etc.), and unresolved sources (`pending`, `leased`, `retry_wait`, etc.).
    - Preserves strict invariant: `official_count == ready_doc_count + terminal_gap_count + unresolved_count`.
    - Enforces scope labeling rules:
      - `PARTIAL_BODY_INDEX`: when `campaign_config` has `source_limit`, `domain_whitelist`, non-all scope, or when `unresolved_count > 0` or sparse docs/chunks are not 100% indexed.
      - `FULL_SOURCE_PASS_WITH_GAPS`: when 0 unresolved sources, 100% sparse indexing, terminal gaps exist, and no dense+reranker.
      - `FULL_BODY_INDEX`: when 0 unresolved sources, 0 terminal gaps, 100% sparse indexing, and no dense+reranker.
      - `FULL_HYBRID_WITH_GAPS`: when 0 unresolved sources, terminal gaps exist, full dense index present, and real reranker configured.
      - `FULL_HYBRID`: when 0 unresolved sources, 0 terminal gaps, full dense index present, and real reranker configured.
  - Implemented `require_scope(ledger, target) -> None`:
    - Strict scope verification checking compatibility between target requirement and actual ledger scope.
    - Rejects `PARTIAL_BODY_INDEX` ledgers from passing full source or body requirements.
    - Rejects ledgers with terminal gaps when complete body index is demanded (`match="terminal gaps"`).
    - Requires hybrid pipeline (dense index + reranker) when hybrid scopes are requested.
    - Supports normalized aliases (`full_source_pass`, `strict_full_body`, etc.).

- **`src/data/phase2_contracts.py`**:
  - Verified and completed Phase 2 contracts:
    - Scope constants: `SCOPE_PARTIAL_BODY_INDEX`, `SCOPE_FULL_SOURCE_PASS_WITH_GAPS`, `SCOPE_FULL_BODY_INDEX`, `SCOPE_FULL_HYBRID_WITH_GAPS`, `SCOPE_FULL_HYBRID`, and `VALID_SCOPES`.
    - `ScopeViolationError` exception class.
    - `CoverageLedger` dataclass with `to_dict()`.
    - `IndexManifest` dataclass matching specification.
    - Verified all configuration dataclasses (`FullCampaignConfig`, `SparseConfig`, `DenseConfig`, `EvidenceConfig`, `RerankConfig`, `OutputPolicy`) match Section 12 specifications.

## 2. Testing and Test Results

### TDD Evidence
- **RED Phase**:
  Command: `pytest tests/operations/test_full_scope.py -q`
  Result: Failed with `ModuleNotFoundError: No module named 'src.operations.coverage'`.
- **GREEN Phase**:
  Command: `pytest tests/operations/test_full_scope.py -q`
  Result: 10 passed in 0.24s.
  - Test 1: Full registry (4.39M) + pilot bodies (465) correctly labels `PARTIAL_BODY_INDEX` and rejects full-scope claims.
  - Test 2: In-flight or pending jobs block full source pass.
  - Test 3: Terminal failures reported accurately (`FULL_SOURCE_PASS_WITH_GAPS`).
  - Test 4: Missing docs or chunks from sparse index force `PARTIAL_BODY_INDEX`.
  - Test 5: Equal count with forged/mismatched membership hash rejects with `membership hash mismatch`.
  - Test 6: Snapshot ID mismatch in index manifests rejects with `snapshot_id mismatch`.
  - Test 7: Distinguishes official rows (N) from unique URLs (U).
  - Test 8: Campaign filters (`source_limit`, `domain_whitelist`) force `PARTIAL_BODY_INDEX`.
  - Test 9: Complete clean content labels `FULL_BODY_INDEX`.
  - Test 10: Dense index + real reranker enables `FULL_HYBRID_WITH_GAPS` and `FULL_HYBRID`.

- **Full Test Suite Regression**:
  Command: `pytest -q`
  Result: 380 passed, 3 warnings in 118.96s (100% passing across entire test suite).

## 3. Files Changed

- `src/data/phase2_contracts.py`: Added scope constants, `ScopeViolationError`, `CoverageLedger.to_dict()`, `IndexManifest`.
- `src/operations/coverage.py`: Implemented coverage ledger builder and strict scope guards.
- `tests/operations/test_full_scope.py`: 10 comprehensive unit tests covering all scope invariant and guard scenarios.
- `.superpowers/sdd/2026-10-05-vibiomir-phase-2-full-corpus-implementation-plan/progress.md`: Updated task ledger.

## 4. Self-Review Findings

- Invariant verification: `official_count == ready_doc_count + terminal_gap_count + unresolved_count` is guaranteed and verified in all test scenarios.
- Handling mock registries: `registry.conn is None` gracefully defaults to reporting unjoined sources as unresolved without crashing.
- No silent relabeling: Any partial scope or unindexed documents/chunks immediately downgrade the scope to `PARTIAL_BODY_INDEX` and trigger `ScopeViolationError` upon attempting publish or export.

## 5. Issues or Concerns

- None. All coverage invariants and guards are strictly enforced and verified.

## 6. Review Fix Report (Post-Review Follow-up)

### Reviewer Feedback Addressed
1. **Strict Non-Empty Membership Hash Guard**:
   - `src/operations/coverage.py`: Updated document manifest check so that when `member_count >= snapshot.ready_document_count`, `manifest.membership_sha256` MUST be strictly non-empty and match `snapshot.source_membership_sha256`. An empty string or missing hash no longer bypasses verification and immediately raises `ScopeViolationError`.
   - Added unit test `test_doc_manifest_empty_hash_rejected_when_full_count` verifying empty hash rejection.

2. **Explicit Target Scope Compatibility**:
   - `src/operations/coverage.py`: Updated `require_scope` with `ALLOWED_SCOPES_FOR_TARGET` map. Now explicitly asserts that `ledger.source_scope` belongs to the allowed set of scopes for the target (e.g. `FULL_HYBRID` strictly requires `FULL_HYBRID`, rejecting non-hybrid scopes with a clear error message).
   - Added unit test `test_require_scope_explicit_scope_disallowed` verifying explicit scope rejection.

3. **Registry Hash Check and Cleanup**:
   - `src/operations/coverage.py`: Added registry ingest state hash verification against `snapshot.registry_sha256` if present in registry, rejecting divergent hashes with `ScopeViolationError("registry_sha256 mismatch")`.
   - Added unit test `test_registry_hash_mismatch_rejected`.
   - Utilized `UNRESOLVED_STATES` in registry diagnostic logging.

### Test Verification
- `pytest tests/operations/test_full_scope.py -q`: 13 passed in 0.24s.
- `pytest -q`: 383 passed, 3 warnings in 58.94s (100% passing across entire test suite).

