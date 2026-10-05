### Task 3: Full snapshot và source-store batch I/O

**Files:** snapshot.py, source_store.py; tests/data/test_full_snapshot.py.

**Interfaces:** SnapshotSourceStore batch methods; EvidenceSourceStore wrapper skeleton; source/index membership from Phase1.

- [ ] Tests immutable versions, exact source offsets, und usable content, all alias IDs, cache eviction, batched reads grouped by row group; no full-corpus scan per ID.
- [ ] Run pytest tests/data/test_full_snapshot.py -q; expected FAIL missing additions.
- [ ] Implement full snapshot metadata/ledger, streaming manifest membership; batch lookups và bounded LRU. Validate actual model token budgets and re-chunk when needed before freeze. Do not load all docs into memory.
- [ ] Verify all source-ready IDs included, count/hashes, source provenance and no unresolved sources before full label. Retain gaps separate.
- [ ] Run tests + sample production snapshot integrity scan. Commit: feat: freeze complete source-pass snapshots with batched text access.

