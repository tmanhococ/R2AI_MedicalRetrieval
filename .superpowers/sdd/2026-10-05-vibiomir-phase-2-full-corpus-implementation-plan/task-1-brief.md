### Task 1: Full coverage ledger và strict scope guards

**Files:** phase2_contracts.py, operations/coverage.py; tests/operations/test_full_scope.py.

**Interfaces:** build_coverage_ledger; require_scope; CoverageLedger; config types mục12.

- [ ] Tests full registry+465 bodies must label PARTIAL; pending doc blocks full source pass; terminal failures reported; ready doc/chunk omitted from index blocks publish; equal count/wrong membership rejects.
- [ ] Run pytest tests/operations/test_full_scope.py -q; expected FAIL chưa có guard.
- [ ] Implement joins/set membership hashes theo official IDs, distinguish URL rows/unique URLs. Source limit/domain filter đặt partial flag tự động.
- [ ] Run tests; expected PASS và no silent relabeling. Commit: feat: enforce full-corpus coverage contracts.

