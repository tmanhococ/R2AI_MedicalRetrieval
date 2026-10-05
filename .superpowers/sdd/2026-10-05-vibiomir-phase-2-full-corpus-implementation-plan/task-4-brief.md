### Task 4: Disk-backed global sparse indexes

**Files:** lexical_tokens.py, lucene.py, cli/index_sparse.py, existing predict.py/exporter/validators integration, configs/vibiomir_phase2_B1.yaml; tests/indexing/test_lucene_sparse.py và tests/integration/test_phase2_sparse_checkpoint.py.

**Interfaces:** LexicalRecord; LuceneSparseBackend build_partition/assemble/search.

- [ ] Tests clinical symbols/negation survive tokenization+hash encoding, VI/EN/ZH query-document alignment, real Lucene hits outside pilot and IDs above int32.
- [ ] Test partitions with different local DF: assembled search uses global stats; no per-shard raw score merge. Crash/replay cannot duplicate final IDs.
- [ ] Run pytest tests/indexing/test_lucene_sparse.py -q; expected FAIL trước implementation.
- [ ] Build pretokenized document+chunk indexes, analyzer JWhiteSpaceAnalyzer, BM25 settings, partition checkpoints và atomic final publish. Needed text lấy từ SourceStore, không storeRaw mặc định.
- [ ] Implement B1 overlay và sparse_rank_caps route theo mục12; reuse Phase1 RRF, SourceStore và exporter. Sửa Level C validator chỉ ép exact query ID set/count, không ép6.000 docs/12.000 chunks; Task10 bổ sung CE/dev policies trên interface này.
- [ ] Test sparse-only checkpoint từ index → predictions → ZIP trên fixture có IDs ngoài pilot, variable counts và empty hits; không load E5/BGE. Run pytest tests/integration/test_phase2_sparse_checkpoint.py -q.
- [ ] Verify exact memberships of all languages/und; run 1.200 queries sparse predictor và validate submission_B1_sparse.zip. No zero-score forced padding.
- [ ] Run tests, measure 50k/scale RSS/disk, then full build. Commit: feat: add full-corpus Lucene sparse retrieval.

