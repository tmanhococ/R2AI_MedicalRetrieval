### Task 5: Immutable snapshot, source store và adapter/index integration

**Files:** Create ingestion/snapshot.py, data/source_store.py; complete vibiomir_adapter.py; modify indexers và cli/index.py khi cần; tests/data/test_snapshot_adapter.py.

**Interfaces:**

- build_snapshot(registry: SourceRegistry, config: SnapshotConfig) -> SnapshotManifest
- SnapshotSourceStore(snapshot_path: Path) implements SourceStore.
- ViBioMIRAdapter.iter_documents() -> Iterator[DocumentRecord]
- ViBioMIRAdapter.iter_chunks() -> Iterator[ChunkRecord]
- ViBioMIRAdapter.load_queries(split: str) -> list[QueryRecord]
- ViBioMIRAdapter.load_ground_truth(split: str) -> list[EvaluationRecord]
- validate_index_manifest(index_path: Path, snapshot: SnapshotManifest) -> None.

- [ ] Viết tests shard rotation/atomic publication, content hash, same URL giữ hai doc IDs, snapshot A không đổi khi fetch B mới, missing foreign key, index snapshot mismatch và vector→chunk mapping mismatch.
- [ ] Run: pytest tests/data/test_snapshot_adapter.py -q. Expected: FAIL missing snapshot behavior.
- [ ] Implement sharded writes + manifest; adapter chuyển official int IDs sang core strings. Metadata chứa source hashes/offsets và exact chunk text; SourceStore resolve theo snapshot.
- [ ] Thêm batch consumption vào data/embedding path nếu repo đang load list; kiểm tra bằng stream iterator fixture không cho iterate lần hai. Không chuyển mọi text thành list trước batch embed. BM25S build được đo riêng, không claim streaming build chỉ nhờ data loader.
- [ ] Measure pilot peak RAM cho tokenize/BM25 build và dense build, bytes/vector; thêm resource estimate vào index manifest. Model revision/prefix/normalization mismatches phải fail trước search; resource budget breach không âm thầm cắt corpus.
- [ ] Run tests và existing index smoke test. Expected: PASS, count invariants và mappings đúng.
- [ ] Commit: feat: connect frozen source snapshots to retrieval indexes.

