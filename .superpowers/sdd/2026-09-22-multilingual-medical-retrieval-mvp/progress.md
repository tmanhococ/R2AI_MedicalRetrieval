# SDD ledger — plan: docs/superpowers/plans/2026-09-22-multilingual-medical-retrieval-mvp.md

Tasks:
- [x] Task 1: Environment & Project Scaffolding, Base Config
- [x] Task 2: Data Schemas & Validators
- [x] Task 3: Dataset Adapter Protocol & In-Memory / JSONL Adapter
- [x] Task 4: Dynamic Chunker Layer (Prechunked, Auto, Hybrid)
- [x] Task 5: Query Normalization & Adaptive Query Router
- [x] Task 6: Translation Protocol, Offline Cache & Fault-Tolerant Translator
- [x] Task 7: Multilingual Tokenizers & BM25 Indexing / Retrieval (BM25s / Rank-BM25)
- [x] Task 8: Dense Indexing & Vector Search (FAISS & NumPy Fallback)
- [x] Task 9: Retrieval Orchestrator (6 Parallel Streams)
- [x] Task 10: Weighted Reciprocal Rank Fusion (Weighted RRF)
- [x] Task 11: Cross-Encoder Reranker with Chunk Aggregation
- [x] Task 12: Joint Scoring & Structural Consistency Post-Processing
- [x] Task 13: Macro F2 Metrics & Calibration Engine (Anti-Leakage)
- [x] Task 14: Submission Exporter & Error Analysis Reporting
- [x] Task 15: Pipeline CLI Entrypoints & End-to-End Integration Smoke Test

History:
- Task 1: complete (commits 45b3e19, review clean)
- Task 1: minor (deferred): src/utils/config.py: return yaml.safe_load(f) or {} to handle empty YAML files gracefully
- Task 2: complete (commits d01df0b, review clean)
- Task 2: minor (deferred): src/data/schemas.py: tighten language in SearchHit/FusedHit/ScoredHit to Literal['vi', 'en', 'zh']
- Task 2: minor (deferred): src/data/validators.py: add chunk_id uniqueness check
- Task 3: complete (commits 00a0ca2, review clean)
- Task 3: minor (deferred): src/data/dataset_adapter.py: validate split name if queries_dir/qrels_dir points to a direct file
- Task 3: minor (deferred): src/data/dataset_adapter.py: standardize directory check error types
- Task 4: complete (commits 40aacaf, review clean)
- Task 4: minor (deferred): src/indexing/chunker.py: improve mixed Chinese/Latin tokenization heuristic
- Task 4: minor (deferred): src/indexing/chunker.py: emit at least 1 chunk for documents shorter than min_chunk_length
- Task 5: complete (commits 91bbd27, review clean)
- Task 5: minor (deferred): src/query/query_processor.py: ensure downstream DenseRetriever handles subword token truncation boundaries
- Task 5: minor (deferred): src/query/query_processor.py: runtime isinstance check for QueryRecord
- Task 6: complete (commits 7bca314, review clean)
- Task 7: complete (commits e866927, review clean)
- Task 7: minor (deferred): src/indexing/tokenizers.py: module-level cache for PorterStemmer instance
- Task 7: minor (deferred): src/retrieval/bm25_retriever.py: case-normalize language in search methods
- Task 8: complete (commits 497d808, 91b7e25, review clean)
- Task 8: minor (deferred): src/retrieval/dense_retriever.py: use np.argpartition for large corpora speedup
- Task 9: complete (commits 726adbc, review clean)
- Task 10: complete (commits 1a9e8c6, 66a521c, review clean)
- Task 11: complete (commits fe008bf, 4a46ad9, review clean)
- Task 12: complete (commits 75fc216, 3f34f09, review clean)
- Task 13: complete (commits 7c44712, ac3f2b9, review clean)
- Task 14: complete (commits 1acd9eb, dcb4a2e, review clean)
- Task 15: complete (commits 4fa47ed, 9fdd7e6, review clean)
