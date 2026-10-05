# ViBioMIR Phase 2 Runbook

Complete operations guide for Phase 2 full-corpus pipeline execution, stage orchestration, checkpoint resumption, validation, and submission export.

## Overview

ViBioMIR Phase 2 introduces full-corpus retrieval, sparse/dense indexing, hybrid candidate generation, neural BGE reranking, evidence span selection, variable-size output selection, transactional checkpointing, fingerprint validation, and reproducible submission packaging.

## Pipeline Stages & Commands

### 1. Crawl & Ingest
Crawl raw sources and ingest into the corpus repository:
```bash
python -m src.cli.crawl --config configs/vibiomir_phase2.yaml
python -m src.cli.ingest_vibiomir --config configs/vibiomir_phase2.yaml
```

### 2. Build Snapshot
Freeze corpus into an immutable snapshot:
```bash
python -m src.cli.build_snapshot --config configs/vibiomir_phase2.yaml --output-dir data/snapshots/phase2_frozen
```

### 3. Sparse & Dense Indexing
Build Lucene / BM25 sparse index and FAISS dense index:
```bash
python -m src.cli.index_sparse --config configs/vibiomir_phase2.yaml --snapshot data/snapshots/phase2_frozen --output-dir indexes/sparse
python -m src.cli.encode_corpus --config configs/vibiomir_phase2.yaml --snapshot data/snapshots/phase2_frozen --output-dir indexes/embeddings
python -m src.cli.index_dense --config configs/vibiomir_phase2.yaml --embeddings-dir indexes/embeddings --output-dir indexes/dense
```

### 4. End-to-End Prediction & Export
Run stage orchestration, candidate retrieval, reranking, and export `submission_B.zip`:
```bash
python -m src.cli.predict --config configs/vibiomir_phase2.yaml --snapshot data/snapshots/phase2_frozen --index-dir indexes/ --export-submission --resume
```

## Checkpointing & Fingerprints
- Each query is checkpointed atomically into `predictions_checkpoint.jsonl`.
- Run fingerprints incorporate query manifests, snapshot hashes, index manifests, and model revisions.
- Changing snapshots, encoders, or rerankers automatically invalidates downstream stage checkpoints.
