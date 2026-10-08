# ViBioMIR: Multilingual Medical Information Retrieval & Grounded Evidence RAG Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Benchmark](https://img.shields.io/badge/Challenge-ViBioMIR%202026-green.svg)](https://huggingface.co/datasets/AIGuruTinix/ViBioMIR)

A high-performance, fault-tolerant, and fully reproducible multilingual Medical Information Retrieval (IR) and Grounded Evidence Extraction engine designed for the **ViBioMIR 2026** competition.

The system scales across **4.39 million web documents** and provides grounded paragraph and sub-passage evidence retrieval for **1,200 complex clinical queries** across Vietnamese, English, and Chinese.

---

## 📑 Table of Contents
- [Architecture Overview](#-architecture-overview)
- [Key Features](#-key-features)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [End-to-End Pipeline Runbook](#-end-to-end-pipeline-runbook)
  - [1. Corpus Acquisition (Turbo Crawler)](#1-corpus-acquisition-turbo-crawler)
  - [2. Corpus Snapshot Freeze & Extraction](#2-corpus-snapshot-freeze--extraction)
  - [3. Multilingual Sparse Indexing (Lucene / BM25s)](#3-multilingual-sparse-indexing-lucene--bm25s)
  - [4. Dense Encoding & Vector Store (E5-base + Float16 Blocks)](#4-dense-encoding--vector-store-e5-base--float16-blocks)
  - [5. CPU FAISS Indexing & Refinement Audit](#5-cpu-faiss-indexing--refinement-audit)
  - [6. End-to-End Inference & Submission Packaging](#6-end-to-end-inference--submission-packaging)
- [Production Safeguards & Engineering Highlights](#-production-safeguards--engineering-highlights)
- [Offline Validation & Scoring Parity](#-offline-validation--scoring-parity)
- [License](#-license)

---

## 🏛 Architecture Overview

The pipeline implements **Baseline B** (`vibiomir_phase2_B`): a multi-stage hybrid retrieval and reranking architecture tailored for large-scale multilingual medical knowledge:

```
[Official Queries (1,200)]
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Query Processing & Multilingual Translation Cache     │
│  (Original Vietnamese + Cached EN/ZH Variants)         │
└────────────────────────────────────────────────────────┘
         │
         ├───► Lucene / BM25s Sparse Docs (Top-500/lang)
         ├───► Lucene / BM25s Sparse Chunks (Top-1000/lang)
         └───► Multilingual E5-base Dense Chunks (FAISS IVF-PQ + Float16 Exact Cosine Refinement)
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Grouped Weighted RRF Hybrid Fusion                    │
│  (w_sparse_doc=1.0, w_sparse_chunk=0.75, w_dense=1.0)  │
│  Correlated Variant Grouping (Max score per chunk)     │
└────────────────────────────────────────────────────────┘
         │  (Top-200 Candidate Documents)
         ▼
┌────────────────────────────────────────────────────────┐
│  Document-Conditioned Evidence Span Recovery           │
│  - Local chunk scoring (term overlap & dense dot)      │
│  - Contiguous neighbor block expansion in same section │
│  - Strict character-span slicing from raw source text  │
└────────────────────────────────────────────────────────┘
         │  (Top evidence spans)
         ▼
┌────────────────────────────────────────────────────────┐
│  Neural Reranker (BAAI/bge-reranker-v2-m3)             │
│  - Dual subwindowing (passage <=320 tok, query <=128)  │
│  - Max raw logit aggregation across window pairs       │
│  - OOM-resilient batch recovery                        │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Variable-Size Output Selection & Dev Calibration      │
│  - Heuristic CE Positive cutoff (score > 0.0)          │
│  - Expanded span selection (tolerance +0.5)            │
│  - Document cap (10) & Evidence cap (20 spans, 3/doc)  │
└────────────────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────┐
│  Submission Export & Packaging                         │
│  results.json (Exact 1,200 queries) -> submission_B.zip│
└────────────────────────────────────────────────────────┘
```

---

## ⚡ Key Features

1. **Turbo Crawler (4.39M URLs)**:
   - Asynchronous crawling using `aiohttp` with fair per-domain concurrency, smart backoff, gzip compression, and SHA-256 payload integrity hashing.
   - Crawls millions of pages with high throughput (~200+ URLs/sec).
   - Dedicated failure ledger tracking 100% of blocked, dead, and timed-out domains (`source_gaps.parquet`).

2. **Optimized Snapshot Storage & Sharding**:
   - Zero-bloat chunk packing: Structural section parsing prevents micro-chunk explosion, maintaining **~4.5 chunks/doc** (saving ~50 GB of disk space).
   - Sharded `.parquet` storage backed by SQLite location index (`lookup.sqlite`).
   - **Transaction-Safe Resume Checkpointing**: Can safely stop (`Ctrl+C`) or shut down the machine at any time; restarts seamlessly with `--resume`.

3. **Multilingual Hybrid Retrieval**:
   - **Sparse Retrieval**: BM25s / Lucene with Vietnamese word segmentation and subword fallback.
   - **Dense Retrieval**: `intfloat/multilingual-e5-base` with float16 disk blocks and checksum manifests.
   - **FAISS CPU IVF-PQ with Refinement**: Fast approximate nearest-neighbor search followed by exact float16 stored-vector cosine rescoring (ANN retention verified $\ge 0.95$).
   - **Grouped Weighted RRF**: Fair fusion preventing correlated translated variants from dominating scores.

4. **Document-Conditioned Grounded Evidence**:
   - Evidence spans are exact sub-character slices `source_text[start:end]` from the document body. No synthetic generation, hallucinations, or ungrounded text.
   - Contextual section-based expansion recovers surrounding clinical nuances without exceeding model sequence lengths.

5. **Cross-Encoder Neural Reranking**:
   - `BAAI/bge-reranker-v2-m3` running with fp16 mixed precision on CUDA.
   - Subwindow splitting and max raw logit retention to avoid sigmoid squashing distortion.
   - Checkpointed inference with deterministic disk caching.

6. **Variable-Size Output Policy**:
   - No forced top-K padding: Queries with zero relevant medical evidence cleanly return empty sets (`0` docs, `0` chunks).
   - Document score strictly equals the maximum evidence score of its spans (no double-counting).

---

## 📁 Repository Structure

```
mvp-rag-medical/
├── configs/                            # Experiment and pipeline configurations
│   ├── vibiomir_phase1.yaml            # Pilot baseline A config
│   ├── vibiomir_phase2.yaml            # Production full-corpus baseline B config
│   └── vibiomir_phase2_B1.yaml         # Sparse-only fallback route config
├── docs/vibiomir/                      # Specifications, runbooks & summaries
│   ├── baseline_A.md                   # Baseline A specification
│   ├── baseline_B.md                   # Baseline B full-corpus specification
│   ├── phase1_runbook.md               # Phase 1 operational guide
│   └── phase2_runbook.md               # Phase 2 operational runbook
├── src/
│   ├── cli/                            # Command-line entry points
│   │   ├── turbo_crawl.py              # High-speed async bulk crawler
│   │   ├── crawl.py                    # Standard crawler runner
│   │   ├── build_snapshot.py           # Snapshot builder with checkpoint resume
│   │   ├── index_sparse.py             # Lucene / BM25 sparse index builder
│   │   ├── encode_corpus.py            # Multilingual E5 batch embedder
│   │   ├── index_dense.py              # FAISS CPU IVF-PQ index builder
│   │   ├── generate_candidates.py      # Multilingual hybrid candidate retriever
│   │   ├── rerank_candidates.py        # BGE cross-encoder evidence reranker
│   │   ├── predict.py                  # End-to-end stage orchestrator
│   │   ├── validate_submission.py      # Offline submission validator
│   │   └── doctor.py                   # System hardware & resource diagnostic
│   ├── data/                           # Schemas, registry & source stores
│   │   ├── vibiomir_contracts.py       # Core data models and configurations
│   │   ├── phase2_contracts.py         # Phase 2 stage manifests & interfaces
│   │   ├── vibiomir_registry.py        # SQLite acquisition registry
│   │   └── source_store.py             # Batched Parquet + SQLite random access store
│   ├── ingestion/                      # Fetching, parsing & quality gates
│   │   ├── extractors.py               # LXML & Trafilatura text extraction
│   │   ├── quality.py                  # Language identification & quality classification
│   │   ├── snapshot.py                 # Sharded Parquet writer with resume
│   │   └── fetcher.py                  # HTTP request execution & gzip writer
│   ├── indexing/                       # Sparse and dense index engines
│   │   ├── chunker.py                  # Structural span chunker
│   │   ├── lucene.py                   # Lucene / BM25 backend
│   │   ├── dense_encoder.py            # E5 embedding encoder with OOM retry
│   │   ├── embedding_store.py          # Float16 disk block vector store
│   │   └── faiss_index.py              # FAISS IVF-PQ with exact cosine refinement
│   ├── retrieval/                      # Search and hybrid fusion
│   │   ├── hybrid.py                   # Multilingual hybrid candidate retriever
│   │   └── evidence.py                 # Document-conditioned evidence span selector
│   ├── reranking/                      # Neural cross-encoders
│   │   └── neural.py                   # BGE reranker with subwindow aggregation
│   ├── evaluation/                     # Local metrics & output policies
│   │   ├── output_selector.py          # Variable-size output selection
│   │   └── metrics.py                  # Official Macro F2 evaluator
│   ├── submission/                     # Packaging and validators
│   │   └── validators.py               # Schema and provenance validator
│   └── utils/                          # Hardware inspection & configs
└── tests/                              # Comprehensive test suite (191+ tests)
    ├── data/
    ├── evaluation/
    ├── indexing/
    ├── ingestion/
    ├── integration/
    ├── operations/
    ├── reranking/
    └── submission/
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10+ (Python 3.12 recommended)
- Java 17+ (Required for PyPyserini / Lucene JVM bridge)
- CUDA-compatible GPU (e.g. RTX 3050 Ti 4GB or better)

### Setup Virtual Environment

```bash
# Clone the repository
git clone https://github.com/tmanhococ/R2AI_MedicalRetrieval.git
cd R2AI_MedicalRetrieval

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# or: .venv\Scripts\Activate.ps1  # On Windows PowerShell

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## 🚀 End-to-End Pipeline Runbook

### 1. Corpus Acquisition (Turbo Crawler)
Crawls the 4.39M official URLs from `links_corpus.parquet` into compressed raw payloads:
```bash
python -m src.cli.turbo_crawl \
    --input data/official/links_corpus.parquet \
    --registry artifacts/vibiomir/registry.sqlite \
    --raw-root artifacts/vibiomir/raw \
    --concurrency 600
```
- Failed or blocked URLs are automatically recorded into `artifacts/vibiomir/failed_urls.jsonl`.
- Resumable anytime; tracks progress atomically in SQLite.

### 2. Corpus Snapshot Freeze & Extraction
Extracts clean document bodies, assigns character span offsets, and writes sharded Parquet files:
```bash
python -u -m src.cli.build_snapshot \
    --name phase2_full \
    --config configs/vibiomir_phase2.yaml \
    --resume
```
> **Tip:** You can safely pause with `Ctrl+C` or turn off your machine. Rerunning with `--resume` seamlessly resumes from the last completed shard without duplicate processing.

### 3. Multilingual Sparse Indexing (Lucene / BM25s)
Builds disk-backed BM25 sparse indices across all extracted document and chunk texts:
```bash
python -m src.cli.index_sparse \
    --config configs/vibiomir_phase2.yaml \
    --snapshot artifacts/vibiomir/snapshots/phase2_full \
    --output-dir artifacts/indexes/phase2_sparse
```

### 4. Dense Encoding & Vector Store (E5-base + Float16 Blocks)
Encodes chunks using `intfloat/multilingual-e5-base` into float16 disk blocks:
```bash
python -m src.cli.encode_corpus \
    --config configs/vibiomir_phase2.yaml \
    --snapshot artifacts/vibiomir/snapshots/phase2_full \
    --output-dir artifacts/indexes/phase2_embeddings \
    --batch-size 8 \
    --resume
```

### 5. CPU FAISS Indexing & Refinement Audit
Builds the IVF-PQ FAISS index and audits retrieval retention against exact stored vectors:
```bash
python -m src.cli.index_dense \
    --config configs/vibiomir_phase2.yaml \
    --embeddings-dir artifacts/indexes/phase2_embeddings \
    --output-dir artifacts/indexes/phase2_dense
```

### 6. End-to-End Inference & Submission Packaging
Orchestrates candidate retrieval, evidence span expansion, BGE neural reranking, output selection, and exports the final submission ZIP:
```bash
python -m src.cli.predict \
    --config configs/vibiomir_phase2.yaml \
    --snapshot artifacts/vibiomir/snapshots/phase2_full \
    --sparse-index-dir artifacts/indexes/phase2_sparse \
    --dense-index-dir artifacts/indexes/phase2_dense \
    --export-submission \
    --output-dir artifacts/experiments/vibiomir_phase2_B \
    --resume
```

Output deliverables:
- `results.json`: JSON prediction file containing all 1,200 evaluation query IDs.
- `submission_B.zip`: Verified competition submission package.

---

## 🛡 Production Safeguards & Engineering Highlights

- **Hardware Capping (4GB VRAM Protection)**:  
  Sequential GPU memory management frees the E5 embedding encoder from VRAM (`torch.cuda.empty_cache()`) before loading the BGE cross-encoder. Adaptive batch backoff recovers cleanly from CUDA OOM.
- **Strict Anti-Leakage Enforced**:  
  Retriever, reranker, and threshold calibration logic are strictly forbidden from referencing ground-truth labels or test splits.
- **Full Traceability**:  
  Every predicted chunk preserves character offset provenance (`start_char`, `end_char`, `source_sha256`), verifiable by resolving back to the raw crawled payload.
- **Regression Protection**:  
  191 unit and integration tests verify every contract and edge case across all modules:
  ```bash
  pytest -q
  ```

---

## 📊 Offline Validation & Scoring Parity

Before submitting, validate the package locally:
```bash
python -m src.cli.validate_submission \
    --submission-zip artifacts/experiments/vibiomir_phase2_B/submission_B.zip \
    --queries data/official/query.parquet \
    --corpus-links data/official/links_corpus.parquet
```

The validator confirms:
1. Root-level `results.json` exists with no nested directories.
2. Exactly 1,200 unique query IDs matching official evaluation queries.
3. Relevant document IDs are signed 64-bit integers matching official corpus IDs.
4. Relevant chunks are exact substrings of their respective document texts.
5. All caps are adhered to ($\le 10$ documents, $\le 20$ chunks per query, $\le 3$ chunks per document).

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
