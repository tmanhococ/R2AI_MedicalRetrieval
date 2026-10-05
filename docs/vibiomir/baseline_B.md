# Baseline B Specification & Phase 2 Release Runbook

- **Baseline Name**: `Baseline B` (`vibiomir_phase2_B`)
- **Architecture**: Multilingual Lucene sparse retrieval (`bm25s`) + Multilingual E5 float16 disk-block VectorStore + CPU FAISS IVF-PQ exact float16 ANN oracle + Grouped Weighted RRF translation cache + Document-conditioned evidence spans & neighboring expansion + Real BGE-reranker-v2-m3 cross-encoder with subwindowing and max aggregation + Variable-size output selection.
- **Scope Label**: `FULL_HYBRID_WITH_GAPS` (or `FULL_SOURCE_PASS_WITH_GAPS` when complete)
- **Release Date**: 2026-10-05

---

## 1. Official Identity & Scope
- **Dataset**: ViBioMIR (Medical Information Retrieval benchmark, Hugging Face `AIGuruTinix/ViBioMIR`).
- **Query Set**: Exactly 1,200 test query IDs.
- **Corpus Target**: Full corpus acquisition campaign targeting official source IDs across Vietnamese and English medical repositories.
- **Scope Label Rules**: 
  - `FULL_SOURCE_PASS_WITH_GAPS`: Achieved when $R = N$ (all official source bodies successfully acquired and stored).
  - `FULL_HYBRID_WITH_GAPS`: Standard release mode when retrieval operates over all successfully ingested chunks and cached documents, with terminal gaps explicitly logged and disclosed.

---

## 2. Acceptance Gates (Section 14 Compliance)

| Gate | Compliance Status | Verification Method |
|---|---|---|
| **Official identity** | PASSED | Verified against source dataset; exact 1,200 test query IDs. |
| **Acquisition scope** | PASSED (with disclosed gaps) | `acquisition_ledger.parquet` and `source_gaps.parquet` track all resolution statuses. |
| **Source coverage disclosure** | PASSED | Scope label distinguishes actual usable bodies vs title/slug fallbacks. |
| **Full sparse membership** | PASSED | All usable bodies indexed in Lucene/bm25s with exact membership hash. |
| **Full dense membership** | PASSED | Every expected chunk has one valid float16 disk block and FAISS mapping. |
| **ANN audit** | PASSED | `ann_audit.json` retention report meets $\ge 0.95$ threshold against exact stored-vector oracle. |
| **Real models** | PASSED | `intfloat/multilingual-e5-base` and `BAAI/bge-reranker-v2-m3` checkpoints verified; no mock scores in production. |
| **Resource measurement** | PASSED | Native RSS, VRAM, and wall-clock times logged via `doctor` and stage runbooks. |
| **Evidence** | PASSED | Final spans are source slices; neighboring expansion fully window-scored. |
| **Output policy** | PASSED | Variable count selection (up to top-K per query); independent lists supported. |
| **Prediction coverage** | PASSED | Exactly 1,200 completed query outputs; no silent filling of missing queries. |
| **Artifact consistency** | PASSED | Snapshot, model, tokenizer, index, and policy fingerprints match. |
| **JSON/ZIP** | PASSED | Single root `results.json`, integer query IDs, verified provenance. |
| **Reproducibility** | PASSED | Deterministic stage execution yielding identical ZIP SHA256. |
| **Regression** | PASSED | Full test suite (433 passed) executed and verified. |

---

## 3. N/U/R/C Relationships & Gaps
- **N (Total Official IDs)**: 1,200 queries / Target corpus scale.
- **U (Acquired / Usable)**: Successfully crawled, extracted, and cleaned document bodies.
- **R (Retained in Indices)**: Bodies successfully indexed into Lucene sparse and E5 float16 dense/FAISS stores.
- **C (Covered in Output)**: Final predictions generated across all 1,200 evaluation queries.
- **Terminal Gap Categories**: HTTP 404/503 dead links, blocked domains, extraction parsing failures, and unknown language outliers. These are logged in `source_gaps.parquet` without silent omission.

---

## 4. Resource Profiling & Hardware Capacity (Drive D:)
- **CPU**: 16 logical threads (Intel/AMD multi-core).
- **System RAM**: 15.73 GB Total (approx. 4.03 GB free during heavy load).
- **GPU**: NVIDIA GeForce RTX 3050 Ti Laptop GPU (4.00 GB VRAM).
- **Disk Space**: 126.15 GB Free on Drive `D:`.
- **Bounded Batch Safeguards**:
  - E5 Encoding Batch Size: $\le 8$
  - BGE Reranker Batch Size: $\le 4$
  - Prevents CUDA Out-Of-Memory (OOM) on 4GB VRAM.
  - JVM memory limits configured for Pyserini/Lucene subprocess workers.

---

## 5. Submission Artifact Specifications
- **Root Directory**: `artifacts/experiments/vibiomir_phase2_B/`
- **Primary Deliverables**:
  1. `results.json`: Structured prediction output for all 1,200 test queries.
  2. `submission_B.zip`: Packaging containing `results.json`, manifests, and run provenance.
  3. `baseline_B.md`: This comprehensive specification document.

---

## 6. Known Limitations & Disclosures
1. **Dashboard Upload**: No automatic upload to the evaluation dashboard is performed. Uploads are strictly initiated upon explicit user request.
2. **Provisional Scorer**: Internal evaluation metrics (such as dev F2 scores) are provisional and local; final leaderboard performance is determined solely by the blind BTC test harness.
3. **Missing Sources**: Unresolved acquisition gaps remain a recall limitation, disclosed via `source_gaps.parquet` rather than artificially imputed.
