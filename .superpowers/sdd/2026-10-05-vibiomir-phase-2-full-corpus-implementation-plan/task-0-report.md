# Task 0 Report: Audit Baseline và Benchmark Hardware Thật

**Date:** 2026-10-05  
**Branch:** `feature/vibiomir-phase-2`  
**Commit:** `a523e85 feat: add Phase 2 hardware and runtime diagnostics`  
**Status:** DONE

---

## 1. Executive Summary

Task 0 audited the ViBioMIR Phase 1 baseline artifacts and codebase, executed Phase 1 regression tests, benchmarked the host hardware and runtime environment with real models (`intfloat/multilingual-e5-base` and `BAAI/bge-reranker-v2-m3`), verified the Pyserini/Java 21 environment constraints, and implemented the Phase 2 diagnostic suite (`ResourceReport`, `inspect_resources`, and `src/cli/doctor.py`).

Key findings:
- **Baseline Phase 1 regression:** Re-run passed completely (359 tests passed).
- **Host Hardware:** Intel Core i7 / AMD CPU with 16 logical threads, 15.73 GB RAM (~8.74 GB free), 129.50 GB free disk space on drive `D:`, and NVIDIA GeForce RTX 3050 Ti Laptop GPU with 4.00 GB VRAM.
- **Real Model Smoke Benchmark:**
  - `intfloat/multilingual-e5-base` (rev: `d128750597153bb5987e10b1c3493a34e5a4502a`): 21.58 chunks/sec (batch=8), peak CUDA allocation 1070.6 MB, std dev 0.036084 (>0, clean semantic variance, unit norm 1.0).
  - `BAAI/bge-reranker-v2-m3` (rev: `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`): 75.07 pairs/sec (batch=4), peak CUDA allocation 2175.0 MB, std dev 2.992158 (non-constant, raw logits range `[-1.23, 6.19]`).
  - Both models fit comfortably within the 4.00 GB VRAM when loaded sequentially (one model at a time).
- **Pyserini / Java 21 status:**
  - Java 21 (`java version "21.0.2" 2024-01-16 LTS`) is installed at `C:\Program Files\Java\jdk-21`.
  - Pyserini dry-run revealed that direct `pip install pyserini` attempts to pull and overwrite `torch` with `torch-2.14.1` (CPU), which would break the host PyTorch 2.5.1+cu121 CUDA environment. Therefore, Pyserini must run via an isolated subprocess worker with a dedicated venv and JSONL protocol as specified in the plan.
- **Diagnostics Implementation:** Implemented `ResourceReport` in `src/data/phase2_contracts.py`, `inspect_resources` in `src/operations/resources.py`, and `src/cli/doctor.py` measuring process native RSS via `psutil`, CUDA allocation/reservation, and runtime versions.

---

## 2. What Was Implemented

1. **`src/data/phase2_contracts.py`:**
   - `ResourceReport` dataclass matching exact specification:
     - `cpu_threads: int`
     - `ram_total_bytes: int`
     - `ram_free_bytes: int`
     - `disk_free_bytes: int`
     - `gpu_name: str | None`
     - `vram_bytes: int`
     - `runtime_versions: dict[str, str]`
     - `measured_chunks_per_second: float | None`
     - `measured_pairs_per_second: float | None`
   - Additional Phase 2 contracts: `CoverageLedger`, `EmbeddingBlock`, `EvidenceSpan`, `StageManifest`, `StageMeasurement`, and configuration dataclasses (`FullCampaignConfig`, `SparseConfig`, `DenseConfig`, `EvidenceConfig`, `RerankConfig`, `OutputPolicy`).

2. **`src/operations/resources.py`:**
   - `measure_process_rss(pid=None) -> int`: Measures resident set size (RSS) via `psutil.Process`, tracking native C/C++ memory (PyTorch, FAISS, JVM) rather than tracemalloc.
   - `measure_process_tree_rss(pid=None) -> int`: Sums process and all child process RSS (tracking separate JVM workers).
   - `measure_cuda_memory() -> tuple[int, int]`: Safe query of `torch.cuda.memory_allocated()` and `torch.cuda.memory_reserved()`.
   - `detect_java_version() -> str`: Probes java executable or `C:/Program Files/Java/jdk-21/bin/java.exe`.
   - `get_runtime_versions() -> dict[str, str]`: Collects installed versions of Python, platform, psutil, torch, CUDA, transformers, sentence_transformers, faiss, pyarrow, bm25s, java, and pyserini.
   - `inspect_resources(workdir=Path(".")) -> ResourceReport`: Collects full hardware and runtime report; automatically attaches benchmark measurements from `artifacts/smoke_benchmark_results.json` if present; handles missing CUDA gracefully.

3. **`src/cli/doctor.py`:**
   - Full CLI diagnostic tool callable via `python -m src.cli.doctor [--workdir ...] [--config ...] [--json] [--benchmark] [--output ...]`.
   - Formats human-readable diagnostic report table with resource metrics, runtime versions, benchmark throughputs, and health warnings.
   - Warns when VRAM < 6 GB (mandates bounded batch size <=8 for E5 and <=4 for BGE).
   - Warns about Pyserini environment isolation.
   - Supports automated output saving to `artifacts/vibiomir/doctor_report.json`.

---

## 3. TDD Evidence

### RED Phase
- **Command:** `pytest tests/operations/test_resource_report.py -q`
- **Output:**
  ```
  =================================== ERRORS ====================================
  __________ ERROR collecting tests/operations/test_resource_report.py __________
  ImportError while importing test module 'D:\Research\R2AI\mvp-rag-medical\tests\operations\test_resource_report.py'.
  Traceback:
  tests\operations\test_resource_report.py:9: in <module>
      from src.data.phase2_contracts import ResourceReport
  E   ModuleNotFoundError: No module named 'src.data.phase2_contracts'
  1 error in 0.28s
  ```
- **Why expected:** `src/data/phase2_contracts.py` and `src/operations/resources.py` had not been implemented yet.

### GREEN Phase
- **Command:** `pytest tests/operations/test_resource_report.py -q`
- **Output:**
  ```
  ...........                                                              [100%]
  11 passed, 3 warnings in 19.49s
  ```
- **Why expected:** All 11 unit tests passed, covering dataclass fields, real system inspection, mocked non-CUDA fallback, mocked CUDA properties, cached benchmark loading, native RSS via psutil, CUDA memory measurement, CLI text output, CLI JSON output, CLI custom output file, and health warning triggers.

### Full Regression Check
- **Command:** `pytest -q`
- **Output:**
  ```
  ........................................................................ [ 19%]
  ........................................................................ [ 38%]
  ........................................................................ [ 58%]
  ........................................................................ [ 77%]
  ........................................................................ [ 97%]
  ..........                                                               [100%]
  370 passed, 3 warnings in 48.27s
  ```
- All 359 Phase 1 tests + 11 new Phase 2 tests passed without any regression.

---

## 4. Hardware & Runtime Diagnostics Summary

Output from `python -m src.cli.doctor`:
```
======================================================================
          ViBioMIR Phase 2 System & Hardware Diagnostics           
======================================================================
CPU Logical Threads : 16
System RAM Total    : 15.73 GB
System RAM Free     : 8.74 GB
Disk Space Free     : 129.50 GB
----------------------------------------------------------------------
GPU Device Name     : NVIDIA GeForce RTX 3050 Ti Laptop GPU
GPU Total VRAM      : 4.00 GB
CUDA Allocated      : 0.0 MB
CUDA Reserved       : 0.0 MB
----------------------------------------------------------------------
Process Native RSS  : 1634.9 MB (psutil, OS native)
Process Tree RSS    : 1634.9 MB (including child/JVM workers)
----------------------------------------------------------------------
Runtime Versions:
  - bm25s                 : 0.3.11
  - cuda_available        : True
  - faiss                 : 1.11.0
  - gpu_device_name       : NVIDIA GeForce RTX 3050 Ti Laptop GPU
  - java                  : java version "21.0.2" 2024-01-16 LTS
  - platform              : Windows-11-10.0.26200-SP0
  - psutil                : 5.9.8
  - pyarrow               : 25.0.0
  - pyserini              : not installed
  - python                : 3.12.10
  - sentence_transformers : 5.0.0
  - torch                 : 2.5.1+cu121
  - torch_cuda_version    : 12.1
  - transformers          : 4.53.1
----------------------------------------------------------------------
Smoke Benchmark Measurements:
  - E5-base throughput   : 21.58 chunks/sec
  - BGE-reranker throughput: 75.07 pairs/sec
----------------------------------------------------------------------
ATTENTION / WARNINGS:
  [!] GPU VRAM is 4.0 GB (<6 GB). Bounded batches required: E5 batch <= 8, BGE batch <= 4 to prevent OOM.
  [!] Pyserini is not installed in the main environment. Use an isolated subprocess worker with Java 21 to avoid dependency conflicts with PyTorch.
======================================================================
```

---

## 5. Model Smoke Benchmark Details

Saved in `artifacts/smoke_benchmark_results.json`:
```json
{
  "device": "cuda",
  "e5": {
    "model_id": "intfloat/multilingual-e5-base",
    "revision": "d128750597153bb5987e10b1c3493a34e5a4502a",
    "load_time_s": 108.42,
    "chunks_per_sec": 21.58,
    "std_dev": 0.036084,
    "cuda_alloc_mb": 1070.6,
    "cuda_res_mb": 1124.0,
    "rss_mb": 4510.2
  },
  "bge": {
    "model_id": "BAAI/bge-reranker-v2-m3",
    "revision": "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    "load_time_s": 247.7,
    "pairs_per_sec": 75.07,
    "std_dev": 2.992158,
    "cuda_alloc_mb": 2175.0,
    "cuda_res_mb": 2214.0,
    "rss_mb": 4673.4
  }
}
```

- **E5-base**:
  - Encoded 32 sample medical passages in 1.48s at batch size 8.
  - L2 normalized embeddings (shape `32, 768`), norms = 1.0.
  - Standard deviation = 0.036084, cosine similarity of diagonal = 1.0, off-diagonal = [0.797, 0.992].
  - Peak CUDA allocation: 1070.6 MB (well within 4.00 GB limit).
- **BGE-reranker-v2-m3**:
  - Scored 32 sample query-passage pairs in 0.43s at batch size 4.
  - Raw logits range: `[-1.23, 6.19]`, std dev = 2.992158.
  - Peak CUDA allocation: 2175.0 MB (well within 4.00 GB limit).

---

## 6. Files Changed / Added

- `src/data/phase2_contracts.py`: Dataclasses for Phase 2 contracts and configurations.
- `src/operations/__init__.py`: Operations package init.
- `src/operations/resources.py`: System resource inspection and native process memory metrics.
- `src/cli/doctor.py`: Doctor CLI tool and health checking.
- `tests/operations/__init__.py`: Test package init.
- `tests/operations/test_resource_report.py`: 11 unit tests for resource reporting and doctor CLI.
- `artifacts/smoke_benchmark_results.json`: Benchmark cache of real model throughput and memory metrics.
- `artifacts/vibiomir/doctor_report.json`: JSON output of system diagnostic health.

---

## 7. Self-Review & Operational Insights for Next Tasks

1. **GPU Batch Sizes:** VRAM is 4 GB. Batch sizes in `configs/vibiomir_phase2.yaml` must not exceed `embedding_start_batch: 8` and `reranker_start_batch: 4`. OOM retry logic (8 -> 4 -> 2 -> 1) is mandatory.
2. **Never load E5 and BGE simultaneously:** Peak memory of E5 is ~1.1 GB and BGE is ~2.2 GB. Loading both concurrently on GPU would risk exhausting 4.0 GB VRAM. Dense encoding and reranking must run in separated stages with explicit CUDA cache clearing (`torch.cuda.empty_cache(); gc.collect()`).
3. **Pyserini Subprocess Worker:** As verified in pip dry-run, `pyserini` has dependency requirements that would downgrade or break PyTorch CUDA 12.1 in the core environment. Task 3/6 must use an isolated virtual environment (`.pyserini_venv`) and communicate via JSONL/subprocess IPC to index and query Lucene.
4. **Disk Space:** Free disk space is 129.50 GB. Sharded float16 embeddings and index storage must be monitored closely to ensure sufficient headroom during full acquisition.
