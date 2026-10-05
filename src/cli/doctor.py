from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

from src.data.phase2_contracts import ResourceReport
from src.operations.resources import (
    detect_java_version,
    get_runtime_versions,
    inspect_resources,
    measure_cuda_memory,
    measure_process_rss,
    measure_process_tree_rss,
)

logger = logging.getLogger(__name__)


def run_smoke_benchmark(
    device: str = "auto",
    e5_batch_size: int = 8,
    bge_batch_size: int = 4,
) -> dict[str, Any]:
    """Run real model smoke benchmark on E5-base and BGE-reranker-v2-m3.

    Encodes 32 sample passages with E5-base and scores 32 pairs with BGE-reranker.
    Logs output variance, throughput, and peak memory.
    """
    import gc
    import numpy as np
    import torch
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

    if device == "auto":
        target_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        target_device = device

    sample_passages = [
        f"passage: Bệnh nhân {i} có triệu chứng sốt cao, ho khan và khó thở kéo dài 3 ngày. Chẩn đoán sơ bộ viêm phổi cấp tính."
        if i % 2 == 0 else
        f"passage: Patient {i} presents with acute chest pain radiating to the left arm and elevated troponin levels indicating myocardial infarction."
        for i in range(32)
    ]

    sample_pairs = [
        (f"Triệu chứng viêm phổi cấp tính ở bệnh nhân {i}", sample_passages[i])
        for i in range(32)
    ]

    results: dict[str, Any] = {"device": target_device}

    # 1. E5-base benchmark
    e5_id = "intfloat/multilingual-e5-base"
    logger.info("Loading %s on %s...", e5_id, target_device)
    t0 = time.perf_counter()
    e5_tokenizer = AutoTokenizer.from_pretrained(e5_id)
    e5_model = AutoModel.from_pretrained(e5_id).to(target_device)
    e5_model.eval()
    e5_load_time = time.perf_counter() - t0

    t_encode_start = time.perf_counter()
    embeds = []
    with torch.no_grad():
        for start_idx in range(0, len(sample_passages), e5_batch_size):
            batch = sample_passages[start_idx:start_idx + e5_batch_size]
            encoded = e5_tokenizer(
                batch, max_length=512, padding=True, truncation=True, return_tensors="pt"
            ).to(target_device)
            out = e5_model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
            sum_emb = torch.sum(out.last_hidden_state * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            normed = torch.nn.functional.normalize(sum_emb / sum_mask, p=2, dim=1)
            embeds.append(normed.cpu().numpy())

    e5_total_time = time.perf_counter() - t_encode_start
    all_embeds = np.vstack(embeds)
    e5_chunks_per_sec = len(sample_passages) / e5_total_time
    e5_std_dev = float(np.std(all_embeds))
    e5_alloc, e5_res = measure_cuda_memory()
    e5_rss = measure_process_rss()

    results["e5"] = {
        "model_id": e5_id,
        "revision": getattr(e5_model.config, "_commit_hash", None),
        "load_time_s": round(e5_load_time, 2),
        "chunks_per_sec": round(e5_chunks_per_sec, 2),
        "std_dev": round(e5_std_dev, 6),
        "cuda_alloc_mb": round(e5_alloc / 1024**2, 1),
        "cuda_res_mb": round(e5_res / 1024**2, 1),
        "rss_mb": round(e5_rss / 1024**2, 1),
    }

    del e5_model, e5_tokenizer, encoded, out, embeds, all_embeds
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # 2. BGE-reranker benchmark
    bge_id = "BAAI/bge-reranker-v2-m3"
    logger.info("Loading %s on %s...", bge_id, target_device)
    t0 = time.perf_counter()
    bge_tokenizer = AutoTokenizer.from_pretrained(bge_id)
    bge_model = AutoModelForSequenceClassification.from_pretrained(bge_id).to(target_device)
    bge_model.eval()
    bge_load_time = time.perf_counter() - t0

    t_score_start = time.perf_counter()
    scores = []
    with torch.no_grad():
        for start_idx in range(0, len(sample_pairs), bge_batch_size):
            batch = sample_pairs[start_idx:start_idx + bge_batch_size]
            encoded = bge_tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(target_device)
            out = bge_model(**encoded)
            scores.extend(out.logits.view(-1).cpu().float().numpy().tolist())

    bge_total_time = time.perf_counter() - t_score_start
    all_scores = np.array(scores)
    bge_pairs_per_sec = len(sample_pairs) / bge_total_time
    bge_std_dev = float(np.std(all_scores))
    bge_alloc, bge_res = measure_cuda_memory()
    bge_rss = measure_process_rss()

    results["bge"] = {
        "model_id": bge_id,
        "revision": getattr(bge_model.config, "_commit_hash", None),
        "load_time_s": round(bge_load_time, 2),
        "pairs_per_sec": round(bge_pairs_per_sec, 2),
        "std_dev": round(bge_std_dev, 6),
        "cuda_alloc_mb": round(bge_alloc / 1024**2, 1),
        "cuda_res_mb": round(bge_res / 1024**2, 1),
        "rss_mb": round(bge_rss / 1024**2, 1),
    }

    del bge_model, bge_tokenizer, encoded, out, scores, all_scores
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return results


def check_health(report: ResourceReport) -> list[str]:
    """Identify potential resource risks or bottlenecks."""
    warnings: list[str] = []

    # RAM check
    ram_gb = report.ram_total_bytes / (1024**3)
    if ram_gb < 15.0:
        warnings.append(f"Low system RAM: {ram_gb:.1f} GB (>=16 GB recommended, 32 GB optimal).")

    # Disk check
    disk_free_gb = report.disk_free_bytes / (1024**3)
    if disk_free_gb < 30.0:
        warnings.append(f"Low disk space: {disk_free_gb:.1f} GB free (<30 GB may risk full corpus index build).")

    # GPU check
    if report.gpu_name is None:
        warnings.append("No CUDA GPU detected. Models will run on CPU, which is slow for full corpus.")
    else:
        vram_gb = report.vram_bytes / (1024**3)
        if vram_gb < 6.0:
            warnings.append(
                f"GPU VRAM is {vram_gb:.1f} GB (<6 GB). Bounded batches required: "
                "E5 batch <= 8, BGE batch <= 4 to prevent OOM."
            )

    # Java check
    java_ver = report.runtime_versions.get("java", "")
    if "not" in java_ver or not java_ver:
        warnings.append("Java runtime not detected. Required for Lucene/Pyserini sparse index worker.")

    # Pyserini check
    pyserini_ver = report.runtime_versions.get("pyserini", "")
    if pyserini_ver == "not installed":
        warnings.append(
            "Pyserini is not installed in the main environment. Use an isolated subprocess worker "
            "with Java 21 to avoid dependency conflicts with PyTorch."
        )

    return warnings


def format_report(
    report: ResourceReport,
    process_rss: int,
    tree_rss: int,
    cuda_alloc: int,
    cuda_res: int,
    warnings: list[str],
) -> str:
    """Format diagnostic report into human-readable text."""
    lines = [
        "=" * 70,
        "          ViBioMIR Phase 2 System & Hardware Diagnostics           ",
        "=" * 70,
        f"CPU Logical Threads : {report.cpu_threads}",
        f"System RAM Total    : {report.ram_total_bytes / (1024**3):.2f} GB",
        f"System RAM Free     : {report.ram_free_bytes / (1024**3):.2f} GB",
        f"Disk Space Free     : {report.disk_free_bytes / (1024**3):.2f} GB",
        "-" * 70,
        f"GPU Device Name     : {report.gpu_name or 'None (CPU only)'}",
        f"GPU Total VRAM      : {report.vram_bytes / (1024**3):.2f} GB" if report.vram_bytes else "GPU Total VRAM      : 0.00 GB",
        f"CUDA Allocated      : {cuda_alloc / (1024**2):.1f} MB",
        f"CUDA Reserved       : {cuda_res / (1024**2):.1f} MB",
        "-" * 70,
        f"Process Native RSS  : {process_rss / (1024**2):.1f} MB (psutil, OS native)",
        f"Process Tree RSS    : {tree_rss / (1024**2):.1f} MB (including child/JVM workers)",
        "-" * 70,
        "Runtime Versions:",
    ]

    for k, v in sorted(report.runtime_versions.items()):
        lines.append(f"  - {k:<22}: {v}")

    lines.append("-" * 70)
    lines.append("Smoke Benchmark Measurements:")
    if report.measured_chunks_per_second is not None:
        lines.append(f"  - E5-base throughput   : {report.measured_chunks_per_second:.2f} chunks/sec")
    else:
        lines.append("  - E5-base throughput   : Not measured (run with --benchmark to test)")

    if report.measured_pairs_per_second is not None:
        lines.append(f"  - BGE-reranker throughput: {report.measured_pairs_per_second:.2f} pairs/sec")
    else:
        lines.append("  - BGE-reranker throughput: Not measured (run with --benchmark to test)")

    lines.append("-" * 70)
    if warnings:
        lines.append("ATTENTION / WARNINGS:")
        for w in warnings:
            lines.append(f"  [!] {w}")
    else:
        lines.append("Status: All critical system health checks passed.")
    lines.append("=" * 70)

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ViBioMIR Phase 2 Hardware & Runtime Diagnostics")
    parser.add_argument("--config", type=str, default=None, help="Path to config file (optional)")
    parser.add_argument("--workdir", type=str, default=".", help="Working directory (default: .)")
    parser.add_argument("--json", action="store_true", help="Output JSON format instead of table")
    parser.add_argument("--benchmark", action="store_true", help="Execute live smoke benchmark on E5 & BGE")
    parser.add_argument("--output", type=str, default=None, help="Save report JSON to this path")

    args = parser.parse_args(argv)

    workdir = Path(args.workdir)

    if args.benchmark:
        print("Running live smoke benchmark on real E5-base and BGE models...", file=sys.stderr)
        bench_results = run_smoke_benchmark()
        artifacts_dir = workdir / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        bench_file = artifacts_dir / "smoke_benchmark_results.json"
        with open(bench_file, "w", encoding="utf-8") as f:
            json.dump(bench_results, f, indent=2)

    report = inspect_resources(workdir=workdir)
    proc_rss = measure_process_rss()
    tree_rss = measure_process_tree_rss()
    cuda_alloc, cuda_res = measure_cuda_memory()
    warnings = check_health(report)

    report_dict = report.to_dict()
    report_dict["process_rss_bytes"] = proc_rss
    report_dict["process_rss_mb"] = round(proc_rss / (1024**2), 1)
    report_dict["process_tree_rss_bytes"] = tree_rss
    report_dict["process_tree_rss_mb"] = round(tree_rss / (1024**2), 1)
    report_dict["cuda_allocated_bytes"] = cuda_alloc
    report_dict["cuda_allocated_mb"] = round(cuda_alloc / (1024**2), 1)
    report_dict["cuda_reserved_bytes"] = cuda_res
    report_dict["cuda_reserved_mb"] = round(cuda_res / (1024**2), 1)
    report_dict["health_warnings"] = warnings
    report_dict["healthy"] = len(warnings) == 0

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

    # Also automatically save to artifacts/vibiomir/doctor_report.json if artifacts directory exists
    auto_out = workdir / "artifacts" / "vibiomir" / "doctor_report.json"
    if auto_out.parent.exists():
        with open(auto_out, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

    if args.json:
        print(json.dumps(report_dict, indent=2))
    else:
        text = format_report(report, proc_rss, tree_rss, cuda_alloc, cuda_res, warnings)
        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
