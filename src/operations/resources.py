from __future__ import annotations

import json
import logging
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any

import psutil

from src.data.phase2_contracts import ResourceReport

logger = logging.getLogger(__name__)


def measure_process_rss(pid: int | None = None) -> int:
    """Measure resident set size (RSS) in bytes for a process.

    Uses OS-level psutil to capture native C/C++ memory (PyTorch, FAISS, JVM),
    NOT Python-only tracemalloc.
    """
    proc = psutil.Process(pid)
    return int(proc.memory_info().rss)


def measure_process_tree_rss(pid: int | None = None) -> int:
    """Measure total RSS in bytes for the process and all child processes (e.g. JVM workers)."""
    parent = psutil.Process(pid)
    total_rss = int(parent.memory_info().rss)
    try:
        children = parent.children(recursive=True)
        for child in children:
            try:
                total_rss += int(child.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return total_rss


def measure_cuda_memory(device_idx: int = 0) -> tuple[int, int]:
    """Measure current CUDA allocated and reserved memory in bytes.

    Returns:
        (allocated_bytes, reserved_bytes). Returns (0, 0) if CUDA is unavailable.
    """
    try:
        import torch
        if torch.cuda.is_available():
            allocated = int(torch.cuda.memory_allocated(device_idx))
            reserved = int(torch.cuda.memory_reserved(device_idx))
            return allocated, reserved
    except Exception as exc:
        logger.debug("Failed to query CUDA memory: %s", exc)
    return 0, 0


def detect_java_version() -> str:
    """Detect Java runtime version by probing java executable or JAVA_HOME."""
    java_bin = shutil.which("java")
    if not java_bin:
        java_home = Path("C:/Program Files/Java/jdk-21/bin/java.exe")
        if java_home.exists():
            java_bin = str(java_home)

    if not java_bin:
        return "not installed"

    try:
        proc = subprocess.run(
            [java_bin, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        output = proc.stderr or proc.stdout
        first_line = output.strip().splitlines()[0] if output else ""
        return first_line.strip() if first_line else "unknown"
    except Exception as exc:
        logger.debug("Failed to query java version: %s", exc)
        return "not available"


def get_runtime_versions() -> dict[str, str]:
    """Collect runtime and package version information."""
    versions: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "psutil": psutil.__version__,
    }

    # PyTorch and CUDA
    try:
        import torch
        versions["torch"] = torch.__version__
        versions["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["torch_cuda_version"] = str(torch.version.cuda)
            versions["gpu_device_name"] = str(torch.cuda.get_device_name(0))
    except ImportError:
        versions["torch"] = "not installed"

    # Transformers
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = "not installed"

    # Sentence Transformers
    try:
        import sentence_transformers
        versions["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        versions["sentence_transformers"] = "not installed"

    # FAISS
    try:
        import faiss
        versions["faiss"] = getattr(faiss, "__version__", "installed")
    except ImportError:
        versions["faiss"] = "not installed"

    # PyArrow
    try:
        import pyarrow
        versions["pyarrow"] = pyarrow.__version__
    except ImportError:
        versions["pyarrow"] = "not installed"

    # BM25S
    try:
        import bm25s
        versions["bm25s"] = getattr(bm25s, "__version__", "installed")
    except ImportError:
        versions["bm25s"] = "not installed"

    # Java
    versions["java"] = detect_java_version()

    # Pyserini
    try:
        import pyserini
        versions["pyserini"] = getattr(pyserini, "__version__", "installed")
    except ImportError:
        versions["pyserini"] = "not installed"

    return versions


def inspect_resources(workdir: Path = Path(".")) -> ResourceReport:
    """Inspect system hardware, memory, disk, and runtime environments."""
    workdir_path = Path(workdir)

    cpu_threads = psutil.cpu_count(logical=True) or 1
    vm = psutil.virtual_memory()
    ram_total = int(vm.total)
    ram_free = int(vm.available)

    target_disk = workdir_path if workdir_path.exists() else Path(".")
    disk_usage = shutil.disk_usage(target_disk)
    disk_free = int(disk_usage.free)

    gpu_name: str | None = None
    vram_bytes: int = 0

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_bytes = int(getattr(props, "total_memory", 0))
    except Exception as exc:
        logger.debug("Failed to inspect CUDA properties: %s", exc)

    runtime_versions = get_runtime_versions()

    # Load measured benchmarks if available
    measured_chunks_per_sec: float | None = None
    measured_pairs_per_sec: float | None = None

    possible_bench_paths = [
        workdir_path / "artifacts" / "smoke_benchmark_results.json",
        workdir_path / "smoke_benchmark_results.json",
        Path("artifacts") / "smoke_benchmark_results.json",
    ]

    for p in possible_bench_paths:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    bench_data = json.load(f)
                measured_chunks_per_sec = float(bench_data.get("e5", {}).get("chunks_per_sec", 0.0)) or None
                measured_pairs_per_sec = float(bench_data.get("bge", {}).get("pairs_per_sec", 0.0)) or None
                if measured_chunks_per_sec is not None or measured_pairs_per_sec is not None:
                    break
            except Exception as exc:
                logger.debug("Failed reading benchmark results from %s: %s", p, exc)

    return ResourceReport(
        cpu_threads=cpu_threads,
        ram_total_bytes=ram_total,
        ram_free_bytes=ram_free,
        disk_free_bytes=disk_free,
        gpu_name=gpu_name,
        vram_bytes=vram_bytes,
        runtime_versions=runtime_versions,
        measured_chunks_per_second=measured_chunks_per_sec,
        measured_pairs_per_second=measured_pairs_per_sec,
    )
