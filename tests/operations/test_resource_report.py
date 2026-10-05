from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data.phase2_contracts import ResourceReport
from src.operations.resources import (
    inspect_resources,
    measure_process_rss,
    measure_cuda_memory,
)


def test_resource_report_dataclass_fields():
    report = ResourceReport(
        cpu_threads=16,
        ram_total_bytes=16000000000,
        ram_free_bytes=8000000000,
        disk_free_bytes=100000000000,
        gpu_name="NVIDIA GeForce RTX 3050 Ti Laptop GPU",
        vram_bytes=4000000000,
        runtime_versions={"python": "3.12.10", "torch": "2.5.1"},
        measured_chunks_per_second=21.5,
        measured_pairs_per_second=75.0,
    )
    assert report.cpu_threads == 16
    assert report.ram_total_bytes == 16000000000
    assert report.ram_free_bytes == 8000000000
    assert report.disk_free_bytes == 100000000000
    assert report.gpu_name == "NVIDIA GeForce RTX 3050 Ti Laptop GPU"
    assert report.vram_bytes == 4000000000
    assert report.runtime_versions["python"] == "3.12.10"
    assert report.measured_chunks_per_second == 21.5
    assert report.measured_pairs_per_second == 75.0


def test_inspect_resources_real_system(tmp_path: Path):
    report = inspect_resources(workdir=tmp_path)
    assert isinstance(report, ResourceReport)
    assert report.cpu_threads >= 1
    assert report.ram_total_bytes > 0
    assert report.ram_free_bytes > 0
    assert report.disk_free_bytes > 0
    assert isinstance(report.runtime_versions, dict)
    assert "python" in report.runtime_versions
    assert "psutil" in report.runtime_versions


def test_inspect_resources_without_cuda(tmp_path: Path):
    with patch("torch.cuda.is_available", return_value=False):
        report = inspect_resources(workdir=tmp_path)
        assert report.gpu_name is None
        assert report.vram_bytes == 0


def test_inspect_resources_with_mocked_cuda(tmp_path: Path):
    mock_props = MagicMock()
    mock_props.total_memory = 8 * 1024**3

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 4090"), \
         patch("torch.cuda.get_device_properties", return_value=mock_props):
        report = inspect_resources(workdir=tmp_path)
        assert report.gpu_name == "NVIDIA RTX 4090"
        assert report.vram_bytes == 8 * 1024**3


def test_inspect_resources_loads_cached_benchmarks(tmp_path: Path):
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    bench_file = artifacts_dir / "smoke_benchmark_results.json"
    bench_data = {
        "e5": {"chunks_per_sec": 42.5},
        "bge": {"pairs_per_sec": 88.0},
    }
    with open(bench_file, "w") as f:
        json.dump(bench_data, f)

    report = inspect_resources(workdir=tmp_path)
    assert report.measured_chunks_per_second == pytest.approx(42.5)
    assert report.measured_pairs_per_second == pytest.approx(88.0)


def test_measure_process_rss_uses_native_psutil():
    rss = measure_process_rss()
    assert isinstance(rss, int)
    # Process RSS must be at least 10 MB for a running Python interpreter with pytest
    assert rss > 10 * 1024 * 1024


def test_measure_cuda_memory():
    allocated, reserved = measure_cuda_memory()
    assert isinstance(allocated, int)
    assert isinstance(reserved, int)
    assert allocated >= 0
    assert reserved >= 0


def test_doctor_cli_text_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from src.cli.doctor import main
    rc = main(["--workdir", str(tmp_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "ViBioMIR Phase 2 System & Hardware Diagnostics" in captured.out
    assert "CPU Logical Threads" in captured.out
    assert "Process Native RSS" in captured.out


def test_doctor_cli_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    from src.cli.doctor import main
    rc = main(["--workdir", str(tmp_path), "--json"])
    assert rc == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "cpu_threads" in data
    assert "ram_total_bytes" in data
    assert "process_rss_bytes" in data
    assert "runtime_versions" in data


def test_doctor_cli_output_file(tmp_path: Path):
    from src.cli.doctor import main
    out_file = tmp_path / "doctor_out.json"
    rc = main(["--workdir", str(tmp_path), "--output", str(out_file)])
    assert rc == 0
    assert out_file.exists()
    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "cpu_threads" in data
    assert data["cpu_threads"] >= 1


def test_doctor_check_health():
    from src.cli.doctor import check_health

    # Test report with low RAM and small VRAM
    report = ResourceReport(
        cpu_threads=4,
        ram_total_bytes=8 * 1024**3,  # 8 GB
        ram_free_bytes=2 * 1024**3,
        disk_free_bytes=10 * 1024**3,  # 10 GB
        gpu_name="NVIDIA GTX 1650",
        vram_bytes=4 * 1024**3,       # 4 GB
        runtime_versions={"python": "3.12.10", "java": "not installed", "pyserini": "not installed"},
    )
    warnings = check_health(report)
    assert any("Low system RAM" in w for w in warnings)
    assert any("Low disk space" in w for w in warnings)
    assert any("GPU VRAM is 4.0 GB" in w for w in warnings)
    assert any("Java runtime not detected" in w for w in warnings)
    assert any("Pyserini is not installed" in w for w in warnings)

