from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

from src.data.source_store import SnapshotSourceStore
from src.data.vibiomir_adapter import load_official_queries
from src.data.vibiomir_contracts import ValidationReport
from src.submission.validators import validate_zip
from src.utils.config import load_config


def run_validate_submission(
    snapshot_dir: Path | str,
    zip_path: Path | str,
    query_path: Path | str,
    expected_count: int | None = None,
    output_report_file: Path | str | None = None,
) -> ValidationReport:
    """Validates submission ZIP file against official queries and frozen snapshot."""
    snap_p = Path(snapshot_dir)
    zip_p = Path(zip_path)
    q_p = Path(query_path)

    if not snap_p.exists():
        raise FileNotFoundError(f"Snapshot directory not found: {snap_p}")
    if not zip_p.exists():
        raise FileNotFoundError(f"Submission ZIP file not found: {zip_p}")
    if not q_p.exists():
        raise FileNotFoundError(f"Query file not found: {q_p}")

    store = SnapshotSourceStore(snap_p)
    queries = load_official_queries(q_p, expected_count=expected_count)

    report = validate_zip(zip_p, queries=queries, store=store, expected_count=expected_count)

    report_data = {
        "valid": report.valid,
        "query_count": report.query_count,
        "document_count": report.document_count,
        "chunk_count": report.chunk_count,
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "snapshot_id": report.snapshot_id,
    }

    out_file = Path(output_report_file) if output_report_file else zip_p.parent / "validation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    return report


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Offline validation of official ViBioMIR submission ZIP archive"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="Path to frozen snapshot directory",
    )
    parser.add_argument(
        "--zip",
        type=Path,
        required=True,
        help="Path to submission ZIP archive",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        default=None,
        help="Path to query.parquet",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Path to output validation_report.json",
    )

    parsed = parser.parse_args(args)

    cfg: dict[str, Any] = {}
    config_file = parsed.config
    if config_file is None and Path("configs/vibiomir_phase1.yaml").exists():
        config_file = Path("configs/vibiomir_phase1.yaml")

    if config_file is not None and Path(config_file).exists():
        cfg = load_config(config_file)

    ds_cfg = cfg.get("dataset", {})
    query_p = parsed.query_path or Path(ds_cfg.get("query_path", "data/official/query.parquet"))
    expected_count = ds_cfg.get("expected_query_count", 1200)

    print("=== ViBioMIR Submission Validation ===")
    print(f"Snapshot:  {parsed.snapshot}")
    print(f"ZIP file:  {parsed.zip}")
    print(f"Queries:   {query_p}")

    report = run_validate_submission(
        snapshot_dir=parsed.snapshot,
        zip_path=parsed.zip,
        query_path=query_p,
        expected_count=expected_count,
        output_report_file=parsed.output_report,
    )

    print("\n=== Validation Report ===")
    print(f"Valid:           {report.valid}")
    print(f"Query Count:     {report.query_count}")
    print(f"Document Count:  {report.document_count}")
    print(f"Chunk Count:     {report.chunk_count}")
    print(f"Errors:          {len(report.errors)}")
    print(f"Warnings:        {len(report.warnings)}")

    if report.errors:
        print("\nErrors encountered:")
        for err in report.errors[:10]:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print("\nAll checks passed successfully!")


if __name__ == "__main__":
    main()
