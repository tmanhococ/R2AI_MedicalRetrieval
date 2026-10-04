from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
from pathlib import Path
import sys
import time
from typing import Any

from src.data.vibiomir_contracts import ExtractionConfig, ExtractionReport
from src.data.vibiomir_registry import SourceRegistry
from src.ingestion.extractors import extract_document
from src.ingestion.quality import (
    FastTextLanguageDetector,
    LanguageDetector,
    classify_quality,
)
from src.ingestion.snapshot import FallbackLanguageDetector
from src.utils.config import load_config


def run_extract(
    registry_path: Path | str,
    raw_root: Path | str | None = None,
    extraction_config: ExtractionConfig | None = None,
    language_detector: LanguageDetector | None = None,
    limit: int | None = None,
) -> ExtractionReport:
    """Offline extraction of raw HTML/PDF documents from the registry store."""
    reg_path = Path(registry_path)
    if not reg_path.exists():
        raise FileNotFoundError(f"Registry DB not found: {reg_path}")

    ext_cfg = extraction_config or ExtractionConfig()

    if language_detector is None:
        if ext_cfg.language_model_path.exists():
            language_detector = FastTextLanguageDetector(ext_cfg.language_model_path)
        else:
            language_detector = FallbackLanguageDetector()

    attempted = 0
    ready = 0
    quarantined = 0
    failed = 0
    flags_by_reason: Counter[str] = Counter()

    with SourceRegistry(reg_path) as registry:
        cursor = registry.conn.execute(
            """
            SELECT s.doc_id, f.state, f.raw_path, f.raw_sha256, f.mime_type
            FROM sources s
            JOIN fetch_jobs f ON s.fetch_job_id = f.job_id
            WHERE f.state = 'fetched' AND f.raw_path IS NOT NULL
            ORDER BY s.doc_id ASC
            """
        )

        rows = cursor.fetchall()
        if limit is not None and limit > 0:
            rows = rows[:limit]

        for doc_id, state, raw_path_str, raw_sha256, mime_type in rows:
            attempted += 1
            raw_path = Path(raw_path_str)
            if not raw_path.exists() and raw_root is not None:
                candidate = Path(raw_root) / raw_path_str
                if candidate.exists():
                    raw_path = candidate

            if not raw_path.exists():
                failed += 1
                flags_by_reason["raw_file_missing"] += 1
                continue

            try:
                if str(raw_path).endswith(".gz"):
                    with gzip.open(raw_path, "rb") as gf:
                        raw_bytes = gf.read()
                else:
                    raw_bytes = raw_path.read_bytes()

                computed_raw_sha = hashlib.sha256(raw_bytes).hexdigest()
                if computed_raw_sha != raw_sha256:
                    failed += 1
                    flags_by_reason["sha256_mismatch"] += 1
                    continue

                extracted_doc = extract_document(
                    doc_id=doc_id,
                    raw=raw_bytes,
                    mime_type=mime_type or "text/html",
                    raw_sha256=raw_sha256,
                    config=ext_cfg,
                )

                quality = classify_quality(extracted_doc, language_detector, ext_cfg)
                if quality.state == "ready":
                    ready += 1
                else:
                    quarantined += 1
                    for flg in quality.flags:
                        flags_by_reason[flg] += 1

            except Exception:
                failed += 1
                flags_by_reason["extraction_exception"] += 1

    return ExtractionReport(
        attempted=attempted,
        ready=ready,
        quarantined=quarantined,
        failed=failed,
        flags_by_reason=dict(flags_by_reason),
    )


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Offline document extraction from raw payloads"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default=None,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Path to SQLite registry database",
    )
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=None,
        help="Path to raw payload storage directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume offline extraction",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to extract",
    )

    parsed = parser.parse_args(args)

    cfg: dict[str, Any] = {}
    config_file = parsed.config
    if config_file is None and Path("configs/vibiomir_phase1.yaml").exists():
        config_file = Path("configs/vibiomir_phase1.yaml")

    if config_file is not None and Path(config_file).exists():
        cfg = load_config(config_file)

    ds_cfg = cfg.get("dataset", {})
    cr_cfg = cfg.get("crawl", {})
    ex_cfg = cfg.get("extraction", {})

    reg_p = parsed.registry_path or Path(ds_cfg.get("registry_path", "artifacts/vibiomir/registry.sqlite"))
    raw_p = parsed.raw_root or Path(cr_cfg.get("raw_root", "artifacts/vibiomir/raw"))

    extraction_config = ExtractionConfig(
        html_backend=ex_cfg.get("html_backend", "trafilatura"),
        pdf_backend=ex_cfg.get("pdf_backend", "pymupdf"),
        language_model_path=Path(ex_cfg.get("language_model_path", "artifacts/models/lid.176.ftz")),
        language_confidence_min=ex_cfg.get("language_confidence_min", 0.70),
        accepted_languages=tuple(ex_cfg.get("accepted_languages", ["vi", "en", "zh"])),
        quarantine_unknown_language=ex_cfg.get("quarantine_unknown_language", True),
    )

    print("=== ViBioMIR Offline Document Extraction ===")
    print(f"Registry DB:     {reg_p}")
    print(f"Raw storage:     {raw_p}")

    t0 = time.time()
    report = run_extract(
        registry_path=reg_p,
        raw_root=raw_p,
        extraction_config=extraction_config,
        limit=parsed.limit,
    )
    elapsed = time.time() - t0

    print("\n=== Extraction Summary ===")
    print(f"Attempted:    {report.attempted:,}")
    print(f"Ready:        {report.ready:,}")
    print(f"Quarantined:  {report.quarantined:,}")
    print(f"Failed:       {report.failed:,}")
    print(f"Flags:        {report.flags_by_reason}")
    print(f"Elapsed Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
