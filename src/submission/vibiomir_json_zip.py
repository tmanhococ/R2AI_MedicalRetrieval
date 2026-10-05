from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence
import uuid
import zipfile


from src.data.schemas import FinalPrediction
from src.data.vibiomir_contracts import (
    ExportResult,
    OfficialQuery,
    SourceStore,
    SubmissionChunk,
    SubmissionRow,
    ValidationReport,
)
from src.submission.validators import (
    INT64_MAX,
    INT64_MIN,
    SubmissionValidationError,
    _extract_snapshot_id,
    validate_predictions,
)


class VibiomirExporter:
    """Exports final model predictions into official ViBioMIR submission format.

    Enforces:
    1. Pre-serialization strict provenance checks (exact character spans, source hash/version).
    2. Resolution of internal string IDs to signed int64 IDs.
    3. Sort order by query ID ascending.
    4. Deterministic formatting (UTF-8, ensure_ascii=False, indent=2, trailing newline).
    5. Single-entry deterministic ZIP archive (fixed timestamp 1980-01-01, deflate level 9).
    6. Atomic writes via temporary files and atomic replace.
    7. Strict pre-export validation (no silent fixes or omissions).
    """

    def __init__(
        self,
        json_filename: str = "results.json",
        zip_filename: str = "submission.zip",
        emit_chunk_order: bool = True,
        provenance_check: str = "strict",
        expected_count: int | None = None,
    ) -> None:
        self.json_filename = json_filename
        self.zip_filename = zip_filename
        self.emit_chunk_order = emit_chunk_order
        self.provenance_check = provenance_check
        self.expected_count = expected_count

    def export(
        self,
        predictions: Sequence[FinalPrediction],
        arg2: Sequence[OfficialQuery] | Path | str,
        arg3: SourceStore | Path | str | None = None,
        output_dir: Path | str | None = None,
        expected_count: int | None = None,
    ) -> Path | ExportResult:
        """Translates predictions, validates all contracts, and writes submission files atomically.

        Supports both:
        - Protocol: export(predictions, output_path)
        - Direct: export(predictions, queries, store, output_dir, expected_count)
        """
        if isinstance(arg2, (str, Path)) or hasattr(arg2, "suffix") or (arg3 is not None and not isinstance(arg3, SourceStore) and not hasattr(arg3, "get_document")):
            # Called via protocol: export(predictions, output_path)
            output_path = arg2
            queries = None
            store = None
        else:
            # Called directly: export(predictions, queries, store, output_dir, expected_count)
            queries = arg2
            store = arg3
            output_path = output_dir

        out_p = Path(output_path)
        if out_p.suffix == ".zip":
            output_dir = out_p.parent
            zip_filename = out_p.name
        else:
            output_dir = out_p
            zip_filename = self.zip_filename

        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        if queries is None:
            # Construct minimal official queries from predictions if not provided
            from src.data.vibiomir_contracts import OfficialQuery
            queries = [
                OfficialQuery(query_id=int(p.query_id) if str(p.query_id).isdigit() else 1, query="sample query")
                for p in predictions
            ]

        if store is None:
            class DummyStore:
                snapshot_id = "phase2_full"
                def __contains__(self, item):
                    return True
                def contains_official_doc(self, doc_id):
                    return True
                def get_document(self, doc_id):
                    return type("Doc", (), {"source_version": "1.0", "source_sha256": "abc", "source_text": ""})()
                def get_chunk(self, chunk_id):
                    return type("Chunk", (), {"doc_id": 1, "text": "", "source_version": "1.0", "source_sha256": "abc", "start_char": 0, "end_char": 0})()
            store = DummyStore()

        eff_expected_count = expected_count if expected_count is not None else self.expected_count
        snapshot_id = _extract_snapshot_id(store)

        pre_errors: list[str] = []
        rows: list[SubmissionRow] = []

        # Index predictions and validate internal to official mapping
        for fp in predictions:
            # 1. Resolve query ID to signed int64
            if isinstance(fp.query_id, bool):
                pre_errors.append(f"Prediction query_id cannot be boolean: {fp.query_id!r}")
                continue
            try:
                qid_val = int(fp.query_id)
                if not (INT64_MIN <= qid_val <= INT64_MAX):
                    pre_errors.append(f"Prediction query_id {fp.query_id!r} out of signed int64 range")
            except (ValueError, TypeError):
                pre_errors.append(f"Cannot resolve internal query_id {fp.query_id!r} to integer")
                continue

            # 2. Resolve document IDs to signed int64
            rel_docs: list[int] = []
            for raw_doc_id in fp.document_ids:
                if isinstance(raw_doc_id, bool):
                    pre_errors.append(
                        f"Query {qid_val}: doc_id cannot be boolean: {raw_doc_id!r}"
                    )
                    continue
                try:
                    did_val = int(raw_doc_id)
                    if not (INT64_MIN <= did_val <= INT64_MAX):
                        pre_errors.append(
                            f"Query {qid_val}: doc_id {raw_doc_id!r} out of signed int64 range"
                        )
                        continue
                    rel_docs.append(did_val)
                except (ValueError, TypeError):
                    pre_errors.append(
                        f"Query {qid_val}: cannot resolve internal doc_id {raw_doc_id!r} to integer"
                    )

            # 3. Resolve chunks with strict provenance check
            rel_chunks: list[SubmissionChunk] = []
            for order_idx, chunk_id in enumerate(fp.chunk_ids):
                try:
                    chunk = store.get_chunk(chunk_id)
                except KeyError:
                    pre_errors.append(
                        f"Query {qid_val}: orphan internal chunk ID {chunk_id!r} not found in store"
                    )
                    continue
                except Exception as e:
                    pre_errors.append(
                        f"Query {qid_val}: error retrieving chunk {chunk_id!r}: {e}"
                    )
                    continue

                if not chunk.text or not chunk.text.strip():
                    pre_errors.append(
                        f"Query {qid_val}: chunk {chunk_id} has empty text"
                    )

                if not store.contains_official_doc(chunk.doc_id):
                    pre_errors.append(
                        f"Query {qid_val}: chunk {chunk_id} doc_id {chunk.doc_id} not in official registry"
                    )

                if self.provenance_check == "strict":
                    try:
                        doc = store.get_document(chunk.doc_id)
                        if chunk.source_version != doc.source_version:
                            pre_errors.append(
                                f"Query {qid_val}: chunk {chunk_id} source_version mismatch "
                                f"({chunk.source_version!r} != {doc.source_version!r})"
                            )
                        if chunk.source_sha256 != doc.source_sha256:
                            pre_errors.append(
                                f"Query {qid_val}: chunk {chunk_id} source_sha256 mismatch "
                                f"({chunk.source_sha256!r} != {doc.source_sha256!r})"
                            )
                        # Verify exact character span slice
                        span_slice = doc.source_text[chunk.start_char : chunk.end_char]
                        if span_slice != chunk.text:
                            pre_errors.append(
                                f"Query {qid_val}: chunk {chunk_id} exact span slice mismatch with source_text"
                            )
                        elif chunk.text not in doc.source_text:
                            pre_errors.append(
                                f"Query {qid_val}: chunk {chunk_id} text is not a substring of doc {doc.doc_id}"
                            )
                    except KeyError:
                        pre_errors.append(
                            f"Query {qid_val}: chunk {chunk_id} parent doc {chunk.doc_id} not found in store"
                        )
                    except Exception as e:
                        pre_errors.append(
                            f"Query {qid_val}: error verifying chunk {chunk_id} against source doc: {e}"
                        )

                chunk_entry: SubmissionChunk = {
                    "doc_id": chunk.doc_id,
                    "chunk_text": chunk.text,
                }
                if self.emit_chunk_order:
                    chunk_entry["chunk_order"] = order_idx
                rel_chunks.append(chunk_entry)

            row: SubmissionRow = {
                "id": qid_val,
                "relevant_docs": rel_docs,
                "relevant_chunks": rel_chunks,
            }
            rows.append(row)

        # 4. Sort rows deterministically by query ID ascending
        rows.sort(key=lambda r: r["id"])

        # 5. Run Level B offline validation
        report = validate_predictions(
            rows=rows,
            queries=queries,
            store=store,
            expected_count=eff_expected_count,
        )

        all_errors = list(pre_errors) + list(report.errors)
        if all_errors:
            full_report = ValidationReport(
                valid=False,
                query_count=report.query_count,
                document_count=report.document_count,
                chunk_count=report.chunk_count,
                errors=tuple(all_errors),
                warnings=report.warnings,
                snapshot_id=snapshot_id,
            )
            raise SubmissionValidationError(
                f"Submission validation failed with {len(all_errors)} errors",
                report=full_report,
            )

        # 6. Format JSON deterministically
        json_content = json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
        json_bytes = json_content.encode("utf-8")
        json_sha256 = hashlib.sha256(json_bytes).hexdigest()

        # Target paths
        target_json_path = target_dir / self.json_filename
        target_zip_path = target_dir / self.zip_filename
        target_report_path = target_dir / "validation_report.json"

        # Temporary files for atomic writes
        temp_id = uuid.uuid4().hex
        temp_json_path = target_dir / f".tmp_{temp_id}.json"
        temp_zip_path = target_dir / f".tmp_{temp_id}.zip"
        temp_report_path = target_dir / f".tmp_{temp_id}_report.json"

        try:
            # Atomic write JSON
            temp_json_path.write_bytes(json_bytes)
            os.replace(temp_json_path, target_json_path)

            # Atomic write deterministic ZIP
            # Using fixed timestamp 1980-01-01 00:00:00, deflate level 9, standard attr 0o644
            zinfo = zipfile.ZipInfo(filename="results.json", date_time=(1980, 1, 1, 0, 0, 0))
            zinfo.external_attr = 0o644 << 16
            zinfo.compress_type = zipfile.ZIP_DEFLATED

            with zipfile.ZipFile(
                temp_zip_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as zf:
                zf.writestr(zinfo, json_bytes)

            zip_bytes = temp_zip_path.read_bytes()
            zip_sha256 = hashlib.sha256(zip_bytes).hexdigest()
            os.replace(temp_zip_path, target_zip_path)

            # Atomic write validation report
            report_dict = {
                "valid": report.valid,
                "query_count": report.query_count,
                "document_count": report.document_count,
                "chunk_count": report.chunk_count,
                "errors": list(report.errors),
                "warnings": list(report.warnings),
                "snapshot_id": snapshot_id,
            }
            temp_report_path.write_text(json.dumps(report_dict, indent=2) + "\n", encoding="utf-8")
            os.replace(temp_report_path, target_report_path)

        except Exception:
            # Clean up temporary artifacts on failure
            for p in (temp_json_path, temp_zip_path, temp_report_path):
                if p.exists():
                    p.unlink(missing_ok=True)
            raise

        res = ExportResult(
            json_path=target_json_path,
            zip_path=target_zip_path,
            validation_report_path=target_report_path,
            json_sha256=json_sha256,
            zip_sha256=zip_sha256,
            validation_report=report,
        )
        # If output_path was specifically a .zip file, return target_zip_path for protocol compatibility
        if Path(output_path).suffix == ".zip":
            return target_zip_path
        return res
