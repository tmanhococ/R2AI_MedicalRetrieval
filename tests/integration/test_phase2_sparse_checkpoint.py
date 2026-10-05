from __future__ import annotations

import json
from pathlib import Path
import pytest
import yaml

from src.cli.predict import main as predict_main
from src.submission.validators import validate_zip
from src.data.vibiomir_contracts import OfficialQuery, SourceStore
from src.indexing.lucene import LexicalRecord, LuceneSparseBackend
from src.indexing.lexical_tokens import tokenize_and_encode


@pytest.fixture
def b1_sparse_env(tmp_path: Path):
    """Sets up a B1 sparse-only phase 2 environment with a sample index and config."""
    index_dir = tmp_path / "indexes"
    index_dir.mkdir()
    
    # Create sparse partition for vi_document
    backend = LuceneSparseBackend()
    large_id = 5_000_000_001
    records = [
        LexicalRecord(
            external_id=str(large_id),
            official_doc_id=large_id,
            language="vi",
            item_type="document",
            encoded_contents=tokenize_and_encode("Bệnh nhân tiểu đường cấp cứu điều trị insulin 100mg", "vi"),
        ),
        LexicalRecord(
            external_id="doc_vi_normal",
            official_doc_id=1002,
            language="vi",
            item_type="document",
            encoded_contents=tokenize_and_encode("Bệnh nhân tăng huyết áp điều trị amlodipine", "vi"),
        ),
        # Empty hit / zero result case
        LexicalRecord(
            external_id="doc_vi_empty",
            official_doc_id=1003,
            language="vi",
            item_type="document",
            encoded_contents=tokenize_and_encode("Hoàn toàn không liên quan đến từ khóa tìm kiếm", "vi"),
        ),
    ]
    part_dir = tmp_path / "part_vi_doc"
    backend.build_partition(records, part_dir)
    backend.assemble([part_dir], index_dir / "vi_document")

    # Create dataset directory containing test.jsonl and documents.jsonl
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    queries_path = dataset_dir / "test.jsonl"
    queries_data = [
        {"query_id": 1, "text": "tiểu đường insulin", "language": "vi", "item_type": "document"},
        {"query_id": 2, "text": "tăng huyết áp", "language": "vi", "item_type": "document"},
        {"query_id": 3, "text": "nonexistenttermzxy", "language": "vi", "item_type": "document"},  # zero hit case
    ]
    with open(queries_path, "w", encoding="utf-8") as f:
        for q in queries_data:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    docs_path = dataset_dir / "documents.jsonl"
    docs_data = [
        {"doc_id": large_id, "text": "Bệnh nhân tiểu đường cấp cứu điều trị insulin 100mg", "language": "vi", "item_type": "document", "source_version": "1.0", "source_sha256": "abc"},
        {"doc_id": 1002, "text": "Bệnh nhân tăng huyết áp điều trị amlodipine", "language": "vi", "item_type": "document", "source_version": "1.0", "source_sha256": "abc"},
        {"doc_id": 1003, "text": "Hoàn toàn không liên quan đến từ khóa tìm kiếm", "language": "vi", "item_type": "document", "source_version": "1.0", "source_sha256": "abc"},
    ]
    with open(docs_path, "w", encoding="utf-8") as f:
        for d in docs_data:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Also keep a copy for validation step
    val_queries_path = tmp_path / "queries.jsonl"
    with open(val_queries_path, "w", encoding="utf-8") as f:
        for q in queries_data:
            f.write(json.dumps({"id": q["query_id"], "query": q["text"]}, ensure_ascii=False) + "\n")

    zip_path = tmp_path / "vibiomir_phase2_B1_sparse_test" / "submission.jsonl" / "submission.zip"

    # Create config file for B1 sparse-only using valid allowed sections
    config_data = {
        "run": {
            "experiment_name": "vibiomir_phase2_B1_sparse_test",
            "seed": 42,
            "output_dir": str(tmp_path),
        },
        "dataset": {
            "adapter": "jsonl",
            "expected_query_count": 3,
            "verify_actual_counts": False,
        },
        "retrieval": {
            "candidate_top_k": 5,
            "sparse": {
                "backend": "lucene",
                "index_dir": str(index_dir),
                "top_k": 5,
            },
            "dense": {
                "enabled": False,
            }
        },
        "fusion": {
            "method": "weighted_rrf",
            "weights": {"vi_document_sparse": 1.0},
        },
        "reranker": {
            "enabled": False,
        },
        "output": {
            "min_results": 0,
            "document_cap": 5,
            "evidence_cap": 5,
        },
        "submission": {
            "exporter": "vibiomir_json_zip",
            "zip_filename": "submission.zip",
            "require_completed_queries": 3,
            "provenance_check": "none",
        }
    }
    config_path = tmp_path / "config_b1.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    return config_path, zip_path, val_queries_path, dataset_dir, [large_id, 1002, 1003]


def test_phase2_sparse_checkpoint_flow(b1_sparse_env):
    """Tests the B1 sparse-only checkpoint flow: building, predicting, exporting, and validating submission ZIP."""
    config_path, zip_path, queries_path, dataset_dir, doc_ids = b1_sparse_env

    # Run prediction via sparse route (without loading E5 or BGE)
    predict_main(["--config", str(config_path), "--dataset", str(dataset_dir), "--exporter", "vibiomir_json_zip", "--export-submission"])

    assert zip_path.exists(), f"submission ZIP must be generated at {zip_path}"

    # Load queries and source store for validation
    queries = []
    with open(queries_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            queries.append(OfficialQuery(query_id=item["id"], query=item["query"]))

    # Dummy SourceStore for validation supporting contains_official_doc
    class DummyStore:
        snapshot_id = "phase2_full"
        def __contains__(self, item):
            return True
        def contains_official_doc(self, doc_id):
            return True
        def get_document(self, doc_id):
            return type("Doc", (), {"source_text": "sample source source text"})()

    store = DummyStore()

    # Validate submission ZIP via validate_zip
    report = validate_zip(str(zip_path), queries=queries, store=store)
    assert report.valid, f"validate_zip failed: {report.errors}"
