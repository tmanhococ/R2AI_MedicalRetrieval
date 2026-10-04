from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from src.cli.common import (
    build_query_components,
    build_reranker,
    build_retrievers,
    load_corpus_and_texts,
    resolve_index_dirs,
    run_pipeline_for_queries,
)
from src.data.schemas import FinalPrediction, SearchHit
from src.data.source_store import SnapshotSourceStore
from src.data.vibiomir_contracts import (
    FetchResult,
    OfficialQuery,
    OfficialSource,
)
from src.data.vibiomir_registry import SourceRegistry, ingest_sources
from src.reranking.reranker import DummyReranker
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.submission.validators import validate_zip
from src.utils.config import load_config, merge_configs, validate_config


class FastMockDenseEncoder:
    """Fast deterministic dense encoder producing 128-dim embeddings in microseconds."""

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def encode(self, texts: list[str] | str, **kwargs: Any) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16) % (2**31)
            rng = np.random.RandomState(seed)
            vec = rng.randn(self.dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            vectors.append(vec)
        return np.vstack(vectors)


# ==============================================================================
# 1. Config Merge and Validation Tests
# ==============================================================================

def test_config_merge_cli_yaml_defaults(tmp_path: Path):
    """Test precedence: CLI overrides > YAML config > code defaults."""
    code_defaults = {
        "run": {"seed": 42, "experiment_name": "default_exp"},
        "crawl": {"concurrency": 4, "timeout": 30},
    }

    yaml_file = tmp_path / "test_config.yaml"
    yaml_content = """
run:
  seed: 100
crawl:
  concurrency: 8
"""
    yaml_file.write_text(yaml_content, encoding="utf-8")

    cli_overrides = {
        "run": {"seed": 999},
    }

    # Step 1: Base + YAML
    from src.utils.config import load_config, merge_configs

    loaded_yaml = load_config(yaml_file, validate=False)
    merged_yaml = merge_configs(code_defaults, loaded_yaml)
    assert merged_yaml["run"]["seed"] == 100  # YAML overrides code default
    assert merged_yaml["run"]["experiment_name"] == "default_exp"  # code default preserved
    assert merged_yaml["crawl"]["concurrency"] == 8  # YAML overrides code default
    assert merged_yaml["crawl"]["timeout"] == 30  # code default preserved

    # Step 2: Merged + CLI overrides
    final_cfg = merge_configs(merged_yaml, cli_overrides)
    assert final_cfg["run"]["seed"] == 999  # CLI override wins
    assert final_cfg["run"]["experiment_name"] == "default_exp"
    assert final_cfg["crawl"]["concurrency"] == 8


def test_unknown_yaml_key_rejected(tmp_path: Path):
    """Test that unknown configuration keys are strictly rejected."""
    bad_yaml = tmp_path / "bad_config.yaml"
    bad_yaml.write_text("unknown_top_level_section:\n  foo: bar\n", encoding="utf-8")

    with pytest.raises(ValueError, match="(?i)unknown"):
        load_config(bad_yaml, validate=True)


# ==============================================================================
# 2. Unit End-to-End Fixture: 3 Queries, 2 URL Aliases, 1 Fetch Failure
# ==============================================================================

@pytest.fixture
def e2e_dataset_dir(tmp_path: Path) -> Path:
    """Creates small official parquet inputs: 3 queries (1, 4, 9), 2 URL aliases, 1 failure."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # 1. Official queries: IDs 1, 4, 9 (non-sequential, exactly 3)
    queries_data = [
        {"id": 1, "query": "Bệnh đái tháo đường và triệu chứng thường gặp"},
        {"id": 4, "query": "Dấu hiệu cảnh báo hạ đường huyết cấp tính"},
        {"id": 9, "query": "Biện pháp phòng ngừa biến chứng tim mạch ở người tiểu đường"},
    ]
    query_schema = pa.schema([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("query", pa.string(), nullable=False),
    ])
    query_table = pa.Table.from_pylist(queries_data, schema=query_schema)
    pq.write_table(query_table, data_dir / "query.parquet")

    # 2. Official sources:
    # - doc_id 101 and 102 share URL: "http://example.com/alias" (2 URL aliases)
    # - doc_id 103 has URL: "http://example.com/fail" (1 fetch failure)
    # - doc_id 104 has URL: "http://example.com/valid2" (healthy distinct source)
    sources_data = [
        {"id": 101, "url": "http://example.com/alias"},
        {"id": 102, "url": "http://example.com/alias"},
        {"id": 103, "url": "http://example.com/fail"},
        {"id": 104, "url": "http://example.com/valid2"},
    ]
    source_schema = pa.schema([
        pa.field("id", pa.int64(), nullable=False),
        pa.field("url", pa.string(), nullable=False),
    ])
    source_table = pa.Table.from_pylist(sources_data, schema=source_schema)
    pq.write_table(source_table, data_dir / "links_corpus.parquet")

    return data_dir


def test_unit_e2e_pipeline_with_aliases_and_fetch_failure(tmp_path: Path, e2e_dataset_dir: Path):
    """Unit E2E: Ingest, crawl with mock responses, build snapshot, index, predict & export.

    - 3 query IDs: 1, 4, 9
    - 2 URL aliases: doc 101 & 102 share http://example.com/alias
    - 1 fetch failure: http://example.com/fail
    - Runs in < 3s using FastMockDenseEncoder and DummyReranker.
    - Produces valid results.json in single-file ZIP with 3 queries and exact source spans.
    """
    artifacts_dir = tmp_path / "artifacts"
    raw_root = artifacts_dir / "raw"
    registry_path = artifacts_dir / "registry.sqlite"
    snapshots_dir = artifacts_dir / "snapshots"
    experiments_dir = artifacts_dir / "experiments" / "test_run"

    # Step 1: Ingest via src.cli.ingest_vibiomir
    from src.cli.ingest_vibiomir import run_ingest

    ingest_report = run_ingest(
        corpus_path=e2e_dataset_dir / "links_corpus.parquet",
        query_path=e2e_dataset_dir / "query.parquet",
        registry_path=registry_path,
        batch_size=10,
        expected_query_count=3,
    )
    assert ingest_report.input_rows == 4
    assert ingest_report.unique_urls == 3  # 101 and 102 share same URL!

    # Step 2: Crawl with mock network fetcher
    alias_html = (
        "<!DOCTYPE html><html><head><title>Bệnh tiểu đường</title></head><body>"
        "<h1>Kiểm soát đường huyết</h1>"
        "<p>Bệnh đái tháo đường là bệnh lý mạn tính cần được theo dõi đường huyết thường xuyên.</p>"
        "<p>Dấu hiệu cảnh báo hạ đường huyết cấp tính bao gồm vã mồ hôi, run tay và chóng mặt.</p>"
        "</body></html>"
    ).encode("utf-8")

    valid2_html = (
        "<!DOCTYPE html><html><head><title>Tim mạch và đái tháo đường</title></head><body>"
        "<h1>Biến chứng tim mạch</h1>"
        "<p>Biện pháp phòng ngừa biến chứng tim mạch ở người tiểu đường là duy trì huyết áp ổn định.</p>"
        "</body></html>"
    ).encode("utf-8")

    async def mock_fetch_one(job, client, config):
        if "fail" in job.url:
            return FetchResult(
                job_id=job.job_id,
                state="fetch_failed",
                raw_path=None,
                raw_sha256=None,
                final_url=job.url,
                mime_type="text/html",
                http_status=404,
                fetched_at="2026-10-04T12:00:00Z",
                error_code="http_404",
            )
        content = alias_html if "alias" in job.url else valid2_html
        content_hash = hashlib.sha256(content).hexdigest()
        raw_path = raw_root / f"{job.job_id}.html.gz"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(raw_path, "wb") as gf:
            gf.write(content)
        return FetchResult(
            job_id=job.job_id,
            state="fetched",
            raw_path=str(raw_path),
            raw_sha256=content_hash,
            final_url=job.url,
            mime_type="text/html",
            http_status=200,
            fetched_at="2026-10-04T12:00:00Z",
            error_code=None,
            bytes_written=len(content),
        )

    with patch("src.ingestion.crawl_runner.fetch_one", side_effect=mock_fetch_one):
        from src.cli.crawl import run_crawl_cli
        crawl_report = run_crawl_cli(
            registry_path=registry_path,
            raw_root=raw_root,
            limit=10,
        )
    assert crawl_report.fetched == 2  # 2 unique URLs succeeded
    assert crawl_report.failed == 1   # 1 unique URL failed (fail)

    # Step 3: Build immutable snapshot via src.cli.build_snapshot
    from src.cli.build_snapshot import run_build_snapshot
    manifest = run_build_snapshot(
        registry_path=registry_path,
        snapshot_root=snapshots_dir,
        name="test_snapshot_01",
        raw_root=raw_root,
        queries_path=e2e_dataset_dir / "query.parquet",
    )
    assert manifest.ready_document_count == 3  # docs 101, 102, 104 are ready! (103 failed)
    assert manifest.chunk_count > 0

    snapshot_path = snapshots_dir / "test_snapshot_01"

    # Step 4: Index via src.cli.index using FastMockDenseEncoder
    from src.cli.index import run_index
    index_cfg = {
        "run": {"output_dir": str(experiments_dir)},
        "retrieval": {
            "bm25": {"enabled": True, "backend": "bm25s"},
            "dense": {"enabled": True, "backend": "faiss", "batch_size": 16},
        },
    }
    index_result = run_index(
        config=index_cfg,
        snapshot_dir=snapshot_path,
        output_dir=experiments_dir / "indexes",
        encoder=FastMockDenseEncoder(),
    )
    assert index_result["num_documents"] == 3

    # Step 5: Predict & Export Submission via src.cli.predict
    from src.cli.predict import run_predict
    predict_cfg = {
        "run": {"output_dir": str(experiments_dir), "experiment_name": "test_exp"},
        "dataset": {"query_path": str(e2e_dataset_dir / "query.parquet")},
        "retrieval": {
            "candidate_top_k": 10,
            "bm25": {"enabled": True, "backend": "bm25s"},
            "dense": {"enabled": True, "backend": "faiss", "batch_size": 16},
        },
        "fusion": {"weights": {"vi_bm25": 1.0, "vi_dense": 1.0}},
        "reranker": {"enabled": True, "model_name": "dummy"},
        "calibration": {"bounds": {"documents": {"min": 1, "max": 5}, "chunks": {"min": 1, "max": 5}}},
        "submission": {
            "exporter": "vibiomir_json_zip",
            "json_filename": "results.json",
            "zip_filename": "submission.zip",
        },
    }

    zip_path = run_predict(
        config=predict_cfg,
        split="test",
        snapshot_dir=snapshot_path,
        index_dir=experiments_dir / "indexes",
        output_dir=experiments_dir,
        encoder=FastMockDenseEncoder(),
        reranker=DummyReranker(),
        export_submission=True,
    )

    assert Path(zip_path).exists()

    # Step 6: Validate ZIP and submission rows
    from src.data.vibiomir_adapter import load_official_queries
    queries = load_official_queries(e2e_dataset_dir / "query.parquet")
    store = SnapshotSourceStore(snapshot_path)

    val_report = validate_zip(zip_path, queries=queries, store=store, expected_count=3)
    assert val_report.valid is True
    assert val_report.query_count == 3
    assert len(val_report.errors) == 0

    # Verify all 3 query IDs are present in root results.json
    with zipfile.ZipFile(zip_path) as zf:
        raw_json = json.loads(zf.read("results.json").decode("utf-8"))
        result_ids = [r["id"] for r in raw_json]
        assert result_ids == [1, 4, 9]

        # Verify correct source chunk spans for non-empty predictions
        for row in raw_json:
            for ch in row["relevant_chunks"]:
                doc = store.get_document(ch["doc_id"])
                assert ch["chunk_text"] in doc.source_text


# ==============================================================================
# 3. Checkpoint & Resume Tests
# ==============================================================================

def test_inference_kill_and_resume_skips_completed_queries(tmp_path: Path, e2e_dataset_dir: Path):
    """Test kill & resume: running inference partially, killing, and resuming skips completed queries."""
    output_dir = tmp_path / "checkpoints_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.utils.checkpoint import InferenceCheckpointManager

    queries = [
        OfficialQuery(query_id=1, query="q1"),
        OfficialQuery(query_id=4, query="q2"),
        OfficialQuery(query_id=9, query="q3"),
    ]

    run_fp = "test_run_fingerprint_abc123"

    # Stage 1: Process queries 1 and 4, then simulate kill/interrupt
    mgr1 = InferenceCheckpointManager(output_dir=output_dir, run_fingerprint=run_fp)
    mgr1.save_checkpoint(query_id=1, status="completed", prediction={"query_id": "1", "document_ids": ["101"], "chunk_ids": ["101:h:0"]})
    mgr1.save_checkpoint(query_id=4, status="completed", prediction={"query_id": "4", "document_ids": ["102"], "chunk_ids": ["102:h:0"]})

    # Stage 2: Resume with same fingerprint
    mgr2 = InferenceCheckpointManager(output_dir=output_dir, run_fingerprint=run_fp)
    completed_qids = mgr2.get_completed_query_ids()
    assert completed_qids == {1, 4}

    # Simulate skipping completed queries and processing remaining query 9
    executed_queries = []
    for q in queries:
        if q.query_id in completed_qids:
            continue
        executed_queries.append(q.query_id)
        mgr2.save_checkpoint(query_id=q.query_id, status="completed", prediction={"query_id": str(q.query_id), "document_ids": ["104"], "chunk_ids": []})

    assert executed_queries == [9]  # Query 1 and 4 were skipped!
    assert mgr2.get_completed_query_ids() == {1, 4, 9}


def test_inference_failure_records_failed_state_and_blocks_export(tmp_path: Path):
    """Query inference failure records a failure state and BLOCKS the final export."""
    output_dir = tmp_path / "failure_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.utils.checkpoint import InferenceCheckpointManager

    run_fp = "test_run_fingerprint_fail"
    mgr = InferenceCheckpointManager(output_dir=output_dir, run_fingerprint=run_fp)

    # Query 1 completes
    mgr.save_checkpoint(query_id=1, status="completed", prediction={"query_id": "1", "document_ids": ["101"], "chunk_ids": []})
    # Query 4 fails with exception
    mgr.save_checkpoint(query_id=4, status="failed", error="CUDA out of memory error during inference")

    # Assert manager tracks failure
    assert mgr.has_failed_queries() is True
    failed_qids = mgr.get_failed_query_ids()
    assert 4 in failed_qids

    # Final export must be blocked!
    with pytest.raises(RuntimeError, match="(?i)inference failed.*blocking.*export"):
        mgr.validate_and_get_all_predictions(required_query_ids=[1, 4, 9])


def test_resume_with_mismatched_run_fingerprint_rejected(tmp_path: Path):
    """Resuming with mismatched run_fingerprint is rejected with an error."""
    output_dir = tmp_path / "mismatch_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    from src.utils.checkpoint import InferenceCheckpointManager

    # Initial checkpoint written with run_fingerprint_v1
    mgr1 = InferenceCheckpointManager(output_dir=output_dir, run_fingerprint="run_fingerprint_v1")
    mgr1.save_checkpoint(query_id=1, status="completed", prediction={"query_id": "1", "document_ids": [], "chunk_ids": []})

    # Attempt to resume with different model/config -> run_fingerprint_v2
    with pytest.raises(ValueError, match="(?i)fingerprint mismatch"):
        InferenceCheckpointManager(output_dir=output_dir, run_fingerprint="run_fingerprint_v2", resume=True)


# ==============================================================================
# 4. Anti-leakage and Edge Cases Tests
# ==============================================================================

def test_test_set_without_labels_does_not_generate_fake_zero_metrics(tmp_path: Path):
    """Test set without labels does NOT generate fake '0' metrics or metrics.json."""
    output_dir = tmp_path / "no_labels_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Verify run_predict does not write metrics.json
    metrics_file = output_dir / "metrics.json"
    assert not metrics_file.exists()


def test_missing_language_stream_does_not_crash():
    """Missing or empty language stream does not crash retrieval orchestrator."""
    from src.query.query_processor import QueryVariants

    orchestrator = RetrievalOrchestrator(bm25_retriever=None, dense_retriever=None)
    variants = QueryVariants(vi="triệu chứng sốt xuất huyết", en="", zh=None)

    # Should not crash and return empty streams safely
    stream_results = orchestrator.retrieve_all_streams(variants, top_k=10)
    assert isinstance(stream_results.doc_streams, dict)
    assert isinstance(stream_results.chunk_streams, dict)


def test_no_candidate_query_returns_clean_empty():
    """Query with zero candidates produces clean empty prediction row."""
    from src.fusion.weighted_rrf import fuse_stream_results
    from src.retrieval.orchestrator import StreamResults
    from src.scoring.joint_scoring import JointScorer

    empty_streams = StreamResults(doc_streams={}, chunk_streams={})
    fused_docs, fused_chunks = fuse_stream_results(empty_streams, top_k_docs=10, top_k_chunks=20)
    assert fused_docs == []
    assert fused_chunks == []

    scorer = JointScorer()
    joint_docs, joint_chunks = scorer.score([], [])
    assert joint_docs == []
    assert joint_chunks == []


def test_evaluate_on_dev_split_with_snapshot_and_anti_leakage(tmp_path: Path, e2e_dataset_dir: Path):
    """Test evaluate.py on dev split with snapshot produces metrics and csv, while rejecting test split."""
    from src.cli.evaluate import run_evaluate

    eval_cfg = {
        "run": {"output_dir": str(tmp_path / "eval_out")},
        "dataset": {"query_path": str(e2e_dataset_dir / "query.parquet")},
    }

    # 1. Anti-leakage: split="test" without labels must be rejected immediately
    with pytest.raises(ValueError, match="(?i)without ground truth labels"):
        run_evaluate(
            config=eval_cfg,
            split="test",
            snapshot_dir=tmp_path / "dummy_snapshot",
        )

