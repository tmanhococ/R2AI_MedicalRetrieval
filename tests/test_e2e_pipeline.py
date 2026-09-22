from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import numpy as np
import pytest

from src.data.schemas import ChunkRecord, DocumentRecord
from src.indexing.dense_index import DenseIndexer
from src.reranking.reranker import DummyReranker
from src.retrieval.dense_retriever import DenseRetriever


class SyntheticDummyEncoder:
    """Mock dense encoder generating deterministic embeddings for testing."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        **kwargs: Any,
    ) -> np.ndarray:
        vecs = []
        for t in texts:
            val = float(len(t) % 10) / 10.0 + 0.1
            v = np.full(self.dim, val, dtype=np.float32)
            char_factor = (sum(ord(c) for c in t) % 100) / 1000.0
            v[0] += char_factor
            if normalize_embeddings:
                norm = np.linalg.norm(v)
                v = v / (norm if norm > 0 else 1.0)
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


@pytest.fixture
def synthetic_corpus_dir(tmp_path: Path) -> Path:
    """Create a synthetic multilingual corpus directory with VI, EN, ZH documents and chunks."""
    corpus_dir = tmp_path / "dataset"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    # 2 VI, 2 EN, 2 ZH documents
    documents = [
        {"doc_id": "doc_vi_1", "language": "vi", "title": "Bệnh đái tháo đường", "text": "Bệnh đái tháo đường tuýp 2 gây tăng đường huyết mãn tính và rối loạn chuyển hóa insulin."},
        {"doc_id": "doc_vi_2", "language": "vi", "title": "Bệnh tăng huyết áp", "text": "Tăng huyết áp là yếu tố nguy cơ tim mạch hàng đầu gây đột quỵ và suy tim."},
        {"doc_id": "doc_en_1", "language": "en", "title": "Type 2 Diabetes Mellitus", "text": "Type 2 diabetes mellitus is characterized by hyperglycemia and progressive insulin resistance."},
        {"doc_id": "doc_en_2", "language": "en", "title": "Hypertension Clinical Overview", "text": "Essential hypertension elevates cardiovascular risk and requires ongoing blood pressure monitoring."},
        {"doc_id": "doc_zh_1", "language": "zh", "title": "2型糖尿病临床诊疗", "text": "2型糖尿病患者存在外周胰岛素抵抗和进行性胰岛贝塔细胞功能减退。"},
        {"doc_id": "doc_zh_2", "language": "zh", "title": "原发性高血压规范", "text": "原发性高血压需要长期监测血压水平以降低心脑血管事件的发生率。"},
    ]

    docs_file = corpus_dir / "documents.jsonl"
    with open(docs_file, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    # 2 VI, 2 EN, 2 ZH chunks
    chunks = [
        {"chunk_id": "chunk_vi_1", "doc_id": "doc_vi_1", "language": "vi", "text": "Kiểm soát đường huyết mục tiêu HbA1c dưới 7 phần trăm cho bệnh nhân đái tháo đường."},
        {"chunk_id": "chunk_vi_2", "doc_id": "doc_vi_2", "language": "vi", "text": "Huyết áp mục tiêu thường được khuyến cáo dưới 130 trên 80 mmHg."},
        {"chunk_id": "chunk_en_1", "doc_id": "doc_en_1", "language": "en", "text": "Glycemic control target for HbA1c is generally below seven percent for adults with diabetes."},
        {"chunk_id": "chunk_en_2", "doc_id": "doc_en_2", "language": "en", "text": "Target blood pressure reading is usually below 130 over 80 mmHg for high-risk patients."},
        {"chunk_id": "chunk_zh_1", "doc_id": "doc_zh_1", "language": "zh", "text": "糖尿病患者的糖化血红蛋白控制目标一般建议维持在7百分比以下。"},
        {"chunk_id": "chunk_zh_2", "doc_id": "doc_zh_2", "language": "zh", "text": "大多数高血压患者的推荐降压目标通常为130比80毫米汞柱以下。"},
    ]

    chunks_file = corpus_dir / "chunks.jsonl"
    with open(chunks_file, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # Validation queries
    val_queries = [
        {"query_id": "q_val_1", "text_vi": "Kiểm soát đường huyết mục tiêu HbA1c đái tháo đường như thế nào?"},
        {"query_id": "q_val_2", "text_vi": "Huyết áp mục tiêu khuyến cáo trong điều trị tăng huyết áp là bao nhiêu?"},
    ]
    queries_val_file = corpus_dir / "queries_validation.jsonl"
    with open(queries_val_file, "w", encoding="utf-8") as f:
        for q in val_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Validation qrels (Ground Truth)
    val_qrels = [
        {
            "query_id": "q_val_1",
            "relevant_doc_ids": ["doc_vi_1", "doc_en_1", "doc_zh_1"],
            "relevant_chunk_ids": ["chunk_vi_1", "chunk_en_1", "chunk_zh_1"],
        },
        {
            "query_id": "q_val_2",
            "relevant_doc_ids": ["doc_vi_2", "doc_en_2", "doc_zh_2"],
            "relevant_chunk_ids": ["chunk_vi_2", "chunk_en_2", "chunk_zh_2"],
        },
    ]
    qrels_val_file = corpus_dir / "qrels_validation.jsonl"
    with open(qrels_val_file, "w", encoding="utf-8") as f:
        for gt in val_qrels:
            f.write(json.dumps(gt, ensure_ascii=False) + "\n")

    # Test queries (unlabeled - strictly NO ground truth file)
    test_queries = [
        {"query_id": "q_test_1", "text_vi": "Bệnh nhân đái tháo đường cần theo dõi chỉ số HbA1c định kỳ không?"},
        {"query_id": "q_test_2", "text_vi": "Khuyến cáo chỉ số huyết áp an toàn cho người lớn tuổi?"},
    ]
    queries_test_file = corpus_dir / "queries_test.jsonl"
    with open(queries_test_file, "w", encoding="utf-8") as f:
        for q in test_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    return corpus_dir


@pytest.fixture
def mvp_test_config(tmp_path: Path) -> dict[str, Any]:
    """Test configuration avoiding external heavy model downloads."""
    return {
        "run": {
            "experiment_name": "test_e2e_pipeline",
            "seed": 42,
            "output_dir": str(tmp_path / "artifacts"),
        },
        "indexing": {
            "chunking": {
                "mode": "hybrid",
                "auto_strategy": {
                    "type": "sliding_window",
                    "chunk_size": 256,
                    "chunk_overlap": 32,
                    "min_chunk_length": 10,
                },
            },
            "validation": {
                "strict_foreign_keys": True,
                "drop_duplicates": True,
            },
        },
        "query_understanding": {
            "normalize": True,
            "adaptive_routing": {
                "enabled": True,
                "length_threshold": 50,
                "strip_clinical_boilerplate": True,
                "dense_strategy": "head_tail",
                "max_dense_tokens": 128,
            },
            "translation": {
                "enabled": False,
                "provider": "passthrough",
            },
        },
        "retrieval": {
            "candidate_top_k": 20,
            "bm25": {
                "enabled": True,
                "backend": "bm25s",
                "top_k": 20,
                "tokenizers": {
                    "vi": "syllable",
                    "en": "whitespace",
                    "zh": "jieba",
                },
            },
            "dense": {
                "enabled": True,
                "backend": "numpy",
                "device": "cpu",
                "model_name": "synthetic_dummy",
                "top_k": 20,
                "batch_size": 16,
                "normalize_embeddings": True,
                "instruction_prefix": {
                    "query": "",
                    "passage": "",
                },
            },
        },
        "fusion": {
            "method": "weighted_rrf",
            "rrf_k": 60,
            "weights": {
                "vi_bm25": 1.0,
                "vi_dense": 1.0,
                "en_bm25": 1.0,
                "en_dense": 1.0,
                "zh_bm25": 1.0,
                "zh_dense": 1.0,
            },
        },
        "reranker": {
            "enabled": True,
            "model_name": "dummy",
            "document_strategy": "chunk_aggregation",
            "document_top_k": 10,
            "chunk_top_k": 20,
            "batch_size": 16,
            "max_length": 256,
        },
        "joint_scoring": {
            "document": {
                "base_weight": 0.70,
                "best_chunk_weight": 0.20,
                "top_chunks_mean_weight": 0.10,
                "top_m_chunks": 3,
            },
            "chunk": {
                "base_weight": 0.85,
                "parent_document_weight": 0.15,
            },
        },
        "post_processing": {
            "consistency_mode": "enforce_parent_doc",
        },
        "calibration": {
            "metric": "macro_f2",
            "beta": 2.0,
            "policies": [
                "fixed_top_k",
                "global_threshold",
            ],
            "bounds": {
                "documents": {"min": 1, "max": 10},
                "chunks": {"min": 1, "max": 20},
            },
        },
        "submission": {
            "exporter": "internal_jsonl",
        },
    }


def test_full_pipeline_cycle(synthetic_corpus_dir: Path, mvp_test_config: dict[str, Any], tmp_path: Path):
    """Full End-to-End cycle test:
    1. run_index
    2. run_calibrate
    3. run_evaluate
    4. run_predict
    """
    from src.cli.index import run_index
    from src.cli.calibrate import run_calibrate
    from src.cli.evaluate import run_evaluate
    from src.cli.predict import run_predict

    dummy_encoder = SyntheticDummyEncoder()
    dummy_reranker = DummyReranker()

    index_dir = tmp_path / "indexes"
    artifacts_dir = tmp_path / "artifacts"

    # Step 1: Indexing
    index_res = run_index(
        config=mvp_test_config,
        dataset_dir=synthetic_corpus_dir,
        output_dir=index_dir,
        encoder=dummy_encoder,
    )
    assert index_res["num_documents"] == 6
    assert index_res["num_chunks"] == 6
    assert Path(index_res["bm25_index_dir"]).exists()
    assert (Path(index_res["bm25_index_dir"]) / "manifest.json").exists()
    assert Path(index_res["dense_index_dir"]).exists()
    assert (Path(index_res["dense_index_dir"]) / "manifest.json").exists()

    # Step 2: Calibrate on validation split
    calib_res = run_calibrate(
        config=mvp_test_config,
        split="validation",
        dataset_dir=synthetic_corpus_dir,
        output_dir=artifacts_dir,
        index_dir=index_dir,
        encoder=dummy_encoder,
        reranker=dummy_reranker,
    )
    assert "calibration_file" in calib_res
    calibration_file = Path(calib_res["calibration_file"])
    assert calibration_file.exists()
    with open(calibration_file, "r", encoding="utf-8") as f:
        calib_data = json.load(f)
    assert "documents" in calib_data
    assert "chunks" in calib_data
    assert "best_policy" in calib_data["documents"]
    assert "best_policy" in calib_data["chunks"]

    # Step 3: Evaluate on validation split
    eval_res = run_evaluate(
        config=mvp_test_config,
        split="validation",
        dataset_dir=synthetic_corpus_dir,
        output_dir=artifacts_dir,
        index_dir=index_dir,
        calibration_path=calibration_file,
        encoder=dummy_encoder,
        reranker=dummy_reranker,
    )
    assert "documents" in eval_res
    assert "chunks" in eval_res
    assert "macro_f2" in eval_res["documents"]
    assert eval_res["documents"]["macro_f2"] >= 0.0
    assert (artifacts_dir / "metrics.json").exists()
    assert (artifacts_dir / "per_query_metrics.csv").exists()
    assert (artifacts_dir / "summary.json").exists()

    # Step 4: Predict on test split (Unlabeled - Anti-Leakage verification)
    submission_path = artifacts_dir / "submission.jsonl"
    pred_file = run_predict(
        config=mvp_test_config,
        split="test",
        dataset_dir=synthetic_corpus_dir,
        calibration_path=calibration_file,
        output_file=submission_path,
        exporter_type="internal_jsonl",
        index_dir=index_dir,
        encoder=dummy_encoder,
        reranker=dummy_reranker,
    )
    assert pred_file == submission_path
    assert pred_file.exists()

    # Verify submission file contents
    with open(pred_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    assert len(lines) == 2
    qids = {line["query_id"] for line in lines}
    assert qids == {"q_test_1", "q_test_2"}
    for line in lines:
        assert "document_ids" in line
        assert "chunk_ids" in line
        assert "document_scores" in line
        assert "chunk_scores" in line
        assert isinstance(line["document_ids"], list)
        assert isinstance(line["chunk_ids"], list)
        assert isinstance(line["document_scores"], dict)
        assert isinstance(line["chunk_scores"], dict)


def test_cli_help_flags():
    """Verify that all 4 CLI entrypoints respond to --help properly."""
    entrypoints = [
        "src.cli.index",
        "src.cli.evaluate",
        "src.cli.calibrate",
        "src.cli.predict",
    ]
    for ep in entrypoints:
        result = subprocess.run(
            [sys.executable, "-m", ep, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Failed --help on {ep}: {result.stderr}"
        assert "usage:" in result.stdout.lower()
