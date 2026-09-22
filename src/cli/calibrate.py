from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from src.cli.common import (
    build_query_components,
    build_reranker,
    build_retrievers,
    load_corpus_and_texts,
    resolve_index_dirs,
    run_pipeline_for_queries,
)
from src.data.dataset_adapter import JsonlDatasetAdapter
from src.evaluation.calibrator import CutoffCalibrator
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.scoring.joint_scoring import JointScorer
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def run_calibrate(
    config: dict[str, Any],
    split: str = "validation",
    dataset_dir: Path | str = "",
    output_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
    reranker: Any = None,
    encoder: Any = None,
    translator: Any = None,
) -> dict[str, Any]:
    """Execute validation set cutoff calibration for document and chunk predictions.

    ANTI-LEAKAGE PRINCIPLE:
    Calibration MUST strictly be conducted on the validation split. Fitting on test data
    violates evaluation integrity.

    Args:
        config: Application configuration dictionary.
        split: Dataset split to use for calibration (default: "validation").
        dataset_dir: Dataset directory containing corpus, queries, and qrels.
        output_dir: Destination directory for calibration.json.
        index_dir: Directory containing pre-built BM25 and Dense indexes.
        reranker: Optional pre-instantiated reranker override.
        encoder: Optional pre-instantiated dense encoder override.
        translator: Optional pre-instantiated translator override.

    Returns:
        Dictionary containing calibration file path and state summaries.
    """
    if split.lower() in ("test", "hidden_test", "submission"):
        raise ValueError(f"Cannot calibrate on split '{split}': anti-leakage violation.")

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(config.get("run", {}).get("output_dir", "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load queries and ground truth for validation split
    adapter = JsonlDatasetAdapter(dataset_path)
    queries = adapter.load_queries(split)
    ground_truths = adapter.load_ground_truth(split)

    # Load corpus text lookups
    documents, chunks, doc_texts, chunk_texts, _ = load_corpus_and_texts(
        dataset_dir=dataset_path,
        index_dir=index_dir,
        config=config,
    )

    # 2. Setup retrieval & query pipeline components
    bm25_dir, dense_dir = resolve_index_dirs(config, index_dir)
    bm25_retriever, dense_retriever = build_retrievers(
        config=config,
        bm25_dir=bm25_dir,
        dense_dir=dense_dir,
        encoder=encoder,
    )

    retrieval_cfg = config.get("retrieval", {})
    orchestrator = RetrievalOrchestrator(
        bm25_retriever=bm25_retriever,
        dense_retriever=dense_retriever,
        default_top_k=retrieval_cfg.get("candidate_top_k", 200),
    )

    router, query_builder = build_query_components(config, translator=translator)
    reranker_inst = build_reranker(config, reranker=reranker)
    joint_scorer = JointScorer.from_config(config)

    # Run retrieval pipeline up to joint scoring
    query_scores = run_pipeline_for_queries(
        queries=queries,
        config=config,
        orchestrator=orchestrator,
        router=router,
        query_builder=query_builder,
        reranker=reranker_inst,
        joint_scorer=joint_scorer,
        doc_texts=doc_texts,
        chunk_texts=chunk_texts,
    )

    # 3. Fit CutoffCalibrator independently for documents and chunks
    calib_cfg = config.get("calibration", {})
    calibrator = CutoffCalibrator(
        policies=calib_cfg.get(
            "policies",
            ["fixed_top_k", "global_threshold", "relative_threshold", "score_gap"],
        ),
        bounds=calib_cfg.get(
            "bounds",
            {"documents": {"min": 1, "max": 50}, "chunks": {"min": 1, "max": 100}},
        ),
        beta=calib_cfg.get("beta", 2.0),
    )

    doc_query_scores = {qid: pair[0] for qid, pair in query_scores.items()}
    chunk_query_scores = {qid: pair[1] for qid, pair in query_scores.items()}

    doc_state = calibrator.fit(doc_query_scores, ground_truths, item_type="document")
    chunk_state = calibrator.fit(chunk_query_scores, ground_truths, item_type="chunk")

    # 4. Save calibration.json
    calibration_file = out_dir / "calibration.json"
    calib_summary = {
        "calibration_file": str(calibration_file),
        "documents": doc_state.to_dict(),
        "chunks": chunk_state.to_dict(),
    }
    with open(calibration_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "documents": doc_state.to_dict(),
                "chunks": chunk_state.to_dict(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info(
        "Calibration saved to %s (Best doc policy: %s, Best chunk policy: %s)",
        calibration_file,
        doc_state.best_policy,
        chunk_state.best_policy,
    )

    return calib_summary


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual Medical Retrieval Cutoff Calibration CLI."
    )
    parser.add_argument(
        "-c",
        "--config",
        default="configs/r2ai_mvp.yaml",
        help="Path to YAML configuration file (default: configs/r2ai_mvp.yaml)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Path to dataset directory containing validation split",
    )
    parser.add_argument(
        "-s",
        "--split",
        default="validation",
        help="Split to use for calibration (default: validation)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory to save calibration.json artifact",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing BM25 and Dense indexes",
    )

    parsed = parser.parse_args(args)
    cfg = load_config(parsed.config)
    res = run_calibrate(
        config=cfg,
        split=parsed.split,
        dataset_dir=parsed.dataset,
        output_dir=parsed.output_dir,
        index_dir=parsed.index_dir,
    )
    print("Calibration completed successfully:")
    print(f"  Artifact: {res['calibration_file']}")
    print(f"  Doc policy:   {res['documents']['best_policy']} (F2: {res['documents']['best_f2']:.4f})")
    print(f"  Chunk policy: {res['chunks']['best_policy']} (F2: {res['chunks']['best_f2']:.4f})")


if __name__ == "__main__":
    main()
