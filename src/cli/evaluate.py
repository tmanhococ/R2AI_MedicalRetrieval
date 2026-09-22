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
from src.data.schemas import FinalPrediction, ScoredHit
from src.evaluation.calibrator import CutoffCalibrator
from src.evaluation.error_analysis import generate_error_analysis_report
from src.evaluation.metrics import evaluate_predictions
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.scoring.consistency import apply_structural_consistency
from src.scoring.joint_scoring import JointScorer
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def run_evaluate(
    config: dict[str, Any],
    split: str = "validation",
    dataset_dir: Path | str = "",
    output_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
    calibration_path: Path | str | None = None,
    reranker: Any = None,
    encoder: Any = None,
    translator: Any = None,
) -> dict[str, Any]:
    """Execute end-to-end retrieval evaluation on a split with Macro F2 calculation and error analysis.

    Args:
        config: Configuration dictionary.
        split: Dataset split to evaluate (default: "validation").
        dataset_dir: Path to dataset directory.
        output_dir: Path where metrics.json and error analysis artifacts will be saved.
        index_dir: Path where BM25 and Dense indexes reside.
        calibration_path: Optional path to calibration.json.
        reranker: Optional pre-instantiated reranker override.
        encoder: Optional pre-instantiated dense encoder override.
        translator: Optional pre-instantiated translator override.

    Returns:
        Dictionary containing overall and per-query evaluation metrics.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(config.get("run", {}).get("output_dir", "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load queries and ground truth for split
    adapter = JsonlDatasetAdapter(dataset_path)
    queries = adapter.load_queries(split)
    ground_truths = adapter.load_ground_truth(split)

    # Load corpus text lookups
    documents, chunks, doc_texts, chunk_texts, chunk_to_doc = load_corpus_and_texts(
        dataset_dir=dataset_path,
        index_dir=index_dir,
        config=config,
    )

    # 2. Setup pipeline components
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

    # Resolve calibration if provided or present in output directory
    calib_data = None
    if calibration_path:
        calib_file = Path(calibration_path)
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_data = json.load(f)
    else:
        calib_file = out_dir / "calibration.json"
        if calib_file.exists():
            with open(calib_file, "r", encoding="utf-8") as f:
                calib_data = json.load(f)

    calibrator: CutoffCalibrator | None = None
    if calib_data:
        calibrator = CutoffCalibrator(beta=config.get("calibration", {}).get("beta", 2.0))

    consistency_mode = config.get("post_processing", {}).get("consistency_mode", "independent")
    calib_bounds = config.get("calibration", {}).get("bounds", {})
    default_doc_top_k = int(calib_bounds.get("documents", {}).get("min", 10))
    default_chunk_top_k = int(calib_bounds.get("chunks", {}).get("min", 20))

    predictions: list[FinalPrediction] = []

    for qid, (joint_docs, joint_chunks) in query_scores.items():
        # 8. Apply cutoff threshold/top-k
        if calibrator is not None and calib_data is not None:
            doc_state = calib_data.get("documents", calib_data)
            chunk_state = calib_data.get("chunks", calib_data)
            selected_docs = calibrator.filter_items(joint_docs, doc_state)
            selected_chunks = calibrator.filter_items(joint_chunks, chunk_state)
        else:
            selected_docs = joint_docs[:default_doc_top_k]
            selected_chunks = joint_chunks[:default_chunk_top_k]

        # 9. Apply structural consistency
        final_docs, final_chunks = apply_structural_consistency(
            selected_docs=selected_docs,
            selected_chunks=selected_chunks,
            chunk_to_doc=chunk_to_doc,
            mode=consistency_mode,
            available_docs=joint_docs,
        )

        predictions.append(
            FinalPrediction(
                query_id=qid,
                document_ids=[d.doc_id if isinstance(d, ScoredHit) else str(d) for d in final_docs],
                chunk_ids=[c.item_id if isinstance(c, ScoredHit) else str(c) for c in final_chunks],
                document_scores={
                    d.doc_id: float(d.normalized_score)
                    for d in final_docs
                    if isinstance(d, ScoredHit)
                },
                chunk_scores={
                    c.item_id: float(c.normalized_score)
                    for c in final_chunks
                    if isinstance(c, ScoredHit)
                },
            )
        )

    # 10. Compute metrics via evaluate_predictions
    beta = float(config.get("calibration", {}).get("beta", 2.0))
    metrics = evaluate_predictions(
        predictions=predictions,
        ground_truth=ground_truths,
        beta=beta,
    )

    # 11. Generate error analysis report
    generate_error_analysis_report(
        predictions=predictions,
        ground_truths=ground_truths,
        output_dir=out_dir,
        queries=queries,
        beta=beta,
    )

    # 12. Save metrics.json
    metrics_file = out_dir / "metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info(
        "Evaluation completed: Doc Macro F2=%.4f, Chunk Macro F2=%.4f",
        metrics["documents"]["macro_f2"],
        metrics["chunks"]["macro_f2"],
    )

    return metrics


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual Medical Retrieval Evaluation CLI."
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
        help="Path to dataset directory",
    )
    parser.add_argument(
        "-s",
        "--split",
        default="validation",
        help="Dataset split to evaluate (default: validation)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Output directory for metrics and error analysis artifacts",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing pre-built BM25 and Dense indexes",
    )
    parser.add_argument(
        "--calibration-path",
        default=None,
        help="Path to calibration.json artifact",
    )

    parsed = parser.parse_args(args)
    cfg = load_config(parsed.config)
    res = run_evaluate(
        config=cfg,
        split=parsed.split,
        dataset_dir=parsed.dataset,
        output_dir=parsed.output_dir,
        index_dir=parsed.index_dir,
        calibration_path=parsed.calibration_path,
    )
    print("Evaluation completed successfully:")
    print(f"  Documents Macro F2: {res['documents']['macro_f2']:.4f}")
    print(f"  Chunks Macro F2:    {res['chunks']['macro_f2']:.4f}")


if __name__ == "__main__":
    main()
