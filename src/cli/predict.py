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
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.scoring.consistency import apply_structural_consistency
from src.scoring.joint_scoring import JointScorer
from src.submission.base import create_exporter
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def run_predict(
    config: dict[str, Any],
    split: str = "test",
    dataset_dir: Path | str = "",
    calibration_path: Path | str | None = None,
    output_file: Path | str | None = None,
    exporter_type: str = "internal_jsonl",
    index_dir: Path | str | None = None,
    reranker: Any = None,
    encoder: Any = None,
    translator: Any = None,
) -> Path:
    """Generate and export final retrieval predictions for unlabeled test sets.

    ANTI-LEAKAGE PRINCIPLE:
    This function strictly operates without loading ground truth. Cutoff thresholds and
    top-k parameters must be derived entirely from pre-computed validation calibration.

    Args:
        config: Configuration dictionary.
        split: Dataset split to predict on (default: "test").
        dataset_dir: Directory containing test queries and corpus documents/chunks.
        calibration_path: Optional path to calibration.json.
        output_file: Target path where exported submission file will be saved.
        exporter_type: Submission format ('internal_jsonl' or 'competition_csv').
        index_dir: Directory containing pre-built BM25 and Dense indexes.
        reranker: Optional pre-instantiated reranker override.
        encoder: Optional pre-instantiated dense encoder override.
        translator: Optional pre-instantiated translator override.

    Returns:
        Path to the exported prediction submission file.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    # 1. Anti-Leakage: Load queries ONLY (never load ground truth on test split)
    adapter = JsonlDatasetAdapter(dataset_path)
    queries = adapter.load_queries(split)

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

    # 3. Load calibration.json if provided
    calib_data = None
    calib_path = Path(calibration_path) if calibration_path else None
    if calib_path and calib_path.exists():
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_data = json.load(f)

    calibrator: CutoffCalibrator | None = None
    if calib_data:
        calibrator = CutoffCalibrator(beta=config.get("calibration", {}).get("beta", 2.0))

    consistency_mode = config.get("post_processing", {}).get("consistency_mode", "independent")
    reranker_cfg = config.get("reranker", {})
    default_doc_top_k = reranker_cfg.get("document_top_k", 10)
    default_chunk_top_k = reranker_cfg.get("chunk_top_k", 20)

    predictions: list[FinalPrediction] = []

    for qid, (joint_docs, joint_chunks) in query_scores.items():
        # 4. Filter documents and chunks with calibrated cutoff
        if calibrator is not None and calib_data is not None:
            doc_state = calib_data.get("documents", calib_data)
            chunk_state = calib_data.get("chunks", calib_data)
            selected_docs = calibrator.filter_items(joint_docs, doc_state)
            selected_chunks = calibrator.filter_items(joint_chunks, chunk_state)
        else:
            selected_docs = joint_docs[:default_doc_top_k]
            selected_chunks = joint_chunks[:default_chunk_top_k]

        # 5. Apply structural consistency
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

    # 6. Determine output file path and exporter
    if output_file is not None:
        target_output = Path(output_file)
    else:
        out_dir = Path(config.get("run", {}).get("output_dir", "artifacts"))
        ext = "csv" if exporter_type == "competition_csv" else "jsonl"
        target_output = out_dir / f"submission.{ext}"

    target_output.parent.mkdir(parents=True, exist_ok=True)
    exporter = create_exporter(exporter_type)
    exported_path = exporter.export(predictions, target_output)

    logger.info("Exported %d predictions to %s", len(predictions), exported_path)
    return exported_path


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual Medical Retrieval Prediction CLI."
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
        default="test",
        help="Dataset split to predict (default: test)",
    )
    parser.add_argument(
        "--calibration-path",
        default=None,
        help="Path to pre-fitted calibration.json file",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        default=None,
        help="Destination path for submission file",
    )
    parser.add_argument(
        "--exporter",
        default="internal_jsonl",
        choices=["internal_jsonl", "competition_csv"],
        help="Exporter format (default: internal_jsonl)",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing pre-built BM25 and Dense indexes",
    )

    parsed = parser.parse_args(args)
    cfg = load_config(parsed.config)
    res_path = run_predict(
        config=cfg,
        split=parsed.split,
        dataset_dir=parsed.dataset,
        calibration_path=parsed.calibration_path,
        output_file=parsed.output_file,
        exporter_type=parsed.exporter,
        index_dir=parsed.index_dir,
    )
    print(f"Predictions exported successfully to: {res_path}")


if __name__ == "__main__":
    main()
