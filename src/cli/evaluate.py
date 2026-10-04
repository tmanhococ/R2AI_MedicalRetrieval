from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
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
from src.data.schemas import FinalPrediction, QueryRecord, ScoredHit
from src.data.source_store import SnapshotSourceStore
from src.data.vibiomir_adapter import ViBioMIRAdapter, load_official_queries
from src.evaluation.calibrator import CutoffCalibrator
from src.evaluation.dev_records import FORBIDDEN_SPLITS, load_dev_records
from src.evaluation.error_analysis import generate_error_analysis_report
from src.evaluation.metrics import evaluate_predictions
from src.evaluation.vibiomir_metrics import (
    PROVISIONAL_SPEC_V1_PROFILE,
    evaluate_dataset,
)
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.scoring.consistency import apply_structural_consistency
from src.scoring.joint_scoring import JointScorer
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def run_evaluate(
    config: dict[str, Any],
    split: str = "validation",
    dataset_dir: Path | str | None = None,
    snapshot_dir: Path | str | None = None,
    dev_records_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
    calibration_path: Path | str | None = None,
    reranker: Any = None,
    encoder: Any = None,
    translator: Any = None,
) -> dict[str, Any]:
    """Execute end-to-end retrieval evaluation on a split with Macro F2 calculation and error analysis.

    ANTI-LEAKAGE PRINCIPLE:
    Rejects evaluation on test splits without ground truth labels.
    """
    clean_split = split.lower().strip()
    if clean_split in FORBIDDEN_SPLITS:
        raise ValueError(
            f"Cannot run evaluation on '{split}' split without ground truth labels. "
            f"Test set does not contain ground truth labels; use src.cli.predict to generate predictions."
        )

    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        out_dir = Path(config.get("run", {}).get("output_dir", "artifacts"))
    out_dir.mkdir(parents=True, exist_ok=True)

    queries: list[QueryRecord] = []
    ground_truths: Any = None
    dev_records: Any = None
    doc_texts: dict[str, str] = {}
    chunk_texts: dict[str, str] = {}
    chunk_to_doc: dict[str, str] = {}

    if snapshot_dir is not None:
        snap_path = Path(snapshot_dir)
        if not snap_path.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snap_path}")

        store = SnapshotSourceStore(snap_path)
        adapter = ViBioMIRAdapter(snap_path)

        # Official queries namespace check
        query_path_str = config.get("dataset", {}).get("query_path", "data/official/query.parquet")
        official_queries = load_official_queries(Path(query_path_str), expected_count=None)
        official_query_ids = {q.query_id for q in official_queries}

        # Dev records loading
        dev_path_str = dev_records_path or config.get("dataset", {}).get("dev_records_path", "data/dev/dev_qrels.jsonl")
        dev_file = Path(dev_path_str)
        if not dev_file.exists() and dataset_dir:
            cand = Path(dataset_dir) / "dev_qrels.jsonl"
            if cand.exists():
                dev_file = cand

        dev_records = load_dev_records(dev_file, official_query_ids=official_query_ids, store=store)
        queries = [
            QueryRecord(
                query_id=str(r.query_id),
                text_vi=r.metadata.query_text if r.metadata else str(r.query_id),
            )
            for r in dev_records
        ]

        doc_texts = {str(d.doc_id): f"{d.title} {d.text}".strip() if d.title else d.text for d in adapter.iter_documents()}
        chunk_texts = {c.chunk_id: c.text for c in adapter.iter_chunks()}
        chunk_to_doc = {c.chunk_id: str(c.doc_id) for c in adapter.iter_chunks()}

    else:
        if dataset_dir is None:
            raise ValueError("Either dataset_dir or snapshot_dir must be provided")
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

        adapter_legacy = JsonlDatasetAdapter(dataset_path)
        queries = adapter_legacy.load_queries(split)
        ground_truths = adapter_legacy.load_ground_truth(split)

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
    calib_cfg = config.get("calibration", {})
    fixed_k = calib_cfg.get("fixed_top_k", {})
    calib_bounds = calib_cfg.get("bounds", {})
    default_doc_top_k = int(fixed_k.get("documents") or calib_bounds.get("documents", {}).get("min") or 5)
    default_chunk_top_k = int(fixed_k.get("chunks") or calib_bounds.get("chunks", {}).get("min") or 10)

    predictions: list[FinalPrediction] = []

    for qid, (joint_docs, joint_chunks) in query_scores.items():
        if calibrator is not None and calib_data is not None:
            doc_state = calib_data.get("documents", calib_data)
            chunk_state = calib_data.get("chunks", calib_data)
            selected_docs = calibrator.filter_items(joint_docs, doc_state)
            selected_chunks = calibrator.filter_items(joint_chunks, chunk_state)
        else:
            selected_docs = joint_docs[:default_doc_top_k]
            selected_chunks = joint_chunks[:default_chunk_top_k]

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

    # 10. Compute metrics
    if snapshot_dir is not None and dev_records is not None:
        sub_rows = []
        for p in predictions:
            row_chunks = []
            for c_id in p.chunk_ids:
                doc_id_val = int(chunk_to_doc.get(c_id, 0))
                c_text = chunk_texts.get(c_id, "")
                row_chunks.append({"doc_id": doc_id_val, "chunk_text": c_text})
            sub_rows.append(
                {
                    "id": int(p.query_id),
                    "relevant_docs": [int(d) for d in p.document_ids],
                    "relevant_chunks": row_chunks,
                }
            )

        summary = evaluate_dataset(
            predictions=sub_rows,
            gold=dev_records,
            query_ids=[r.query_id for r in dev_records],
            profile=PROVISIONAL_SPEC_V1_PROFILE,
        )

        metrics = {
            "documents": {
                "macro_precision": summary.macro_documents.precision,
                "macro_recall": summary.macro_documents.recall,
                "macro_f2": summary.macro_documents.f2,
            },
            "chunks": {
                "macro_precision": summary.macro_chunks.precision,
                "macro_recall": summary.macro_chunks.recall,
                "macro_f2": summary.macro_chunks.f2,
            },
            "inferred_final_mean": summary.inferred_final_mean,
            "query_count": summary.query_count,
            "profile_fingerprint": summary.profile_fingerprint,
        }

        # Write per_query_metrics.csv
        per_query_csv = out_dir / "per_query_metrics.csv"
        with open(per_query_csv, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["query_id", "doc_p", "doc_r", "doc_f2", "chunk_p", "chunk_r", "chunk_f2"])
            for qm in summary.per_query_metrics:
                writer.writerow([
                    qm.query_id,
                    round(qm.documents.precision, 4),
                    round(qm.documents.recall, 4),
                    round(qm.documents.f2, 4),
                    round(qm.chunks.precision, 4),
                    round(qm.chunks.recall, 4),
                    round(qm.chunks.f2, 4),
                ])

    else:
        beta = float(config.get("calibration", {}).get("beta", 2.0))
        metrics = evaluate_predictions(
            predictions=predictions,
            ground_truth=ground_truths,
            beta=beta,
        )
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
        default="configs/vibiomir_phase1.yaml" if Path("configs/vibiomir_phase1.yaml").exists() else "configs/r2ai_mvp.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default=None,
        help="Path to dataset directory",
    )
    parser.add_argument(
        "-s",
        "--snapshot",
        default=None,
        help="Path to frozen snapshot directory",
    )
    parser.add_argument(
        "--dev-records-path",
        default=None,
        help="Path to dev_qrels.jsonl ground truth file",
    )
    parser.add_argument(
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
    if not parsed.dataset and not parsed.snapshot:
        parser.error("At least one of --dataset or --snapshot must be specified")

    cfg = load_config(parsed.config)
    res = run_evaluate(
        config=cfg,
        split=parsed.split,
        dataset_dir=parsed.dataset,
        snapshot_dir=parsed.snapshot,
        dev_records_path=parsed.dev_records_path,
        output_dir=parsed.output_dir,
        index_dir=parsed.index_dir,
        calibration_path=parsed.calibration_path,
    )
    print("Evaluation completed successfully:")
    print(f"  Documents Macro F2: {res['documents']['macro_f2']:.4f}")
    print(f"  Chunks Macro F2:    {res['chunks']['macro_f2']:.4f}")


if __name__ == "__main__":
    main()
