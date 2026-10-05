from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import time
import tracemalloc
from typing import Any, Sequence

from src.cli.common import (
    build_query_components,
    build_reranker,
    build_retrievers,
    load_corpus_and_texts,
    resolve_index_dirs,
)
from src.data.dataset_adapter import JsonlDatasetAdapter
from src.data.schemas import FinalPrediction, QueryRecord, ScoredHit
from src.data.source_store import SnapshotSourceStore
from src.data.vibiomir_adapter import ViBioMIRAdapter, load_official_queries
from src.data.vibiomir_contracts import OfficialQuery
from src.evaluation.calibrator import CutoffCalibrator
from src.fusion.weighted_rrf import fuse_stream_results
from src.retrieval.orchestrator import RetrievalOrchestrator
from src.scoring.consistency import apply_structural_consistency
from src.scoring.joint_scoring import JointScorer
from src.submission.base import create_exporter
from src.submission.validators import validate_zip
from src.submission.vibiomir_json_zip import VibiomirExporter
from src.utils.checkpoint import InferenceCheckpointManager, compute_run_fingerprint
from src.utils.config import load_config

logger = logging.getLogger(__name__)


def _run_pipeline_single_query(
    query: QueryRecord,
    config: dict[str, Any],
    orchestrator: RetrievalOrchestrator,
    router: Any,
    query_builder: Any,
    reranker: Any,
    joint_scorer: JointScorer,
    doc_texts: dict[str, str],
    chunk_texts: dict[str, str],
    chunk_to_doc: dict[str, str],
    calibrator: CutoffCalibrator | None,
    calib_data: dict[str, Any] | None,
    consistency_mode: str,
    default_doc_top_k: int,
    default_chunk_top_k: int,
) -> tuple[FinalPrediction, dict[str, Any]]:
    """Executes full retrieval, fusion, reranking, scoring, and consistency for a single query."""
    retrieval_cfg = config.get("retrieval", {})
    candidate_top_k = retrieval_cfg.get("candidate_top_k", 200)

    fusion_cfg = config.get("fusion", {})
    fusion_weights = fusion_cfg.get("weights")
    rrf_k = fusion_cfg.get("rrf_k", 60)

    reranker_cfg = config.get("reranker", {})
    reranker_enabled = reranker_cfg.get("enabled", True) and reranker is not None
    doc_strategy = reranker_cfg.get("document_strategy", "chunk_aggregation")
    doc_top_k = reranker_cfg.get("document_top_k", 100)
    chunk_top_k = reranker_cfg.get("chunk_top_k", 200)
    max_length = reranker_cfg.get("max_length", 512)

    qid = query.query_id
    processed = router.route(query)
    variants = query_builder.build(processed)

    # 1. Multi-stream candidate retrieval
    stream_results = orchestrator.retrieve_all_streams(variants, top_k=candidate_top_k)

    # 2. Weighted RRF fusion
    fused_docs, fused_chunks = fuse_stream_results(
        stream_results=stream_results,
        weights=fusion_weights,
        rrf_k=rrf_k,
        top_k_docs=doc_top_k,
        top_k_chunks=chunk_top_k,
    )

    # 3. Reranking
    if reranker_enabled:
        scored_chunks = reranker.rerank_chunks(
            query_variants=variants,
            candidates=fused_chunks,
            chunk_texts=chunk_texts,
            top_k=chunk_top_k,
        )
        if doc_strategy == "chunk_aggregation":
            scored_docs = reranker.rerank_documents_via_chunk_aggregation(
                doc_candidates=fused_docs,
                scored_chunks=scored_chunks,
                top_k=doc_top_k,
            )
        else:
            scored_docs = reranker.rerank_documents_direct(
                query_variants=variants,
                candidates=fused_docs,
                doc_texts=doc_texts,
                max_length=max_length,
                top_k=doc_top_k,
            )
    else:
        scored_docs = [
            ScoredHit(
                item_id=h.item_id,
                item_type="document",
                doc_id=h.doc_id,
                language=h.language,
                rerank_score=h.fusion_score,
                normalized_score=h.fusion_score,
                raw_fusion_score=h.fusion_score,
            )
            for h in fused_docs
        ]
        scored_chunks = [
            ScoredHit(
                item_id=h.item_id,
                item_type="chunk",
                doc_id=h.doc_id,
                language=h.language,
                rerank_score=h.fusion_score,
                normalized_score=h.fusion_score,
                raw_fusion_score=h.fusion_score,
            )
            for h in fused_chunks
        ]

    # 4. Joint Scoring
    joint_docs, joint_chunks = joint_scorer.score(scored_docs, scored_chunks)

    # 5. Cutoff Filtering
    output_cfg = config.get("output", {})
    fallback_policy = (
        output_cfg.get("fallback_policy")
        or config.get("calibration", {}).get("fallback_policy")
    )

    if fallback_policy == "sparse_rank_caps":
        doc_cap = int(output_cfg.get("document_cap", 10))
        chunk_cap = int(output_cfg.get("evidence_cap", 20))
        chunk_per_doc_cap = int(output_cfg.get("evidence_cap_per_doc", 2))

        # 1. Filter chunks with score > 0
        pos_chunks = [
            c for c in joint_chunks
            if getattr(c, "raw_fusion_score", getattr(c, "rerank_score", getattr(c, "normalized_score", 0.0))) > 0.0
        ]
        selected_chunks = []
        chunks_per_doc: dict[str, int] = {}
        for c in pos_chunks:
            c_did = str(c.doc_id)
            if chunks_per_doc.get(c_did, 0) < chunk_per_doc_cap:
                selected_chunks.append(c)
                chunks_per_doc[c_did] = chunks_per_doc.get(c_did, 0) + 1
                if len(selected_chunks) >= chunk_cap:
                    break

        # 2. Filter docs with score > 0, prioritizing docs with evidence
        pos_docs = [
            d for d in joint_docs
            if getattr(d, "raw_fusion_score", getattr(d, "rerank_score", getattr(d, "normalized_score", 0.0))) > 0.0
        ]
        evidence_doc_ids = set(chunks_per_doc.keys())
        docs_with_evidence = [d for d in pos_docs if str(d.doc_id) in evidence_doc_ids]
        other_docs = [d for d in pos_docs if str(d.doc_id) not in evidence_doc_ids]
        selected_docs = (docs_with_evidence + other_docs)[:doc_cap]

        final_docs = selected_docs
        final_chunks = selected_chunks
    elif calibrator is not None and calib_data is not None:
        doc_state = calib_data.get("documents", calib_data)
        chunk_state = calib_data.get("chunks", calib_data)
        selected_docs = calibrator.filter_items(joint_docs, doc_state)
        selected_chunks = calibrator.filter_items(joint_chunks, chunk_state)
        final_docs, final_chunks = apply_structural_consistency(
            selected_docs=selected_docs,
            selected_chunks=selected_chunks,
            chunk_to_doc=chunk_to_doc,
            mode=consistency_mode,
            available_docs=joint_docs,
        )
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

    pred = FinalPrediction(
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

    debug_trace = {
        "query_id": qid,
        "query_text": query.text_vi,
        "variants": {"vi": variants.vi, "en": variants.en, "zh": variants.zh},
        "candidate_docs_count": len(fused_docs),
        "candidate_chunks_count": len(fused_chunks),
        "selected_docs_count": len(final_docs),
        "selected_chunks_count": len(final_chunks),
        "final_doc_ids": pred.document_ids,
        "final_chunk_ids": pred.chunk_ids,
    }

    return pred, debug_trace


def run_predict(
    config: dict[str, Any],
    split: str = "test",
    dataset_dir: Path | str | None = None,
    snapshot_dir: Path | str | None = None,
    calibration_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    output_file: Path | str | None = None,
    exporter_type: str = "internal_jsonl",
    index_dir: Path | str | None = None,
    reranker: Any = None,
    encoder: Any = None,
    translator: Any = None,
    resume: bool = True,
    export_submission: bool = False,
) -> Path:
    """Generate and export final retrieval predictions with per-query checkpoints and error blocking."""
    tracemalloc.start()
    try:
        return _run_predict_core(
            config=config,
            split=split,
            dataset_dir=dataset_dir,
            snapshot_dir=snapshot_dir,
            calibration_path=calibration_path,
            output_dir=output_dir,
            output_file=output_file,
            exporter_type=exporter_type,
            index_dir=index_dir,
            reranker=reranker,
            encoder=encoder,
            translator=translator,
            resume=resume,
            export_submission=export_submission,
        )
    finally:
        if tracemalloc.is_tracing():
            tracemalloc.stop()


def _run_predict_core(
    config: dict[str, Any],
    split: str = "test",
    dataset_dir: Path | str | None = None,
    snapshot_dir: Path | str | None = None,
    calibration_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    output_file: Path | str | None = None,
    exporter_type: str = "internal_jsonl",
    index_dir: Path | str | None = None,
    reranker: Any = None,
    encoder: Any = None,
    translator: Any = None,
    resume: bool = True,
    export_submission: bool = False,
) -> Path:
    """Core prediction pipeline logic."""
    t_start = time.time()

    # Determine destination directory
    if output_dir is not None:
        out_dir = Path(output_dir)
    elif output_file is not None:
        out_dir = Path(output_file).parent
    else:
        run_cfg = config.get("run", {})
        base_out = run_cfg.get("output_dir", "artifacts/experiments")
        exp_name = run_cfg.get("experiment_name", "vibiomir_phase1_A")
        out_dir = Path(base_out) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load queries and corpus/snapshot
    queries: list[QueryRecord] = []
    official_queries: list[OfficialQuery] = []
    doc_texts: dict[str, str] = {}
    chunk_texts: dict[str, str] = {}
    chunk_to_doc: dict[str, str] = {}
    snapshot_membership = None
    store: SnapshotSourceStore | None = None

    if snapshot_dir is not None:
        snap_p = Path(snapshot_dir)
        if not snap_p.exists():
            raise FileNotFoundError(f"Snapshot directory not found: {snap_p}")
        store = SnapshotSourceStore(snap_p)
        adapter = ViBioMIRAdapter(snap_p)
        snapshot_membership = adapter.manifest.source_membership_sha256

        # Load queries from official query.parquet
        query_path_str = config.get("dataset", {}).get("query_path", "data/official/query.parquet")
        q_path = Path(query_path_str)
        if not q_path.exists() and dataset_dir is not None:
            candidate_q = Path(dataset_dir) / "query.parquet"
            if candidate_q.exists():
                q_path = candidate_q

        expected_count = config.get("dataset", {}).get("expected_query_count", None)
        official_queries = list(load_official_queries(q_path, expected_count=expected_count))
        queries = [QueryRecord(query_id=str(q.query_id), text_vi=q.query) for q in official_queries]

        doc_texts = {str(d.doc_id): f"{d.title} {d.text}".strip() if d.title else d.text for d in adapter.iter_documents()}
        chunk_texts = {c.chunk_id: c.text for c in adapter.iter_chunks()}
        chunk_to_doc = {c.chunk_id: str(c.doc_id) for c in adapter.iter_chunks()}

    else:
        if dataset_dir is None:
            raise ValueError("Either snapshot_dir or dataset_dir must be specified")
        ds_path = Path(dataset_dir)
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset directory not found: {ds_path}")

        ds_adapter = JsonlDatasetAdapter(ds_path)
        queries = ds_adapter.load_queries(split)
        official_queries = []
        for q in queries:
            if str(q.query_id).isdigit():
                official_queries.append(OfficialQuery(query_id=int(q.query_id), query=q.text_vi))

        documents, chunks, doc_texts, chunk_texts, chunk_to_doc = load_corpus_and_texts(
            dataset_dir=ds_path,
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

    # 3. Load calibration if provided
    calib_data = None
    if calibration_path:
        calib_file = Path(calibration_path)
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
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

    # 4. Compute run fingerprint
    index_manifests = []
    if (bm25_dir / "manifest.json").exists():
        with open(bm25_dir / "manifest.json", "r", encoding="utf-8") as f:
            index_manifests.append(json.load(f))
    if (dense_dir / "manifest.json").exists():
        with open(dense_dir / "manifest.json", "r", encoding="utf-8") as f:
            index_manifests.append(json.load(f))

    model_revisions = {
        "dense": str(type(encoder).__name__ if encoder else config.get("retrieval", {}).get("dense", {}).get("model_name")),
        "reranker": str(type(reranker_inst).__name__ if reranker_inst else config.get("reranker", {}).get("model_name")),
    }

    query_identifiers = [int(q.query_id) if str(q.query_id).isdigit() else str(q.query_id) for q in queries]
    run_fp = compute_run_fingerprint(
        query_manifest_or_ids=query_identifiers,
        snapshot_manifest_or_membership=snapshot_membership,
        index_manifests=index_manifests,
        model_revisions=model_revisions,
        config=config,
    )

    # 5. Initialize Checkpoint Manager
    ckpt_mgr = InferenceCheckpointManager(
        output_dir=out_dir,
        run_fingerprint=run_fp,
        resume=resume,
    )
    completed_qids = ckpt_mgr.get_completed_query_ids()

    # 6. Execute per-query inference with checkpointing
    debug_path = out_dir / "retrieval_debug.jsonl"
    with open(debug_path, "a", encoding="utf-8") as debug_file:
        for query in queries:
            qid_val = int(query.query_id) if str(query.query_id).isdigit() else str(query.query_id)
            if qid_val in completed_qids:
                logger.info("Skipping already completed query %s", qid_val)
                continue

            try:
                pred, trace = _run_pipeline_single_query(
                    query=query,
                    config=config,
                    orchestrator=orchestrator,
                    router=router,
                    query_builder=query_builder,
                    reranker=reranker_inst,
                    joint_scorer=joint_scorer,
                    doc_texts=doc_texts,
                    chunk_texts=chunk_texts,
                    chunk_to_doc=chunk_to_doc,
                    calibrator=calibrator,
                    calib_data=calib_data,
                    consistency_mode=consistency_mode,
                    default_doc_top_k=default_doc_top_k,
                    default_chunk_top_k=default_chunk_top_k,
                )
                ckpt_mgr.save_checkpoint(query_id=qid_val, status="completed", prediction=pred)
                debug_file.write(json.dumps(trace, ensure_ascii=False) + "\n")
                debug_file.flush()

            except Exception as exc:
                logger.error("Inference failed for query %s: %s", qid_val, exc)
                ckpt_mgr.save_checkpoint(query_id=qid_val, status="failed", error=str(exc))
                # Never silently convert to empty predictions; keep failure recorded

    # 7. Validate all queries and block export if any failed
    all_predictions = ckpt_mgr.validate_and_get_all_predictions(required_query_ids=query_identifiers)

    # 8. Save predictions.jsonl
    predictions_path = out_dir / "predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as f:
        for p in all_predictions:
            f.write(
                json.dumps(
                    {
                        "query_id": p.query_id,
                        "document_ids": p.document_ids,
                        "chunk_ids": p.chunk_ids,
                        "document_scores": p.document_scores,
                        "chunk_scores": p.chunk_scores,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    # 9. Save run_manifest.json
    total_time = time.time() - t_start
    _, peak_bytes = tracemalloc.get_traced_memory()
    peak_ram_mb = round(peak_bytes / (1024 * 1024), 2)

    manifest_data = {
        "run_fingerprint": run_fp,
        "seed": config.get("run", {}).get("seed", 42),
        "models": model_revisions,
        "runtime_seconds": round(total_time, 2),
        "peak_ram_mb": peak_ram_mb,
        "gpu_vram_mb": 0.0,
        "total_queries": len(all_predictions),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "snapshot_membership_sha256": snapshot_membership,
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    # 10. Submission Export
    effective_exporter = config.get("submission", {}).get("exporter", exporter_type)
    if store is not None and (export_submission or effective_exporter == "vibiomir_json_zip"):
        sub_cfg = config.get("submission", {})
        vibiomir_exporter = VibiomirExporter(
            json_filename=sub_cfg.get("json_filename", "results.json"),
            zip_filename=sub_cfg.get("zip_filename", "submission_A.zip"),
            emit_chunk_order=sub_cfg.get("emit_chunk_order", True),
            provenance_check=sub_cfg.get("provenance_check", "strict"),
            expected_count=len(official_queries),
        )
        export_result = vibiomir_exporter.export(
            predictions=all_predictions,
            queries=official_queries,
            store=store,
            output_dir=out_dir,
        )

        # Validate ZIP
        val_report = validate_zip(
            export_result.zip_path,
            queries=official_queries,
            store=store,
            expected_count=len(official_queries),
        )
        val_data = {
            "valid": val_report.valid,
            "query_count": val_report.query_count,
            "document_count": val_report.document_count,
            "chunk_count": val_report.chunk_count,
            "errors": list(val_report.errors),
            "warnings": list(val_report.warnings),
            "snapshot_id": val_report.snapshot_id,
        }
        with open(out_dir / "validation_report.json", "w", encoding="utf-8") as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)

        if not val_report.valid:
            raise RuntimeError(f"Submission ZIP failed validation: {val_report.errors}")

        return export_result.zip_path

    if output_file is not None:
        target_out = Path(output_file)
    else:
        ext = "csv" if exporter_type == "competition_csv" else "jsonl"
        target_out = out_dir / f"submission.{ext}"

    target_out.parent.mkdir(parents=True, exist_ok=True)
    exporter = create_exporter(exporter_type)
    return exporter.export(all_predictions, target_out)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Multilingual Medical Retrieval Prediction CLI."
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
        "--output-dir",
        default=None,
        help="Output directory for predictions and checkpoints",
    )
    parser.add_argument(
        "--exporter",
        default="internal_jsonl",
        choices=["internal_jsonl", "competition_csv", "vibiomir_json_zip"],
        help="Exporter format",
    )
    parser.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing pre-built BM25 and Dense indexes",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume inference from existing query checkpoints",
    )
    parser.add_argument(
        "--export-submission",
        action="store_true",
        help="Export official ViBioMIR results.json and ZIP archive",
    )

    parsed = parser.parse_args(args)
    if not parsed.dataset and not parsed.snapshot:
        parser.error("At least one of --dataset or --snapshot must be specified")

    cfg = load_config(parsed.config)
    res_path = run_predict(
        config=cfg,
        split=parsed.split,
        dataset_dir=parsed.dataset,
        snapshot_dir=parsed.snapshot,
        calibration_path=parsed.calibration_path,
        output_dir=parsed.output_dir,
        output_file=parsed.output_file,
        exporter_type=parsed.exporter,
        index_dir=parsed.index_dir,
        resume=parsed.resume,
        export_submission=parsed.export_submission,
    )
    print(f"Predictions and artifacts successfully generated at: {res_path}")


if __name__ == "__main__":
    main()
